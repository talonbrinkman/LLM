import argparse
from pathlib import Path
import sys
import torch

from src.config import ModelConfig, TrainingConfig
from src.dataset import create_dataloaders, load_text_from_path
from src.engine import Trainer
from src.model import CustomTransformerLM
from src.tokenizer import Tokenizer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Custom LLM causal decoder-only transformer from scratch."
    )
    # Data & Paths
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="Path to text file, directory of .txt files, or comma-separated list of paths.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="model_weights", help="Directory to save weights & configs.")

    # Architecture Hyperparameters
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["small", "medium", "large"],
        help="Architecture preset: small (~7M params, 256 ctx), medium (~30M params, 512 ctx), large (~90M params, 1024 ctx).",
    )
    parser.add_argument("--d-model", type=int, default=256, help="Transformer embedding and hidden dimension.")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of decoder transformer layers.")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--seq-len", type=int, default=512, help="Maximum sequence context window.")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Maximum tokenizer vocabulary size.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of full training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per training step.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Decoupled weight decay coefficient.")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Linear learning rate warmup steps.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient norm threshold.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation dataset fraction.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device.")
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing checkpoint in --checkpoint-dir.")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional explicit path to a checkpoint file (.pt) or folder to resume/fine-tune from.",
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Enable Supervised Fine-Tuning mode: filters to dialogue files and masks user prompt tokens from loss.",
    )
    parser.add_argument(
        "--early-stopping",
        nargs="?",
        const=2,
        type=int,
        default=0,
        metavar="PATIENCE",
        help="Auto-stop training if validation loss does not improve for N consecutive epochs (default: 2 when flag passed, 0 = disabled).",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    sub_paths = [Path(s.strip()) for s in args.data.split(",") if s.strip()]

    for sp in sub_paths:
        if not sp.exists():
            print(f"Error: Specified data path '{sp}' does not exist.")
            print("Please place text files in 'data/' or point to a file via --data <path>.")
            sys.exit(1)

    # Apply architecture preset if specified
    if args.preset == "small":
        args.d_model = 256
        args.n_layers = 6
        args.n_heads = 8
        args.seq_len = 256
        print("Loaded preset 'small': 6 layers, 8 heads, 256 d_model, 256 seq_len (~6.8M params)")
    elif args.preset == "medium":
        args.d_model = 512
        args.n_layers = 8
        args.n_heads = 8
        args.seq_len = 512
        print("Loaded preset 'medium': 8 layers, 8 heads, 512 d_model, 512 seq_len (~30M params)")
    elif args.preset == "large":
        args.d_model = 768
        args.n_layers = 12
        args.n_heads = 12
        args.seq_len = 1024
        print("Loaded preset 'large': 12 layers, 12 heads, 768 d_model, 1024 seq_len (~90M params)")

    print(f"Loading corpus from: {args.data}")
    raw_text = load_text_from_path(args.data, sft_only=args.sft)

    if not raw_text.strip():
        print("Error: No text data found in the specified path.")
        sys.exit(1)

    print(f"Corpus size: {len(raw_text):,} characters")

    # Resolve resume checkpoint & tokenizer paths
    resume_active = args.resume or (args.resume_from is not None)
    resume_source_dir = Path(args.checkpoint_dir)
    ckpt_candidates = []

    if args.resume_from is not None:
        rf_path = Path(args.resume_from)
        if rf_path.is_dir():
            resume_source_dir = rf_path
            ckpt_candidates = [rf_path / "checkpoint.pt", rf_path / "best_model.pt"]
        else:
            resume_source_dir = rf_path.parent
            ckpt_candidates = [rf_path]
    elif args.resume:
        ckpt_candidates = [
            Path(args.checkpoint_dir) / "checkpoint.pt",
            Path(args.checkpoint_dir) / "best_model.pt",
        ]

    # Fit or Resume Tokenizer
    existing_tok_path = resume_source_dir / "tokenizer.json"
    if resume_active and existing_tok_path.exists():
        print(f"Resuming with existing tokenizer from: {existing_tok_path}")
        tokenizer = Tokenizer.load(existing_tok_path)
    else:
        print("Fitting Byte-Level BPE tokenizer on corpus...")
        tokenizer = Tokenizer(vocab_size=args.vocab_size)
        tokenizer.fit([raw_text])
        print(
            f"Vocabulary built: {len(tokenizer):,} unique tokens "
            f"(pad={tokenizer.pad_id}, eos={tokenizer.eos_id}, eot={tokenizer.eot_id})"
        )

    token_ids = tokenizer.encode(raw_text)
    print(f"Tokenized corpus length: {len(token_ids):,} tokens")

    # Check for existing checkpoint when resuming
    resumed_ckpt = None
    model_config = None

    if resume_active:
        for cand in ckpt_candidates:
            if cand.exists():
                print(f"Loading checkpoint metadata from: {cand}")
                resumed_ckpt = torch.load(cand, map_location="cpu", weights_only=False)
                if isinstance(resumed_ckpt, dict) and "model_config" in resumed_ckpt:
                    model_config = ModelConfig.from_dict(resumed_ckpt["model_config"])
                    args.seq_len = model_config.max_seq_len
                    args.d_model = model_config.d_model
                    args.n_layers = model_config.n_layers
                    args.n_heads = model_config.n_heads
                    print(f"Restored model architecture: {args.n_layers} layers, {args.n_heads} heads, {args.d_model} dim, seq_len={args.seq_len}")
                break
        if resumed_ckpt is None:
            print(f"Warning: Resume requested but no checkpoint found at '{resume_source_dir}'. Training from scratch.")

    if len(token_ids) <= args.seq_len:
        print(f"Warning: Corpus length ({len(token_ids)}) is smaller than seq_len ({args.seq_len}).")
        print("Adjusting seq_len to fit corpus size.")
        args.seq_len = max(8, len(token_ids) // 2)

    # Check for SFT tokens
    user_id = tokenizer.user_id if args.sft or ("<|user|>" in raw_text) else None
    assistant_id = tokenizer.assistant_id if args.sft or ("<|assistant|>" in raw_text) else None
    if user_id is not None and assistant_id is not None:
        print(f"SFT Prompt Masking active: loss will only be computed on assistant responses.")

    # Dataloaders
    train_loader, val_loader = create_dataloaders(
        token_ids=token_ids,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        val_split=args.val_split,
        user_id=user_id,
        assistant_id=assistant_id,
    )

    # Model Configuration
    if model_config is None:
        model_config = ModelConfig(
            vocab_size=len(tokenizer),
            max_seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_model * 4,
        )

    train_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_epochs=args.epochs,
        grad_clip=args.grad_clip,
        val_split=args.val_split,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        early_stopping_patience=args.early_stopping,
    )

    model = CustomTransformerLM(model_config)

    # Load weights into model
    if resumed_ckpt is not None:
        if isinstance(resumed_ckpt, dict) and "model_state_dict" in resumed_ckpt:
            model.load_state_dict(resumed_ckpt["model_state_dict"])
            print(f"Successfully resumed from epoch {resumed_ckpt.get('epoch', '?')} (previous loss: {resumed_ckpt.get('loss', 0.0):.4f})")
        elif isinstance(resumed_ckpt, torch.nn.Module):
            model.load_state_dict(resumed_ckpt.state_dict())

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_config=train_config,
        model_config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    trainer.train()

if __name__ == "__main__":
    main()
