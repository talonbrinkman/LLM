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
    parser.add_argument("--data", type=str, default="data", help="Path to text file or directory of .txt files.")
    parser.add_argument("--checkpoint-dir", type=str, default="model_weights", help="Directory to save weights & configs.")

    # Architecture Hyperparameters
    parser.add_argument("--d-model", type=int, default=256, help="Transformer embedding and hidden dimension.")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of decoder transformer layers.")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--seq-len", type=int, default=256, help="Maximum sequence context window.")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Maximum tokenizer vocabulary size.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of full training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per training step.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Decoupled weight decay coefficient.")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Linear learning rate warmup steps.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient norm threshold.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation dataset fraction.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device.")
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing checkpoint if available.")

    return parser.parse_args()

def main():
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        print(f"Error: Specified data path '{data_path}' does not exist.")
        print("Please place text files in 'data/' or point to a file via --data <path>.")
        sys.exit(1)

    print(f"Loading corpus from: {data_path.resolve()}")
    raw_text = load_text_from_path(data_path)

    if not raw_text.strip():
        print("Error: No text data found in the specified path.")
        sys.exit(1)

    print(f"Corpus size: {len(raw_text):,} characters")

    # Fit or Resume Tokenizer
    existing_tok_path = Path(args.checkpoint_dir) / "tokenizer.json"
    if args.resume and existing_tok_path.exists():
        print(f"Resuming with existing tokenizer from: {existing_tok_path}")
        tokenizer = Tokenizer.load(existing_tok_path)
    else:
        print("Fitting tokenizer on corpus...")
        tokenizer = Tokenizer(vocab_size=args.vocab_size)
        tokenizer.fit([raw_text])
        print(f"Vocabulary built: {len(tokenizer):,} unique tokens (pad=0, unk=1, bos=2, eos=3)")

    token_ids = tokenizer.encode(raw_text)
    print(f"Tokenized corpus length: {len(token_ids):,} tokens")

    # Check for existing checkpoint when resuming
    resumed_ckpt = None
    model_config = None

    if args.resume:
        ckpt_candidates = [
            Path(args.checkpoint_dir) / "checkpoint.pt",
            Path(args.checkpoint_dir) / "best_model.pt",
        ]
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
            print(f"Warning: --resume passed but no checkpoint found in '{args.checkpoint_dir}'. Training from scratch.")

    if len(token_ids) <= args.seq_len:
        print(f"Warning: Corpus length ({len(token_ids)}) is smaller than seq_len ({args.seq_len}).")
        print("Adjusting seq_len to fit corpus size.")
        args.seq_len = max(8, len(token_ids) // 2)

    # Dataloaders
    train_loader, val_loader = create_dataloaders(
        token_ids=token_ids,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        val_split=args.val_split,
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
