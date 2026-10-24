import argparse
import os
from pathlib import Path
import pickle
import sys
from typing import Optional
import torch

from src.config import ModelConfig
from src.model import CustomTransformerLM
from src.tokenizer import Tokenizer

def load_inference_artifacts(
    checkpoint_path: Path,
    tokenizer_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = checkpoint_path.parent

    # 1. Load Tokenizer
    tokenizer = None
    possible_tok_paths = [
        tokenizer_path,
        checkpoint_dir / "tokenizer.json",
        checkpoint_dir / "tokenizer.pkl",
    ]
    for p in possible_tok_paths:
        if p and Path(p).exists():
            if str(p).endswith(".json"):
                tokenizer = Tokenizer.load(p)
                break
            elif str(p).endswith(".pkl"):
                with open(p, "rb") as f:
                    tokenizer = pickle.load(f)
                break

    if tokenizer is None:
        raise FileNotFoundError(f"Could not find tokenizer in {checkpoint_dir}. Train a model first or specify --tokenizer.")

    # 2. Load Model & Config
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Check if checkpoint is full state dict dict or raw model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_config_dict = ckpt.get("model_config")
        if model_config_dict:
            model_config = ModelConfig.from_dict(model_config_dict)
        elif (checkpoint_dir / "config.json").exists():
            model_config = ModelConfig.load_json(checkpoint_dir / "config.json")
        else:
            model_config = ModelConfig(vocab_size=len(tokenizer))

        model = CustomTransformerLM(model_config)
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, torch.nn.Module):
        model = ckpt
    else:
        # Bare state dict
        if (checkpoint_dir / "config.json").exists():
            model_config = ModelConfig.load_json(checkpoint_dir / "config.json")
        else:
            model_config = ModelConfig(vocab_size=len(tokenizer))
        model = CustomTransformerLM(model_config)
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model, tokenizer, device

def generate_text(
    model: CustomTransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_tokens: int = 150,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [tokenizer.bos_id if hasattr(tokenizer, "bos_id") else 0]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos_id = getattr(tokenizer, "eos_id", None)

    out_tensor = model.generate(
        input_tensor,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_id=eos_id,
    )

    generated_ids = out_tensor[0].tolist()
    return tokenizer.decode(generated_ids)

def run_repl(model, tokenizer, device, args):
    print("\n" + "=" * 60)
    print(" Custom LLM Interactive Generation Shell")
    print(" Type your prompt and press Enter. Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print("Exiting.")
                break

            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            print(f"\n{response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

def main():
    parser = argparse.ArgumentParser(description="Generate text using trained Custom LLM.")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for one-shot text generation.")
    parser.add_argument("--checkpoint", type=str, default="model_weights/checkpoint.pt", help="Path to checkpoint.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional custom path to tokenizer.")
    parser.add_argument("--max-tokens", type=int, default=150, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (lower = more deterministic).")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K token candidates.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus probability threshold.")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Penalty for repeating tokens.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive shell.")

    args = parser.parse_args()
    ckpt_path = Path(args.checkpoint)

    # Fallback to best_model.pt or model.pt if checkpoint.pt doesn't exist
    if not ckpt_path.exists():
        fallback_best = ckpt_path.parent / "best_model.pt"
        fallback_legacy = ckpt_path.parent / "model.pt"
        if fallback_best.exists():
            ckpt_path = fallback_best
        elif fallback_legacy.exists():
            ckpt_path = fallback_legacy
        else:
            print(f"Error: No trained model checkpoint found at '{ckpt_path}'.")
            print("Please train a model first using: python train.py")
            sys.exit(1)

    tok_path = Path(args.tokenizer) if args.tokenizer else None

    try:
        model, tokenizer, device = load_inference_artifacts(ckpt_path, tok_path)
    except Exception as e:
        print(f"Error initializing inference engine: {e}")
        sys.exit(1)

    if args.interactive or args.prompt is None:
        run_repl(model, tokenizer, device, args)
    else:
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"\n{output}")

if __name__ == "__main__":
    main()
