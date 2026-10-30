import argparse
import os
from pathlib import Path
import pickle
import re
import sys
from typing import List, Optional, Tuple
import torch

from src.config import ModelConfig
from src.model import CustomTransformerLM
from src.tokenizer import Tokenizer

def clean_response(text: str) -> str:
    """Post-processes raw decoded output to clean contractions, quotes, and punctuation."""
    text = re.sub(r"<\|(?:endoftext|user|assistant|system)\|>", "", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"'\s+([a-zA-Z])", r"'\1", text)
    text = re.sub(r"([a-zA-Z])\s+'([a-zA-Z])", r"\1'\2", text)
    text = re.sub(r"(\d+)\s*:\s*(\d+)", r"\1:\2", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r"\.\s*,", ".", text)
    text = re.sub(r"^(jarvis|assistant)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*(user|human)\s*:\s*$", "", text, flags=re.IGNORECASE)
    text = text.strip()
    # If no alphanumeric characters present, return empty
    if not re.search(r"[a-zA-Z0-9]", text):
        return ""
    text = re.sub(r"^[\s.,!?:;]+", "", text)
    if text:
        sentences = re.split(r"([.!?]\s*)", text)
        cleaned_sentences = []
        for s in sentences:
            if s and not re.match(r"^[.!?]\s*$", s):
                s = s[0].upper() + s[1:]
            cleaned_sentences.append(s)
        text = "".join(cleaned_sentences)
    return text.strip()

def load_inference_artifacts(
    checkpoint_path: Path,
    tokenizer_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    fp16: bool = True,
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

    if fp16 and device.type == "cuda":
        model.half()

    model.to(device)
    model.eval()
    return model, tokenizer, device

def generate_text(
    model: CustomTransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    chat_mode: bool = True,
    history: Optional[List[Tuple[str, str]]] = None,
) -> str:
    prompt_clean = prompt.strip()

    uses_special_tokens = (
        (
            hasattr(tokenizer, "_hf_tokenizer")
            and tokenizer._hf_tokenizer is not None
            and tokenizer._hf_tokenizer.token_to_id("<|user|>") is not None
        )
        or (
            hasattr(tokenizer, "word_to_idx")
            and "<|user|>" in tokenizer.word_to_idx
            and "<|assistant|>" in tokenizer.word_to_idx
            and "|" not in tokenizer.word_to_idx
        )
    )

    if chat_mode:
        if uses_special_tokens:
            turns = []
            if history:
                for u, j in history[-2:]:
                    turns.append(f"<|user|>\n{u}\n<|assistant|>\n{j}<|endoftext|>")
            turns.append(f"<|user|>\n{prompt_clean}\n<|assistant|>\n")
            formatted_prompt = "".join(turns)
            stop_seqs = [
                tokenizer.encode("<|endoftext|>"),
                tokenizer.encode("<|user|>"),
            ]
        else:
            turns = []
            if history:
                for u, j in history[-2:]:
                    turns.append(f"user: {u} jarvis: {j}")
            turns.append(f"user: {prompt_clean} jarvis:")
            formatted_prompt = " ".join(turns)
            stop_seqs = [tokenizer.encode("user :"), tokenizer.encode("user:")]
        input_ids = tokenizer.encode(formatted_prompt)
    else:
        input_ids = tokenizer.encode(prompt_clean)
        stop_seqs = None

    if not input_ids:
        input_ids = [tokenizer.bos_id if hasattr(tokenizer, "bos_id") else 0]

    max_seq_len = getattr(model.config, "max_seq_len", 256)
    max_ctx = max(32, max_seq_len - max_tokens)
    if len(input_ids) > max_ctx:
        input_ids = input_ids[-max_ctx:]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    if chat_mode and uses_special_tokens and hasattr(tokenizer, "eot_id"):
        eos_id = tokenizer.eot_id
    else:
        eos_id = getattr(tokenizer, "eos_id", None)

    out_tensor = model.generate(
        input_tensor,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_id=eos_id,
        stop_sequences=stop_seqs,
    )

    if chat_mode:
        new_ids = out_tensor[0][len(input_ids):].tolist()
        raw_output = tokenizer.decode(new_ids)
        response = clean_response(raw_output)
        if not response:
            response = "Hello! How can I assist you today?"
        return response
    else:
        generated_ids = out_tensor[0].tolist()
        return tokenizer.decode(generated_ids)

def run_repl(model, tokenizer, device, args):
    print("\n" + "=" * 60)
    print(" Custom LLM Interactive Generation Shell")
    print(" Type your prompt and press Enter.")
    print(" Type 'reset' to clear history, 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    history: List[Tuple[str, str]] = []

    while True:
        try:
            prompt = input(">>> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print("Exiting.")
                break
            if prompt.lower() == "reset":
                history.clear()
                print("[Session] Conversation history reset.\n")
                continue

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
                chat_mode=not args.raw,
                history=history,
            )
            if not args.raw:
                history.append((prompt, response))
                print(f"\nJarvis: {response}\n")
            else:
                print(f"\n{response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

def main():
    parser = argparse.ArgumentParser(description="Generate text using trained Custom LLM.")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for one-shot text generation.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_weights/best_model.pt",
        help="Path to a checkpoint (.pt) file or weights directory (default: model_weights/best_model.pt).",
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional custom path to tokenizer.")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (lower = more deterministic).")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K token candidates.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus probability threshold.")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Penalty for repeating tokens.")
    parser.add_argument("--raw", action="store_true", help="Generate raw completion without dialogue formatting.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive shell.")
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run inference in FP16 on CUDA (default: True).",
    )

    args = parser.parse_args()
    ckpt_path = Path(args.checkpoint)

    # Support passing a directory (e.g. --checkpoint model_weights_v2) or falling back across standard filenames
    if ckpt_path.is_dir():
        candidates = [
            ckpt_path / "best_model.pt",
            ckpt_path / "checkpoint.pt",
            ckpt_path / "model.pt",
        ]
        resolved = next((c for c in candidates if c.exists()), None)
        if resolved is None:
            print(f"Error: No trained model checkpoint found in directory '{ckpt_path}'.")
            sys.exit(1)
        ckpt_path = resolved
    elif not ckpt_path.exists():
        candidates = [
            ckpt_path.parent / "best_model.pt",
            ckpt_path.parent / "checkpoint.pt",
            ckpt_path.parent / "model.pt",
        ]
        resolved = next((c for c in candidates if c.exists()), None)
        if resolved is not None:
            ckpt_path = resolved
        else:
            print(f"Error: No trained model checkpoint found at '{ckpt_path}'.")
            print("Please train a model first using: python train.py")
            sys.exit(1)

    tok_path = Path(args.tokenizer) if args.tokenizer else None

    try:
        model, tokenizer, device = load_inference_artifacts(ckpt_path, tok_path, fp16=args.fp16)
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
            chat_mode=not args.raw,
        )
        if not args.raw:
            print(f"\nJarvis: {output}")
        else:
            print(f"\n{output}")

if __name__ == "__main__":
    main()
