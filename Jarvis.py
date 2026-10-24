import os
from pathlib import Path
import pickle
import sys
import torch

from src.config import ModelConfig
from src.model import CustomTransformerLM, Transformer
from src.tokenizer import Tokenizer

class Jarvis:
    """
    Interactive Assistant interface powered by the Custom LLM decoder architecture.
    Maintains backwards compatibility with existing scripts.
    """
    def __init__(self, checkpoint_dir: str = "model_weights"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model = None
        self.tokenizer = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if not self.checkpoint_dir.exists():
            print(f"[Jarvis] Checkpoint directory '{self.checkpoint_dir}' does not exist.")
            print("[Jarvis] Train a model first via: python train.py")
            return

        # 1. Load Tokenizer (JSON or legacy PKL)
        tok_json = self.checkpoint_dir / "tokenizer.json"
        tok_pkl = self.checkpoint_dir / "tokenizer.pkl"

        if tok_json.exists():
            self.tokenizer = Tokenizer.load(tok_json)
        elif tok_pkl.exists():
            with open(tok_pkl, "rb") as f:
                self.tokenizer = pickle.load(f)
        else:
            print(f"[Jarvis] No tokenizer found in {self.checkpoint_dir}.")
            return

        # 2. Load Model Checkpoint
        ckpt_pt = self.checkpoint_dir / "checkpoint.pt"
        best_pt = self.checkpoint_dir / "best_model.pt"
        model_pt = self.checkpoint_dir / "model.pt"

        target_file = None
        for cand in [ckpt_pt, best_pt, model_pt]:
            if cand.exists():
                target_file = cand
                break

        if target_file is None:
            print(f"[Jarvis] No model weights found in {self.checkpoint_dir}.")
            return

        try:
            ckpt = torch.load(target_file, map_location=self.device, weights_only=False)

            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                config_dict = ckpt.get("model_config")
                if config_dict:
                    config = ModelConfig.from_dict(config_dict)
                else:
                    config = ModelConfig(vocab_size=len(self.tokenizer))
                self.model = CustomTransformerLM(config)
                self.model.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, torch.nn.Module):
                self.model = ckpt
            else:
                config_path = self.checkpoint_dir / "config.json"
                if config_path.exists():
                    config = ModelConfig.load_json(config_path)
                else:
                    config = ModelConfig(vocab_size=len(self.tokenizer))
                self.model = CustomTransformerLM(config)
                self.model.load_state_dict(ckpt)

            self.model.to(self.device)
            self.model.eval()
            print(f"[Jarvis] Model loaded successfully on {self.device} from '{target_file.name}'!")
        except Exception as e:
            print(f"[Jarvis] Error loading model: {e}")
            self.model = None

    def generate(
        self,
        prompt: str,
        max_length: int = 150,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            return "Jarvis is offline: please train and save model weights first."

        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            input_ids = [getattr(self.tokenizer, "bos_id", 0)]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        eos_id = getattr(self.tokenizer, "eos_id", None)

        out_tokens = self.model.generate(
            input_tensor,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.1,
            eos_id=eos_id,
        )

        return self.tokenizer.decode(out_tokens[0])

if __name__ == "__main__":
    jarvis = Jarvis()
    print("\n[Jarvis Active - Type 'quit' to exit]")
    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                break
            response = jarvis.generate(user_input)
            print(f"\n     {response}\n")
        except (KeyboardInterrupt, EOFError):
            break