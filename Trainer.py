import os
from pathlib import Path
import pickle
import torch

from src.config import ModelConfig, TrainingConfig
from src.dataset import create_dataloaders, load_text_from_path
from src.engine import Trainer as EngineTrainer
from src.model import CustomTransformerLM, Transformer
from src.tokenizer import Tokenizer

def loadData(folder_path="data") -> str:
    """Loads and sanitizes raw text from folder_path."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    return load_text_from_path(folder_path)

def train(
    data_dir: str = "data",
    epochs: int = 50,
    batch_size: int = 8,
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 8,
    lr: float = 3e-4,
    checkpoint_dir: str = "model_weights",
):
    """
    Backwards-compatible training routine using the high-performance causal decoder engine.
    """
    data_path = Path(data_dir)
    raw_text = loadData(str(data_path))

    if not raw_text.strip():
        print(f"No training data found in '{data_dir}/'.")
        print("Creating sample corpus to demonstrate training pipeline...")
        sample_path = data_path / "sample.txt"
        sample_content = (
            "Artificial intelligence and computer science are foundational disciplines of modern engineering. "
            "A language model learns the statistical structure of language through causal next-token prediction. "
            "The attention mechanism calculates dot products between queries and keys to determine token relevance. "
            "Transformers utilize multi-head attention to attend to information from different representation subspaces. "
            "Deep neural networks require careful optimization, learning rate schedules, and normalization layers."
        )
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample_content)
        raw_text = sample_content

    print(f"Dataset loaded: {len(raw_text):,} characters")

    # Fit Tokenizer
    tokenizer = Tokenizer(vocab_size=5000)
    tokenizer.fit([raw_text])
    print(f"Vocabulary size: {len(tokenizer):,}")

    tokens = tokenizer.encode(raw_text)
    seq_len = min(128, max(8, len(tokens) // 2))

    # Create Dataloaders
    train_loader, val_loader = create_dataloaders(
        token_ids=tokens,
        seq_len=seq_len,
        batch_size=batch_size,
        val_split=0.1 if len(tokens) > 200 else 0.0,
    )

    model_config = ModelConfig(
        vocab_size=len(tokenizer),
        max_seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 4,
    )

    train_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=lr,
        max_epochs=epochs,
        checkpoint_dir=checkpoint_dir,
    )

    model = CustomTransformerLM(model_config)

    trainer = EngineTrainer(
        model=model,
        tokenizer=tokenizer,
        train_config=train_config,
        model_config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    results = trainer.train()

    # Save legacy pickle artifacts for backwards compatibility
    os.makedirs(checkpoint_dir, exist_ok=True)
    legacy_model_path = os.path.join(checkpoint_dir, "model.pt")
    legacy_tok_path = os.path.join(checkpoint_dir, "tokenizer.pkl")

    torch.save(model, legacy_model_path)
    with open(legacy_tok_path, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"Legacy model weights saved to '{legacy_model_path}' and '{legacy_tok_path}'.")
    return results

if __name__ == "__main__":
    train()