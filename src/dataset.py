import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset

def clean_text(text: str) -> str:
    """Sanitizes text by stripping control characters and collapsing whitespace."""
    text = re.sub(r"[\r\t\f\v]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_text_from_path(path: Union[str, Path]) -> str:
    """
    Loads text from a single file or recursively scans a directory for all .txt files.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data path does not exist: {path}")

    collected_chunks: List[str] = []

    if p.is_file():
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            content = clean_text(f.read())
            if content:
                collected_chunks.append(content)
    elif p.is_dir():
        for file_path in sorted(p.rglob("*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = clean_text(f.read())
                    if content:
                        collected_chunks.append(content)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")

    return " \n ".join(collected_chunks)

class CausalLMDataset(Dataset):
    """
    Constructs sliding window autoregressive training pairs (x, y)
    where y is shifted by exactly 1 token: x_t -> x_{t+1}.
    """
    def __init__(self, token_ids: List[int], seq_len: int, stride: Optional[int] = None):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.tokens = torch.tensor(token_ids, dtype=torch.long)

        # Total number of full windows
        total_tokens = len(self.tokens)
        if total_tokens <= seq_len:
            # Handle minimal data by padding if necessary
            self.n_samples = 1 if total_tokens > 1 else 0
        else:
            self.n_samples = max(0, (total_tokens - 1 - seq_len) // self.stride + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len

        x = self.tokens[start_idx:end_idx]
        y = self.tokens[start_idx + 1:end_idx + 1]

        # In case the slice is shorter than seq_len
        if len(x) < self.seq_len:
            pad_len = self.seq_len - len(x)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])

        return x, y

def create_dataloaders(
    token_ids: List[int],
    seq_len: int,
    batch_size: int,
    val_split: float = 0.1,
    stride: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Splits token stream into train and validation DataLoaders.
    """
    split_idx = int(len(token_ids) * (1.0 - val_split))
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    train_ds = CausalLMDataset(train_tokens, seq_len=seq_len, stride=stride)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if len(val_tokens) > seq_len:
        val_ds = CausalLMDataset(val_tokens, seq_len=seq_len, stride=seq_len)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader
