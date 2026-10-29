import os
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset

def clean_text(text: str) -> str:
    """
    Sanitizes text by stripping carriage returns and normalizing spaces while preserving newlines.
    Prevents dialogue and paragraph structure from collapsing.
    """
    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Normalize tabs, form feeds, and vertical tabs to single spaces
    text = re.sub(r"[\t\f\v]", " ", text)
    # Collapse multiple horizontal spaces on the same line into a single space (preserves \n)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip leading and trailing whitespace on each line
    text = re.sub(r"[^\S\n]+\n", "\n", text)
    text = re.sub(r"\n[^\S\n]+", "\n", text)
    # Collapse 3+ newlines into 2 (clean paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _has_dialogue_markers(text: str) -> bool:
    """Checks whether text contains structured conversational turns."""
    if "<|user|>" in text and "<|assistant|>" in text:
        return True
    if re.search(r"(?:^|\n)User:\s", text) and re.search(r"(?:^|\n)(?:Jarvis|Assistant):\s", text):
        return True
    return False

def load_text_from_path(
    path: Union[str, Path],
    sft_only: bool = False,
    shuffle_docs: bool = True,
    seed: int = 42,
) -> str:
    """
    Loads text from a single file, comma-separated file paths, or recursively scans a directory for .txt files.
    When sft_only=True on a directory, skips non-conversational raw story/narrative files.
    When shuffle_docs=True, deterministically shuffles documents (<|endoftext|> blocks) so train/val splits
    sample evenly across all datasets instead of taking only the tail end of the last alphabetical file.
    """
    raw_path_str = str(path)
    if "," in raw_path_str:
        sub_paths = [Path(s.strip()) for s in raw_path_str.split(",") if s.strip()]
    else:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Data path does not exist: {path}")
        sub_paths = [p]

    file_list: List[Path] = []
    for sp in sub_paths:
        if not sp.exists():
            raise FileNotFoundError(f"Data path does not exist: {sp}")
        if sp.is_file():
            file_list.append(sp)
        elif sp.is_dir():
            file_list.extend(sorted(sp.rglob("*.txt")))

    loaded_files: List[Tuple[Path, str]] = []
    for file_path in file_list:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = clean_text(f.read())
                if content:
                    loaded_files.append((file_path, content))
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

    if sft_only and len(loaded_files) > 1:
        dialogue_files = [(fp, c) for fp, c in loaded_files if _has_dialogue_markers(c)]
        if dialogue_files:
            skipped = [fp.name for fp, c in loaded_files if not _has_dialogue_markers(c)]
            for name in skipped:
                print(f"[SFT Mode] Skipping non-dialogue file: {name}")
            loaded_files = dialogue_files

    collected_chunks = [c for _, c in loaded_files]
    combined = "\n\n".join(collected_chunks)

    if shuffle_docs and ("<|endoftext|>" in combined or len(loaded_files) > 1):
        if "<|endoftext|>" in combined:
            parts = [doc.strip() for doc in combined.split("<|endoftext|>") if doc.strip()]
            docs = [f"{doc}<|endoftext|>" for doc in parts]
        else:
            docs = [doc.strip() for doc in combined.split("\n\n") if doc.strip()]

        if len(docs) > 1:
            rng = random.Random(seed)
            rng.shuffle(docs)
            combined = "\n\n".join(docs)

    return combined

class CausalLMDataset(Dataset):
    """
    Constructs sliding window autoregressive training pairs (x, y)
    where y is shifted by exactly 1 token: x_t -> x_{t+1}.
    Optionally masks user prompt tokens in y with -100 (Supervised Fine-Tuning).
    """
    def __init__(
        self,
        token_ids: List[int],
        seq_len: int,
        stride: Optional[int] = None,
        user_id: Optional[int] = None,
        assistant_id: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.user_id = user_id
        self.assistant_id = assistant_id

        # Precompute SFT prompt mask across the full token stream so windows
        # starting mid-prompt are accurately masked without per-epoch Python loops.
        self.prompt_mask: Optional[torch.Tensor] = None
        if self.user_id is not None and self.assistant_id is not None and len(token_ids) > 0:
            mask_list = [False] * len(token_ids)
            is_user_prompt = False
            u_id = self.user_id
            a_id = self.assistant_id
            for i, tok in enumerate(token_ids):
                if tok == u_id:
                    is_user_prompt = True
                    mask_list[i] = True
                elif tok == a_id:
                    mask_list[i] = True
                    is_user_prompt = False
                elif is_user_prompt:
                    mask_list[i] = True
            self.prompt_mask = torch.tensor(mask_list, dtype=torch.bool)

        # Total number of full windows
        total_tokens = len(self.tokens)
        if total_tokens <= seq_len:
            self.n_samples = 1 if total_tokens > 1 else 0
        else:
            self.n_samples = max(0, (total_tokens - 1 - seq_len) // self.stride + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len

        x = self.tokens[start_idx:end_idx]
        y = self.tokens[start_idx + 1:end_idx + 1].clone()
        slice_len = len(x)

        # In case the slice is shorter than seq_len
        if slice_len < self.seq_len:
            pad_len = self.seq_len - slice_len
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])

        # Supervised Fine-Tuning (SFT) Prompt Masking:
        # Mask prompt tokens in targets with -100 so loss is computed solely on assistant replies
        if self.prompt_mask is not None:
            mask_slice = self.prompt_mask[start_idx:start_idx + slice_len]
            y[:slice_len][mask_slice] = -100

        return x, y

def create_dataloaders(
    token_ids: List[int],
    seq_len: int,
    batch_size: int,
    val_split: float = 0.1,
    stride: Optional[int] = None,
    num_workers: int = 0,
    user_id: Optional[int] = None,
    assistant_id: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Splits token stream into train and validation DataLoaders.
    """
    split_idx = int(len(token_ids) * (1.0 - val_split))
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    train_ds = CausalLMDataset(
        train_tokens,
        seq_len=seq_len,
        stride=stride,
        user_id=user_id,
        assistant_id=assistant_id,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if len(val_tokens) > seq_len:
        val_ds = CausalLMDataset(
            val_tokens,
            seq_len=seq_len,
            stride=seq_len,
            user_id=user_id,
            assistant_id=assistant_id,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader
