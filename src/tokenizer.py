import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
import torch

class Tokenizer:
    """
    Subword-aware frequency-based tokenizer with reserved special tokens.
    Designed for transparent, zero-dependency causal language model training.
    """
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        self.word_to_idx.clear()
        self.idx_to_word.clear()
        for idx, token in enumerate(self.SPECIAL_TOKENS):
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token

    @property
    def pad_id(self) -> int:
        return self.word_to_idx[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.word_to_idx[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.word_to_idx[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.word_to_idx[self.EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.word_to_idx)

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        """Splits text into words, numbers, and individual punctuation marks."""
        return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

    def fit(self, texts: Iterable[str]) -> "Tokenizer":
        """Builds vocabulary from an iterable of text strings."""
        self._init_special_tokens()
        freqs = Counter()

        for text in texts:
            tokens = self._tokenize_text(text)
            freqs.update(tokens)

        # Allocate slots for non-special tokens up to vocab_size
        available_slots = max(0, self.vocab_size - len(self.SPECIAL_TOKENS))
        most_common = freqs.most_common(available_slots)

        for token, _ in most_common:
            if token not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[token] = idx
                self.idx_to_word[idx] = token

        return self

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> List[int]:
        """Encodes text into a list of token IDs."""
        tokens = self._tokenize_text(text)
        unk_idx = self.unk_id
        ids = [self.word_to_idx.get(t, unk_idx) for t in tokens]

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]

        return ids

    def decode(
        self,
        indices: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decodes token IDs back into human-readable text."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        words = []
        for idx in indices:
            token = self.idx_to_word.get(idx, self.UNK_TOKEN)
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            words.append(token)

        # Format punctuation cleanly
        raw = " ".join(words)
        cleaned = re.sub(r"\s+([.,!?:;'\"])", r"\1", raw)
        return cleaned.strip()

    def save(self, path: Union[str, Path]) -> None:
        """Saves tokenizer state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "word_to_idx": self.word_to_idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Tokenizer":
        """Loads tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        tok = cls(vocab_size=payload.get("vocab_size", len(payload["word_to_idx"])))
        tok.word_to_idx = {k: int(v) for k, v in payload["word_to_idx"].items()}
        tok.idx_to_word = {int(v): k for k, v in tok.word_to_idx.items()}
        return tok
