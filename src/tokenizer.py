import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
import torch

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    HAS_HF_TOKENIZERS = True
except ImportError:
    HAS_HF_TOKENIZERS = False

class Tokenizer:
    """
    Unified Tokenizer supporting both modern Byte-Level BPE (GPT-2/Llama style)
    and legacy frequency-based tokenization with backwards-compatible loading.
    """
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    SYSTEM_TOKEN = "<|system|>"
    ENDOFTEXT_TOKEN = "<|endoftext|>"

    SPECIAL_TOKENS = [
        PAD_TOKEN,
        UNK_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        USER_TOKEN,
        ASSISTANT_TOKEN,
        SYSTEM_TOKEN,
        ENDOFTEXT_TOKEN,
    ]

    def __init__(self, vocab_size: int = 32768, use_bpe: bool = True):
        self.vocab_size = vocab_size
        if use_bpe and vocab_size >= 256 and not HAS_HF_TOKENIZERS:
            print(
                "Warning: 'tokenizers' library not found; falling back to word-level tokenizer. "
                "Install via 'pip install tokenizers' for Byte-Level BPE."
            )
        self.use_bpe = use_bpe and HAS_HF_TOKENIZERS and vocab_size >= 256
        self._hf_tokenizer: Optional[HFTokenizer] = None

        # Fallback / Legacy structures
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
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id(self.PAD_TOKEN) or 0
        return self.word_to_idx.get(self.PAD_TOKEN, 0)

    @property
    def unk_id(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id(self.UNK_TOKEN) or 1
        return self.word_to_idx.get(self.UNK_TOKEN, 1)

    @property
    def bos_id(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id(self.BOS_TOKEN) or 2
        return self.word_to_idx.get(self.BOS_TOKEN, 2)

    @property
    def eos_id(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id(self.EOS_TOKEN) or 3
        return self.word_to_idx.get(self.EOS_TOKEN, 3)

    @property
    def user_id(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id(self.USER_TOKEN) or 4
        return self.word_to_idx.get(self.USER_TOKEN, 4)

    @property
    def assistant_id(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id(self.ASSISTANT_TOKEN) or 5
        return self.word_to_idx.get(self.ASSISTANT_TOKEN, 5)

    @property
    def system_id(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id(self.SYSTEM_TOKEN) or 6
        return self.word_to_idx.get(self.SYSTEM_TOKEN, 6)

    @property
    def eot_id(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id(self.ENDOFTEXT_TOKEN) or 7
        return self.word_to_idx.get(self.ENDOFTEXT_TOKEN, 7)

    def __len__(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.get_vocab_size()
        return len(self.word_to_idx)

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        """Splits text into special tokens, words, numbers, and individual punctuation marks."""
        return re.findall(r"<\|[a-zA-Z0-9_]+\|>|<[a-zA-Z0-9_]+>|\w+|[^\w\s]", text.lower(), re.UNICODE)

    def fit(self, texts: Iterable[str]) -> "Tokenizer":
        """
        Builds vocabulary from an iterable of text strings.
        Uses Byte-Level BPE when vocab_size >= 256, or frequency dictionary when smaller.
        """
        if self.use_bpe:
            hf_tok = HFTokenizer(BPE(unk_token=self.UNK_TOKEN))
            hf_tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
            hf_tok.decoder = ByteLevelDecoder()

            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=self.SPECIAL_TOKENS,
                initial_alphabet=ByteLevel.alphabet(),
            )
            hf_tok.train_from_iterator(texts, trainer)
            self._hf_tokenizer = hf_tok
            return self

        # Legacy / Small vocab fit
        self._init_special_tokens()
        freqs = Counter()
        for text in texts:
            tokens = self._tokenize_text(text)
            freqs.update(tokens)

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
        if self._hf_tokenizer is not None:
            ids = self._hf_tokenizer.encode(text).ids
            if add_special_tokens:
                ids = [self.bos_id] + ids + [self.eos_id]
            return ids

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

        if self._hf_tokenizer is not None:
            # Let HF BPE decode byte levels cleanly
            return self._hf_tokenizer.decode(indices, skip_special_tokens=skip_special_tokens)

        words = []
        for idx in indices:
            token = self.idx_to_word.get(idx, self.UNK_TOKEN)
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            words.append(token)

        raw = " ".join(words)
        cleaned = re.sub(r"\s+([.,!?:;'\"])", r"\1", raw)
        cleaned = re.sub(r"'\s+([a-zA-Z])", r"'\1", cleaned)
        cleaned = re.sub(r"([a-zA-Z])\s+'([a-zA-Z])", r"\1'\2", cleaned)
        cleaned = re.sub(r"(\d+)\s*:\s*(\d+)", r"\1:\2", cleaned)
        return cleaned.strip()

    def save(self, path: Union[str, Path]) -> None:
        """Saves tokenizer state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._hf_tokenizer is not None:
            self._hf_tokenizer.save(str(path))
            return

        payload = {
            "vocab_size": self.vocab_size,
            "word_to_idx": self.word_to_idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Tokenizer":
        """Loads tokenizer from JSON with legacy detection."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Legacy frequency dictionary format
        if isinstance(data, dict) and "word_to_idx" in data:
            tok = cls(vocab_size=data.get("vocab_size", len(data["word_to_idx"])), use_bpe=False)
            tok.word_to_idx = {k: int(v) for k, v in data["word_to_idx"].items()}
            tok.idx_to_word = {int(v): k for k, v in tok.word_to_idx.items()}
            return tok

        # Modern HuggingFace BPE format
        if HAS_HF_TOKENIZERS:
            hf_tok = HFTokenizer.from_file(str(path))
            tok = cls(vocab_size=hf_tok.get_vocab_size(), use_bpe=True)
            tok._hf_tokenizer = hf_tok
            return tok

        raise RuntimeError(f"Could not load BPE tokenizer from '{path}': 'tokenizers' package is required.")
