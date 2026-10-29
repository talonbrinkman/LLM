from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Optional, Union

@dataclass
class ModelConfig:
    vocab_size: int = 5000
    max_seq_len: int = 512
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    bias: bool = False
    tie_weights: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_epochs: int = 20
    grad_clip: float = 1.0
    val_split: float = 0.1
    eval_interval: int = 1
    checkpoint_dir: str = "model_weights"
    device: str = "auto"
    seed: int = 42
    early_stopping_patience: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
