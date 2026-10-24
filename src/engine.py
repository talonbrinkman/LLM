import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.config import ModelConfig, TrainingConfig
from src.model import CustomTransformerLM
from src.tokenizer import Tokenizer

def configure_optimizers(
    model: CustomTransformerLM,
    learning_rate: float,
    weight_decay: float,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """
    Separates 2D weight matrices (which receive weight decay) from 1D biases
    and normalization parameters (which do not receive weight decay).
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(optim_groups, lr=learning_rate, betas=betas)

def get_cosine_schedule_with_warmup(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Calculates cosine learning rate decay factor with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(0.0, progress), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

class Trainer:
    """
    End-to-end training pipeline managing forward/backward passes,
    warmup-decay scheduling, perplexity tracking, and checkpointing.
    """
    def __init__(
        self,
        model: CustomTransformerLM,
        tokenizer: Tokenizer,
        train_config: TrainingConfig,
        model_config: ModelConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.model_config = model_config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Resolve device
        if train_config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(train_config.device)

        self.model.to(self.device)

        # Optimizer and Scheduler
        self.optimizer = configure_optimizers(
            self.model,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        total_steps = len(train_loader) * train_config.max_epochs
        min_lr_ratio = train_config.min_lr / train_config.learning_rate
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=train_config.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )

        self.checkpoint_dir = Path(train_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Calculates cross-entropy loss and perplexity on the validation set."""
        if self.val_loader is None or len(self.val_loader) == 0:
            return 0.0, 1.0

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, targets=y)
            if loss is not None:
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        try:
            ppl = math.exp(min(avg_loss, 20.0))  # Cap to prevent math overflow
        except OverflowError:
            ppl = float("inf")

        return avg_loss, ppl

    def save_checkpoint(self, epoch: int, loss: float, ppl: float, filename: str = "checkpoint.pt") -> None:
        """Saves model weights, tokenizer, and configurations."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config.to_dict(),
            "training_config": self.train_config.to_dict(),
            "loss": loss,
            "perplexity": ppl,
        }, checkpoint_path)

        # Save configs & tokenizer for portable deployment
        self.model_config.save_json(self.checkpoint_dir / "config.json")
        self.tokenizer.save(self.checkpoint_dir / "tokenizer.json")

    def train(self) -> Dict[str, float]:
        """Runs the main training loop with validation checks and checkpointing."""
        print(f"\nTraining on: {self.device}")
        print(f"Total Parameters: {self.model.get_num_params():,}")
        print(f"Batches per Epoch: {len(self.train_loader)}")
        print(f"Total Epochs: {self.train_config.max_epochs}")
        print("-" * 65)

        start_time = time.time()
        best_val_loss = float("inf")

        for epoch in range(1, self.train_config.max_epochs + 1):
            self.model.train()
            running_loss = 0.0

            for step, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                _, loss = self.model(x, targets=y)

                if loss is None:
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()

            epoch_train_loss = running_loss / max(1, len(self.train_loader))
            train_ppl = math.exp(min(epoch_train_loss, 20.0))
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Validation check
            log_str = f"Epoch [{epoch:03d}/{self.train_config.max_epochs:03d}] | Loss: {epoch_train_loss:.4f} | PPL: {train_ppl:.2f} | LR: {current_lr:.2e}"

            if self.val_loader is not None and len(self.val_loader) > 0 and (epoch % self.train_config.eval_interval == 0):
                val_loss, val_ppl = self.evaluate()
                log_str += f" | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, val_ppl, filename="best_model.pt")

            print(log_str)

        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        print("-" * 65)
        print(f"Training completed in {mins:02d}m {secs:02d}s.")

        # Save final checkpoint
        final_loss = epoch_train_loss
        final_ppl = train_ppl
        self.save_checkpoint(self.train_config.max_epochs, final_loss, final_ppl, filename="checkpoint.pt")
        print(f"Checkpoints successfully saved to '{self.checkpoint_dir}'.\n")

        return {"final_loss": final_loss, "final_ppl": final_ppl}
