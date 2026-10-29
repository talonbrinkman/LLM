import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with PyTorch 2.0 SDPA optimization
    and optional Key-Value caching for autoregressive decoding.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout

        # Fused Q, K, V projection for maximum throughput
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precompute causal mask for fallback or non-SDPA execution
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()

        # Project and separate into Q, K, V: shape (B, T, 3 * C)
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache during generation
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_cache = (k, v) if use_cache else None
        total_seq_len = k.size(2)

        # Scaled dot-product attention
        if hasattr(F, "scaled_dot_product_attention") and not use_cache:
            # Leverage PyTorch 2.0 FlashAttention / memory-efficient kernel when possible
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Explicit causal scaled dot-product attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            # Causal mask slice
            mask = self.causal_mask[:, :, total_seq_len - T:total_seq_len, :total_seq_len]
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            if self.training and self.dropout > 0:
                att = F.dropout(att, p=self.dropout)
            y = att @ v

        # Recombine heads to (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(y))
        return out, new_cache

class MLP(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))

class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.ln_1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache

class CustomTransformerLM(nn.Module):
    """
    Autoregressive Decoder-Only Causal Language Model.
    Engineered for rigorous end-to-end training and inference.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share parameters between input embeddings and output projection
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-style weight initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Returns parameter count, optionally excluding position embeddings."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_embedding.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for next-token prediction.
        idx: (B, T) LongTensor of token IDs.
        targets: (B, T) LongTensor of target token IDs.
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        positions = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        # Combine token embeddings and learned positional embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(positions)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        eos_id: Optional[int] = None,
        stop_sequences: Optional[List[List[int]]] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature, top-k, top-p,
        repetition penalty, and stop sequence controls.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)
            # Focus on the final token's logits
            next_token_logits = logits[:, -1, :].clone()

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

            # Apply temperature scaling
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
            else:
                # Greedy argmax
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                idx = torch.cat([idx, next_token], dim=1)
                if eos_id is not None and next_token.item() == eos_id:
                    break
                if stop_sequences:
                    stopped = False
                    for seq in stop_sequences:
                        seq_len = len(seq)
                        if seq_len > 0 and idx.size(1) >= seq_len and idx[0, -seq_len:].tolist() == seq:
                            idx = idx[:, :-seq_len]
                            stopped = True
                            break
                    if stopped:
                        break
                continue

            # Top-K filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float("-inf")

            # Top-P (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Mask tokens past the top_p threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift mask right to ensure at least one token is kept
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)
            if eos_id is not None and next_token.item() == eos_id:
                break
            if stop_sequences:
                stopped = False
                for seq in stop_sequences:
                    seq_len = len(seq)
                    if seq_len > 0 and idx.size(1) >= seq_len and idx[0, -seq_len:].tolist() == seq:
                        idx = idx[:, :-seq_len]
                        stopped = True
                        break
                if stopped:
                    break

        return idx

# Backward-compatible alias for existing scripts
Transformer = CustomTransformerLM
