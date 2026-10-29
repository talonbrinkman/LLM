import torch
import pytest

from src.config import ModelConfig
from src.model import CustomTransformerLM

@pytest.fixture
def small_model():
    config = ModelConfig(
        vocab_size=100,
        max_seq_len=32,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
    )
    return CustomTransformerLM(config)

def test_forward_pass_shapes(small_model):
    B, T = 2, 8
    x = torch.randint(0, 100, (B, T))
    logits, loss = small_model(x)

    assert logits.shape == (B, T, 100)
    assert loss is None

def test_loss_computation(small_model):
    B, T = 2, 8
    x = torch.randint(0, 100, (B, T))
    y = torch.randint(0, 100, (B, T))
    logits, loss = small_model(x, targets=y)

    assert loss is not None
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0

def test_causal_attention_masking(small_model):
    """
    Mathematical verification that the model is strictly causal:
    Modifying token at position t > i must NOT change the logits at position i.
    """
    small_model.eval()
    T = 6
    seq1 = torch.tensor([[10, 20, 30, 40, 50, 60]])
    seq2 = torch.tensor([[10, 20, 30, 99, 88, 77]])  # Altered starting at index 3

    with torch.no_grad():
        logits1, _ = small_model(seq1)
        logits2, _ = small_model(seq2)

    # For positions 0, 1, 2, logits must be identical
    for i in range(3):
        assert torch.allclose(logits1[0, i], logits2[0, i], atol=1e-5), (
            f"Causality leak detected at position {i}! Future tokens influenced past output."
        )

def test_autoregressive_generation(small_model):
    small_model.eval()
    prompt = torch.tensor([[5, 12]])
    max_new = 10
    out = small_model.generate(prompt, max_new_tokens=max_new, temperature=0.7)

    assert out.shape == (1, 2 + max_new)
    assert out[0, 0].item() == 5
    assert out[0, 1].item() == 12

def test_generation_stop_sequences(small_model):
    small_model.eval()
    prompt = torch.tensor([[5, 12]])
    # Force greedy generation to be deterministic
    # Check that generation stops when stop_sequences matches
    out = small_model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.0,
    )
    # Pick a sequence of tokens from the output
    tokens = out[0].tolist()
    if len(tokens) >= 4:
        stop_seq = tokens[2:4]
        out_stopped = small_model.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.0,
            stop_sequences=[stop_seq],
        )
        # Should stop before or cut off stop_seq
        assert out_stopped.size(1) == 2 or out_stopped.size(1) < out.size(1)
