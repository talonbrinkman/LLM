import torch
from src.dataset import CausalLMDataset, create_dataloaders

def test_causal_dataset_next_token_shift():
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    seq_len = 4
    ds = CausalLMDataset(tokens, seq_len=seq_len, stride=1)

    assert len(ds) == (len(tokens) - 1 - seq_len) + 1  # 8 - 1 - 4 + 1 = 4 samples

    x0, y0 = ds[0]
    assert x0.tolist() == [10, 20, 30, 40]
    assert y0.tolist() == [20, 30, 40, 50]  # exact 1-token shift

    # Check alignment across the sequence
    for t in range(seq_len - 1):
        assert x0[t + 1] == y0[t]

def test_create_dataloaders():
    tokens = list(range(100))
    train_loader, val_loader = create_dataloaders(
        token_ids=tokens,
        seq_len=16,
        batch_size=4,
        val_split=0.2,
    )
    assert train_loader is not None
    batch = next(iter(train_loader))
    x, y = batch
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
