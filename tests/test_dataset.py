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

def test_clean_text_preserves_newlines():
    from src.dataset import clean_text
    raw = "User: Hello\r\n\tJarvis:   Hi there! \n\n\nUser: How are you?"
    cleaned = clean_text(raw)
    assert "\n" in cleaned
    assert "User: Hello\nJarvis: Hi there!\n\nUser: How are you?" == cleaned

def test_sft_prompt_masking():
    # User token = 4, Assistant token = 5
    tokens = [4, 10, 20, 5, 30, 40, 1, 4, 50, 5, 60, 70]
    ds = CausalLMDataset(tokens, seq_len=6, stride=1, user_id=4, assistant_id=5)
    x0, y0 = ds[0]
    # x0: [4, 10, 20, 5, 30, 40]
    # In y0: 10, 20 (user prompt) and 5 (transition) should be masked to -100
    # 30, 40 (assistant reply) should NOT be masked
    assert y0[0].item() == -100
    assert y0[1].item() == -100
    assert y0[2].item() == -100
    assert y0[3].item() == -100
    assert y0[4].item() == 40

    # Window starting mid-prompt (index 1 -> token 10) must also mask prompt tokens
    x1, y1 = ds[1]
    assert x1[0].item() == 10
    assert y1[0].item() == -100
    assert y1[1].item() == -100
    assert y1[2].item() == -100
    assert y1[3].item() == 40
