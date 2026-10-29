import tempfile
from pathlib import Path
from src.tokenizer import Tokenizer

def test_tokenizer_fit_and_encode():
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Deep learning and transformer architectures.",
    ]
    tok = Tokenizer(vocab_size=50)
    tok.fit(texts)

    assert len(tok) > 4  # special tokens + words
    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.bos_id == 2
    assert tok.eos_id == 3

    encoded = tok.encode("quick brown fox")
    assert len(encoded) == 3
    assert all(isinstance(i, int) for i in encoded)

def test_tokenizer_decode_roundtrip():
    texts = ["transformer neural networks are powerful."]
    tok = Tokenizer(vocab_size=50)
    tok.fit(texts)

    encoded = tok.encode("transformer neural networks")
    decoded = tok.decode(encoded)
    assert decoded == "transformer neural networks"

def test_tokenizer_unknown_token():
    tok = Tokenizer(vocab_size=10)
    tok.fit(["hello world"])

    encoded = tok.encode("unseen_word")
    assert encoded == [tok.unk_id]

def test_tokenizer_save_load():
    tok = Tokenizer(vocab_size=50)
    tok.fit(["testing tokenizer serialization"])

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "tokenizer.json"
        tok.save(save_path)
        assert save_path.exists()

        loaded_tok = Tokenizer.load(save_path)
        assert len(loaded_tok) == len(tok)
        assert loaded_tok.encode("testing") == tok.encode("testing")

def test_bpe_tokenizer_casing_newlines_and_special_tokens():
    texts = [
        "<|user|>\nHello, my name is Talon!\n<|assistant|>\nNice to meet you, Talon!<|endoftext|>"
    ]
    tok = Tokenizer(vocab_size=500, use_bpe=True)
    tok.fit(texts)

    assert tok.pad_id == 0
    assert tok.user_id == 4
    assert tok.assistant_id == 5
    assert tok.eot_id == 7
    assert tok.encode("<|endoftext|>") == [tok.eot_id]
    assert tok.encode("<|user|>") == [tok.user_id]
    assert tok.encode("<|assistant|>") == [tok.assistant_id]

    sample = "<|user|>\nWhat is your name?\n<|assistant|>\nI'm Jarvis!<|endoftext|>"
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded, skip_special_tokens=False)
    assert decoded == sample
    assert "Talon" not in tok.decode(tok.encode("Talon")) or "Talon" == tok.decode(tok.encode("Talon"))

def test_word_level_tokenizer_preserves_special_tokens():
    texts = ["<|user|>\nHello world\n<|assistant|>\nHi there<|endoftext|>"]
    tok = Tokenizer(vocab_size=50, use_bpe=False)
    tok.fit(texts)

    encoded = tok.encode("<|user|>\nHello\n<|assistant|>\nHi<|endoftext|>")
    assert encoded[0] == tok.user_id
    assert tok.assistant_id in encoded
    assert encoded[-1] == tok.eot_id
