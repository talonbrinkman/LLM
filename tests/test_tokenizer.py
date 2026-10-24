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
