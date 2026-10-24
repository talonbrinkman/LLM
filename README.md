# Custom LLM

**Custom LLM** is an autoregressive causal language model and training pipeline built from scratch in pure [PyTorch](https://pytorch.org/). Designed from foundational first principles, it implements a decoder-only Transformer architecture with causal self-attention, pre-layer normalization, and an end-to-end training and text generation system without relying on high-level wrapper libraries.

## 🚀 Key Features

- **From-Scratch Decoder-Only Transformer**: Pure PyTorch implementation of multi-head causal self-attention, feed-forward MLPs, and residual streams.
- **Strict Causal Masking**: Lower-triangular masking ensures the model strictly predicts the next token without leaking future context.
- **PyTorch 2.0 SDPA Acceleration**: Automatically utilizes fused memory-efficient kernels and FlashAttention when running on supported hardware.
- **Pre-RMSNorm Normalization**: Normalizes inputs prior to attention and MLP blocks to maintain stable gradient flow throughout deep layers.
- **Weight Tying**: Shares weights between token embeddings and the language model head, reducing total parameters and improving generalization.
- **Autoregressive Shifted Dataset**: Automatically formats continuous text into input-target pairs ($x_{0:T-1} \to x_{1:T}$) with sliding-window chunking.
- **Decoupled Weight Decay & Cosine Schedule**: Custom AdamW parameter grouping separates 2D weight matrices from 1D biases and normalization scales, paired with warmup and cosine decay.
- **Live Perplexity Tracking**: Monitors validation progress with both Cross-Entropy loss and Perplexity ($\text{PPL} = e^{\mathcal{L}}$).
- **Flexible Decoding Engine**: Supports greedy generation, temperature scaling, Top-K filtering, Top-P (nucleus) sampling, and repetition penalties.
- **Custom Tokenizer**: Frequency-calibrated tokenizer with reserved special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`), clean punctuation handling, and JSON persistence.
- **Interactive Assistant Shell**: Built-in interactive REPL (`Jarvis.py` / `generate.py`) for live prompt-and-response generation.
- **Unit Test Suite**: Full `pytest` coverage verifying tensor dimensions, loss computation, causality isolation, and tokenizer round-trips.

## 🛠️ Tech Stack

- **Deep Learning**: [Python 3.9+](https://www.python.org/), [PyTorch 2.0+](https://pytorch.org/)
- **Math & Utilities**: [NumPy](https://numpy.org/), [Rich](https://github.com/Textualize/rich)
- **Testing**: [pytest](https://pytest.org/)

## 🏗️ Architecture Overview

```
[Input Text] ──> [Tokenizer] ──> [Token IDs]
                                       │
                                       ▼
    ┌────────────────────────────────────────────────────────┐
    │  Token Embedding (W_e)  +  Learned Positional Embed    │
    └──────────────────────────┬─────────────────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │       Transformer Block (x N)       │
            │  ┌───────────────────────────────┐  │
            │  │ Pre-RMSNorm                   │  │
            │  │ Fused QKV Causal Attention    │  │
            │  │ Residual Add                  │  │
            │  ├───────────────────────────────┤  │
            │  │ Pre-RMSNorm                   │  │
            │  │ Position-wise MLP (GELU)      │  │
            │  │ Residual Add                  │  │
            │  └───────────────────────────────┘  │
            └──────────────────┬──────────────────┘
                               │
                               ▼
                    [Final RMSNorm (ln_f)]
                               │
                               ▼
        [Linear LM Head (Tied to Token Embedding)]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 [Cross-Entropy Loss]                  [Autoregressive Sampling]
  Shift: x[t] -> x[t+1]                 Temp / Top-K / Top-P
  Perplexity: exp(loss)                 Repetition Penalty
```

The model processes token sequences through learnable token and positional embeddings, passes them through $N$ Pre-RMSNorm Transformer blocks with causal attention masks, and projects the final hidden representations back into vocabulary logits via tied weights.

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- CUDA-compatible GPU (optional; trains and runs smoothly on CPU)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/talonbrinkman/custom-llm.git
   cd custom-llm
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the test suite:**
   ```bash
   python -m pytest tests -v
   ```

## 📖 Usage

### 1. Downloading Datasets
Fetch conversational or story datasets ready for training:

```bash
# Downloads DailyDialog (11,000+ turns formatted for User/Jarvis conversation)
python download_data.py --dataset dailydialog

# Or fetch classic benchmarks or stories
python download_data.py --dataset tinystories
python download_data.py --dataset all
```

#### `download_data.py` Parameters:
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--dataset` | `dailydialog` | Dataset to fetch: `dailydialog`, `tinystories`, `tinyshakespeare`, or `all` |
| `--force` | `False` | Re-download and overwrite existing files in `data/` |

---

### 2. Training
Train a model on your custom text corpus (or the downloaded dialogue data):

```bash
python train.py --data data/dailydialog.txt --epochs 20 --batch-size 16 --d-model 256 --n-layers 6 --n-heads 8
```

#### `train.py` Parameters:
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--data` | `data` | Path to a text file or directory of `.txt` files (scans all files if folder) |
| `--resume` | `False` | Resume training from existing checkpoint weights without restarting |
| `--checkpoint-dir` | `model_weights` | Target folder for saving weights, configs, and tokenizer |
| `--d-model` | `256` | Hidden dimension of embeddings and attention layers |
| `--n-layers` | `6` | Number of stacked Transformer decoder layers |
| `--n-heads` | `8` | Number of causal multi-head attention heads |
| `--seq-len` | `256` | Maximum context sequence window length |
| `--vocab-size` | `8000` | Maximum vocabulary size for the custom tokenizer |
| `--epochs` | `20` | Number of complete training epochs |
| `--batch-size` | `16` | Batch size per training step |
| `--lr` | `3e-4` | Peak learning rate for AdamW optimizer |
| `--weight-decay` | `0.01` | Decoupled weight decay coefficient (excludes 1D biases/norms) |
| `--warmup-steps` | `50` | Number of linear learning rate warmup steps |
| `--grad-clip` | `1.0` | Maximum gradient norm threshold to prevent exploding gradients |
| `--val-split` | `0.1` | Fraction of token data reserved for validation evaluation |
| `--device` | `auto` | Target compute device (`auto`, `cuda`, or `cpu`) |

Checkpoints and the fitted tokenizer are automatically saved to `model_weights/checkpoint.pt`, `model_weights/best_model.pt`, and `model_weights/tokenizer.json`.

---

### 3. Generating Text
Generate text from a prompt using trained weights:

```bash
python generate.py --prompt "User: Hello! How are you?" --temperature 0.7 --top-p 0.9 --max-tokens 100
```

#### `generate.py` Parameters:
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--prompt` | `None` | Input text prompt for one-shot text generation |
| `--interactive` | `False` | Launch interactive terminal REPL conversation shell |
| `--checkpoint` | `model_weights/checkpoint.pt` | Path to trained model weights checkpoint |
| `--tokenizer` | `None` | Path to custom `tokenizer.json` (auto-detected if omitted) |
| `--max-tokens` | `150` | Maximum number of new tokens to generate |
| `--temperature` | `0.7` | Sampling temperature (lower = deterministic, higher = creative) |
| `--top-k` | `50` | Truncate sampling distribution to top $k$ highest-probability tokens |
| `--top-p` | `0.9` | Nucleus sampling threshold (cumulative probability mass cutoff) |
| `--repetition-penalty` | `1.1` | Penalty applied to already-generated tokens to reduce repetition |

---

### 4. Interactive Shell & Jarvis
Start an interactive conversation session in your terminal:

```bash
python generate.py --interactive
```
Or launch via the Jarvis entrypoint:
```bash
python Jarvis.py
```

## 📁 Repository Structure

```
Custom LLM/
├── configs/
│   └── default.json          # Default model & training configuration
├── data/
│   ├── dailydialog.txt       # Formatted dialogue dataset
│   └── sample.txt            # Benchmark sample training corpus
├── model_weights/            # Checkpoints and serialized tokenizers
├── src/
│   ├── __init__.py           # Package exports
│   ├── config.py             # ModelConfig and TrainingConfig dataclasses
│   ├── model.py              # CustomTransformerLM, CausalSelfAttention, RMSNorm, MLP
│   ├── tokenizer.py          # Custom tokenizer with special tokens
│   ├── dataset.py            # Next-token autoregressive dataset and dataloaders
│   └── engine.py             # Trainer, AdamW optimizer, and cosine scheduler
├── tests/
│   ├── test_model.py         # Model shapes, causality isolation, and loss tests
│   ├── test_tokenizer.py     # Tokenizer fit, encode, decode, and round-trip tests
│   └── test_dataset.py       # Autoregressive shifting (x_t -> x_t+1) tests
├── download_data.py          # Automated dataset downloader & formatter
├── train.py                  # Primary training CLI
├── generate.py               # Generation CLI & interactive shell
├── Jarvis.py                 # Backwards-compatible interactive assistant
├── Trainer.py                # Backwards-compatible trainer script
├── pyproject.toml            # Python packaging and pytest configuration
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

## 🔮 Upcoming Improvements

- **Rotary Position Embeddings (RoPE)**: Relative position encoding for improved length generalization.
- **KV Cache Decoding**: Caching keys and values for $O(1)$ autoregressive token generation.
- **Byte-Pair Encoding (BPE)**: Subword BPE tokenizer for handling larger and more diverse vocabularies.
- **LoRA Fine-Tuning**: Low-Rank Adaptation module for efficient fine-tuning on downstream tasks.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.
