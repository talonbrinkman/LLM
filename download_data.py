import argparse
import io
import os
from pathlib import Path
import re
import urllib.request
import zipfile

DATA_DIR = Path("data")

DATASETS = {
    "dailydialog": {
        "filename": "dailydialog.txt",
        "description": "DailyDialog: ~11,000 multi-turn human conversations formatted as User/Jarvis dialogue.",
        "url": "https://huggingface.co/datasets/roskoN/dailydialog/resolve/main/train.zip",
        "type": "zip_dailydialog",
    },
    "tinystories": {
        "filename": "tinystories.txt",
        "description": "TinyStories: ~22 MB of coherent, synthetically generated simple English stories.",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
        "type": "direct_text",
    },
    "tinyshakespeare": {
        "filename": "tinyshakespeare.txt",
        "description": "TinyShakespeare: ~1 MB classic benchmark text for fast 30-second sanity checks.",
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "type": "direct_text",
    },
}

def download_file(url: str, desc: str = "Downloading") -> bytes:
    """Downloads content with a progress bar."""
    print(f"Connecting to: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Custom-LLM-Downloader/1.0"}
    )
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 64 * 1024
        chunks = []

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                print(f"\r  [{desc}] {mb_down:.2f} MB / {mb_tot:.2f} MB ({percent:.1f}%)", end="", flush=True)
            else:
                mb_down = downloaded / (1024 * 1024)
                print(f"\r  [{desc}] {mb_down:.2f} MB downloaded", end="", flush=True)

        print()  # newline
        return b"".join(chunks)

def process_dailydialog(zip_bytes: bytes, target_path: Path, format_mode: str = "special") -> None:
    """Extracts raw DailyDialog utterances and reformats them with special tokens or User/Jarvis dialogue turns."""
    print("  Processing DailyDialog into conversational turn format...")
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    raw_text = zf.read("train/dialogues_train.txt").decode("utf-8", errors="ignore")

    formatted_conversations = []
    lines = raw_text.splitlines()
    use_special = format_mode == "special"

    for line in lines:
        turns = [t.strip() for t in line.split("__eou__") if t.strip()]
        if not turns:
            continue

        conv_lines = []
        for i, turn in enumerate(turns):
            # Clean spaces before punctuation and fix contractions
            clean_turn = re.sub(r"\s+([.,!?:;'\"])", r"\1", turn)
            clean_turn = re.sub(r"'\s+([a-zA-Z])", r"'\1", clean_turn)
            clean_turn = clean_turn.replace("’", "'").replace("‘", "'")

            if use_special:
                speaker = "<|user|>" if i % 2 == 0 else "<|assistant|>"
                conv_lines.append(f"{speaker}\n{clean_turn}")
            else:
                speaker = "User" if i % 2 == 0 else "Jarvis"
                conv_lines.append(f"{speaker}: {clean_turn}")

        if use_special:
            formatted_conversations.append("\n".join(conv_lines) + "<|endoftext|>")
        else:
            formatted_conversations.append("\n".join(conv_lines))

    output = "\n\n".join(formatted_conversations)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"  Saved {len(formatted_conversations):,} dialogues ({len(output):,} chars) to {target_path}")

def download_dataset(name: str, force: bool = False, format_mode: str = "special") -> None:
    if name not in DATASETS:
        print(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
        return

    info = DATASETS[name]
    target_path = DATA_DIR / info["filename"]

    if target_path.exists() and not force:
        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"[Skip] '{info['filename']}' already exists ({size_mb:.2f} MB). Pass --force to re-download.")
        return

    print(f"\n=== Fetching: {name.upper()} ===")
    print(f"Info: {info['description']}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        content = download_file(info["url"], desc=name)

        if info["type"] == "zip_dailydialog":
            process_dailydialog(content, target_path, format_mode=format_mode)
        elif info["type"] == "direct_text":
            with open(target_path, "wb") as f:
                f.write(content)
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"  Saved {size_mb:.2f} MB to {target_path}")

        print(f"[OK] Successfully prepared: {target_path}")
    except Exception as e:
        print(f"[Error] Failed to download {name}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare text datasets for Custom LLM training."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dailydialog",
        choices=["dailydialog", "tinystories", "tinyshakespeare", "all"],
        help="Dataset name to download (default: dailydialog)."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="special",
        choices=["special", "plain"],
        help="Dialogue format: 'special' (<|user|>, <|assistant|>) or 'plain' (User: / Jarvis:). Default: special."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing files."
    )

    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASETS.keys():
            download_dataset(ds, force=args.force, format_mode=args.format)
    else:
        download_dataset(args.dataset, force=args.force, format_mode=args.format)

if __name__ == "__main__":
    main()
