import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Optional


def format_record(
    item: Dict[str, Any],
    assistant_name: str = "Jarvis",
    format_mode: str = "special",
) -> Optional[str]:
    """
    Intelligently formats a JSON record into clean conversation or narrative text
    based on the keys present in the dictionary.
    Supports special token format (<|user|>, <|assistant|>, <|endoftext|>) or plain text.
    """
    if not isinstance(item, dict):
        return None

    use_special = format_mode == "special"

    # 1. Multi-turn chat formats: 'messages' or 'conversations' (ShareGPT / OpenAI style)
    chat_list = item.get("messages") or item.get("conversations")
    if isinstance(chat_list, list) and len(chat_list) > 0:
        turns = []
        for turn in chat_list:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role") or turn.get("from", "")
            content = turn.get("content") or turn.get("value", "")
            content = str(content).strip()
            if not content:
                continue

            role_lower = str(role).lower()
            if use_special:
                if any(k in role_lower for k in ["user", "human"]):
                    speaker = "<|user|>"
                elif any(k in role_lower for k in ["assistant", "gpt", "bot", "model"]):
                    speaker = "<|assistant|>"
                elif "system" in role_lower:
                    speaker = "<|system|>"
                else:
                    speaker = "<|user|>"
                turns.append(f"{speaker}\n{content}")
            else:
                if any(k in role_lower for k in ["user", "human"]):
                    speaker = "User"
                elif any(k in role_lower for k in ["assistant", "gpt", "bot", "model"]):
                    speaker = assistant_name
                elif "system" in role_lower:
                    speaker = "System"
                else:
                    speaker = str(role).capitalize()
                turns.append(f"{speaker}: {content}")

        if turns:
            eos = "<|endoftext|>" if use_special else ""
            return "\n".join(turns) + eos

    # 2. Alpaca / Dolly / Instruction format
    if "instruction" in item:
        instruction = str(item.get("instruction", "")).strip()
        context = str(item.get("input") or item.get("context", "")).strip()
        response = str(item.get("output") or item.get("response", "")).strip()

        if instruction and response:
            if context:
                prompt = f"{instruction}\nContext: {context}"
            else:
                prompt = instruction
            if use_special:
                return f"<|user|>\n{prompt}\n<|assistant|>\n{response}<|endoftext|>"
            else:
                return f"User: {prompt}\n{assistant_name}: {response}"

    # 3. Question & Answer format
    if "question" in item and "answer" in item:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            if use_special:
                return f"<|user|>\n{q}\n<|assistant|>\n{a}<|endoftext|>"
            else:
                return f"User: {q}\n{assistant_name}: {a}"

    # 4. Prompt & Completion format
    if "prompt" in item and "completion" in item:
        p = str(item.get("prompt", "")).strip()
        c = str(item.get("completion", "")).strip()
        if p and c:
            if use_special:
                return f"<|user|>\n{p}\n<|assistant|>\n{c}<|endoftext|>"
            else:
                return f"User: {p}\n{assistant_name}: {c}"

    # 5. Raw text / article / story format
    for text_key in ["text", "story", "content", "article", "body", "dialogue"]:
        if text_key in item:
            val = str(item.get(text_key, "")).strip()
            if val:
                return f"{val}\n<|endoftext|>" if use_special else val

    return None


def read_json_or_jsonl(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Yields dictionaries from either a standard JSON or JSONL file."""
    is_jsonl = file_path.suffix.lower() == ".jsonl"

    if is_jsonl:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    else:
        # Standard .json file (can be a list, or jsonlines with .json extension)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item
                elif isinstance(data, dict):
                    # Could be {"data": [...]} or a single object
                    if "data" in data and isinstance(data["data"], list):
                        for item in data["data"]:
                            yield item
                    else:
                        yield data
            except json.JSONDecodeError:
                # Fallback: attempt reading as line-delimited JSON
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def convert_file(
    file_path: Path,
    output_file: Path,
    assistant_name: str = "Jarvis",
    format_mode: str = "special",
) -> int:
    """Converts a single JSON or JSONL file to formatted TXT."""
    count = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for item in read_json_or_jsonl(file_path):
            formatted = format_record(item, assistant_name=assistant_name, format_mode=format_mode)
            if formatted:
                out.write(formatted + "\n\n")
                count += 1
    return count


def upgrade_plain_txt_file(
    file_path: Path,
    assistant_name: str = "Jarvis",
) -> int:
    """
    Upgrades a plain-formatted dialogue .txt file (User: ... / Jarvis: ...)
    in-place to special token format (<|user|>, <|assistant|>, <|endoftext|>).
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if "<|user|>" in text and "<|assistant|>" in text:
        return 0

    assistant_marker = f"\n{assistant_name}:"
    if not text.lstrip().startswith("User:") or assistant_marker not in text:
        return 0

    raw_blocks = re.split(r"(?:^|\n\n)(?=User:\s)", text.strip())
    merged_blocks: List[str] = []
    for block in raw_blocks:
        if not block.strip():
            continue
        if assistant_marker in block or not merged_blocks:
            merged_blocks.append(block)
        else:
            merged_blocks[-1] = merged_blocks[-1] + "\n\n" + block

    converted_records: List[str] = []
    for block in merged_blocks:
        if assistant_marker not in block:
            continue
        user_part, assistant_part = block.split(assistant_marker, 1)
        user_clean = re.sub(r"^User:\s*", "", user_part.strip()).strip()
        assistant_clean = assistant_part.strip()
        if user_clean and assistant_clean:
            converted_records.append(
                f"<|user|>\n{user_clean}\n<|assistant|>\n{assistant_clean}<|endoftext|>"
            )

    if converted_records:
        file_path.write_text("\n\n".join(converted_records) + "\n\n", encoding="utf-8")

    return len(converted_records)


def main():
    parser = argparse.ArgumentParser(
        description="Convert all JSON/JSONL and plain dialogue .txt datasets in 'data/' to clean .txt files for LLM training."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing dataset files (default: 'data').",
    )
    parser.add_argument(
        "--assistant-name",
        type=str,
        default="Jarvis",
        help="Name to use for the assistant responses (default: 'Jarvis').",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="special",
        choices=["special", "plain"],
        help="Formatting style: 'special' (<|user|>, <|assistant|>) or 'plain' (User: / Jarvis:). Default: 'special'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .txt files if they already exist.",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Directory '{data_dir}' does not exist.")
        return

    # Find all .json and .jsonl files
    candidates = sorted(list(data_dir.glob("*.json")) + list(data_dir.glob("*.jsonl")))
    converted_total = 0

    if candidates:
        print(f"\nFound {len(candidates)} JSON/JSONL file(s) in '{data_dir}':")
        print("-" * 65)

        for file_path in candidates:
            out_name = file_path.stem + ".txt"
            out_path = data_dir / out_name

            if out_path.exists() and not args.force:
                print(f"[Skip] '{out_name}' already exists. Pass --force to overwrite.")
                continue

            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"Converting '{file_path.name}' ({size_mb:.2f} MB) -> '{out_name}'...")

            try:
                count = convert_file(
                    file_path,
                    out_path,
                    assistant_name=args.assistant_name,
                    format_mode=args.format,
                )
                out_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"  --> Saved {count:,} records ({out_mb:.2f} MB) to '{out_name}'.")
                converted_total += 1
            except Exception as e:
                print(f"  [Error] Failed to convert {file_path.name}: {e}")

    # Also upgrade any plain User:/Jarvis: .txt files when special format is requested
    if args.format == "special":
        txt_files = sorted(data_dir.glob("*.txt"))
        for txt_path in txt_files:
            try:
                upgraded = upgrade_plain_txt_file(txt_path, assistant_name=args.assistant_name)
                if upgraded > 0:
                    out_mb = txt_path.stat().st_size / (1024 * 1024)
                    print(
                        f"Upgraded plain dialogue file '{txt_path.name}' -> "
                        f"{upgraded:,} special-token records ({out_mb:.2f} MB)."
                    )
                    converted_total += 1
            except Exception as e:
                print(f"  [Error] Failed to upgrade {txt_path.name}: {e}")

    if not candidates and converted_total == 0:
        print(f"No unconverted JSON/JSONL or plain dialogue .txt files found in '{data_dir}'.")
        return

    print("-" * 65)
    print(f"Completed! Processed {converted_total} file(s).")
    print("All .txt files in 'data/' are now ready for training with 'python train.py'.\n")


if __name__ == "__main__":
    main()
