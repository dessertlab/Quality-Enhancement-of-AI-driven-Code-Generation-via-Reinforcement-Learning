import json
import re
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", required=True, help="Path to the input JSONL file.")
args = parser.parse_args()

input_file = Path(args.input_file).resolve()


script_dir    = Path(__file__).resolve().parent          
results_dir   = script_dir.parent                        
inference_dir = results_dir / "inference"                
cleaned_dir   = results_dir / "inference_cleaned"        

try:
    rel = input_file.relative_to(inference_dir)
except ValueError:
    rel = Path(input_file.name)

output_file = cleaned_dir / rel.parent / f"{input_file.stem}_cleaned.jsonl"
output_file.parent.mkdir(parents=True, exist_ok=True)

FENCED_CODE_CLOSED = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
FENCED_CODE_OPEN   = re.compile(r"```[^\n]*\n(.*)",     re.DOTALL)

PYTHON_CODE_INDICATORS = re.compile(
    r"^\s*(import |from |def |class |if |for |while |return |#|@)"
)

QWEN_INLINE_MARKERS = re.compile(
    r"\[(?:CODE(?:\s+COMPLETION\s+STARTS)?|BEGIN\s+CODE|BEGIN\s+TEMPLATE|"
    r"START\s+OF\s+TEMPLATE|TEMPLATE\s+BEGIN)\]",
    re.IGNORECASE,
)

PROSE_PREFIXES = re.compile(
    r"^(?:here(?:'s| is)|below is|sure(?:,| !)|certainly(?:,)?|"
    r"this (?:is|file)|fill the remaining|the following)[^\n]*\n",
    re.IGNORECASE,
)

CHATML_RESIDUALS = re.compile(
    r"<\|im_(?:start|end)\|>(?:system|user|assistant)?\n?|Assistant:\s*"
)

GARBAGE_PATTERN = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff]{4,}|" 
    r"(?:[\U0001F300-\U0001FFFF]){3,}"                                
)


def strip_prose_prefix(text: str) -> str:
    m = QWEN_INLINE_MARKERS.search(text)
    if m:
        return text[m.end():].lstrip("\n ")

    m = PROSE_PREFIXES.match(text)
    if m:
        return text[m.end():]

    return text


def extract_code(text: str) -> str:
    text = CHATML_RESIDUALS.sub("", text).strip()

    if not text:
        return ""

    text_stripped = strip_prose_prefix(text)

    blocks = []

    closed_blocks = FENCED_CODE_CLOSED.findall(text_stripped)
    if closed_blocks:
        blocks.extend([b.strip() for b in closed_blocks])

    last_closed_end = 0
    for m in FENCED_CODE_CLOSED.finditer(text_stripped):
        last_closed_end = m.end()
    remaining = text_stripped[last_closed_end:]
    open_block = FENCED_CODE_OPEN.search(remaining)
    if open_block:
        open_content = open_block.group(1).strip()
        if open_content and open_content not in blocks:
            blocks.append(open_content)

    if blocks:
        return "\n\n".join(blocks)
    
    candidate = text_stripped if text_stripped else text

    first_line = candidate.lstrip().split("\n")[0]
    if PYTHON_CODE_INDICATORS.match(first_line):
        return candidate.strip()

    lines = candidate.split("\n")
    for i, line in enumerate(lines):
        if PYTHON_CODE_INDICATORS.match(line):
            return "\n".join(lines[i:]).strip()
    return ""


def main():
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total     = 0
    extracted = 0
    empty     = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total += 1

            raw_generated = record.get("generated_code", "")
            cleaned_code  = extract_code(raw_generated)

            if cleaned_code:
                extracted += 1
            else:
                empty += 1

            record["generated_code"] = cleaned_code

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Total records  : {total}")
    print(f"Code extracted : {extracted}")
    print(f"No code found  : {empty}")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()