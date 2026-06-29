import json
import re
import os
import argparse
from pathlib import Path

import os

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
    
PYTHON_CODE_INDICATORS = re.compile(
    r"^\s*(import |from |def |class |if |for |while |return |#|@)"
)

FENCED_CODE_CLOSED = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
FENCED_CODE_OPEN   = re.compile(r"```[^\n]*\n(.*)", re.DOTALL)

def extract_code(text: str) -> str:
    blocks = []

    closed_blocks = FENCED_CODE_CLOSED.findall(text)
    if closed_blocks:
        blocks.extend([b.strip() for b in closed_blocks])

    last_closed_end = 0
    for m in FENCED_CODE_CLOSED.finditer(text):
        last_closed_end = m.end()

    remaining = text[last_closed_end:]
    open_block = FENCED_CODE_OPEN.search(remaining)
    if open_block:
        open_content = open_block.group(1).strip()
        if open_content and open_content not in blocks:
            blocks.append(open_content)

    if blocks:
        return "\n\n".join(blocks)

    first_line = text.lstrip().split("\n")[0]
    if PYTHON_CODE_INDICATORS.match(first_line):
        return text.strip()
    
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if PYTHON_CODE_INDICATORS.match(line):
            return "\n".join(lines[i:]).strip()

    return ""



def main():
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total = 0
    extracted = 0
    empty = 0

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