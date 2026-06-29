import json
import re
import argparse
from pathlib import Path
from typing import Optional

ENDOFTEXT_RE = re.compile(r'<\|endoftext\|>', re.IGNORECASE)
GENERIC_TOKEN_RE = re.compile(r'<\|[^|]{1,40}\|>')

COPYRIGHT_LINE_RE = re.compile(
    r'^\s*(?:#|//|\*{1,2}|/\*)?\s*'
    r'(?:copyright|licensed\s+under|all\s+rights\s+reserved|'
    r'permission\s+is\s+hereby\s+granted|redistribution\s+and\s+use|'
    r'license:|this\s+file\s+is\s+part\s+of|this\s+program\s+is\s+free\s+software|'
    r'modification,\s+are\s+permitted|without\s+warranty)',
    re.IGNORECASE,
)

META_HEADER_RE = re.compile(
    r'^#\s*(?:!|[-*]{1,3}|author:|email:|created|date:|version:'
    r'|encoding|proto-rpy|copyright|license|vim:)',
    re.IGNORECASE,
)

NOISE_ONLY_RE = re.compile(r'^[\s#\-=*/<|>~^]+$')

FENCED_BLOCK_RE = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)

PYTHON_SIGNALS_RE = re.compile(
    r'\bdef\s+\w+\s*\('
    r'|\bclass\s+\w+(?:\s*\([^)]*\))?\s*:'
    r'|^from\s+[\w.]+\s+import\b'
    r'|^import\s+[\w.]+\b'
    r'|@\w+(?:\.\w+)*(?:\([^)]*\))?'
    r'|\bif\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:'
    r'|\breturn\b'
    r'|\braise\b'
    r'|\bwith\b'
    r'|\byield\b',
    re.MULTILINE,
)

NL_PROSE_RE = re.compile(
    r'^[A-Za-z][A-Za-z0-9 ,.\'\"\-:;!?()]{30,}$'
)

CJK_RE = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')

INCOMPLETE_TRAIL_RE = re.compile(r'[,(=\[{:+\-*/%&|^~<>\\]\s*$')


def _has_real_code(s: str) -> bool:
    return any(
        l.strip()
        and not l.strip().startswith('#')
        and not l.strip().startswith('"""')
        and not l.strip().startswith("'''")
        for l in s.splitlines()
    )


def _strip_leading_triple_quote(code: str) -> str:
    stripped = code.strip()
    for q in ('"""', "'''"):
        if stripped.startswith(q):
            after = stripped[len(q):]
            if after and (after[0] in ('\n', '\r', ' ', '\t') or
                          re.match(r'[\w\s]', after[0])):
                return after.lstrip('\n\r')
    return code


def _strip_leading_boilerplate(code: str) -> str:
    lines = code.splitlines(keepends=True)
    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            if start == i:
                start = i + 1
            continue
        if COPYRIGHT_LINE_RE.match(s) or META_HEADER_RE.match(s) or NOISE_ONLY_RE.match(s):
            start = i + 1
        else:
            break
    return ''.join(lines[start:]).strip()


def _strip_trailing_noise(code: str) -> str:
    lines = code.splitlines()
    end = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        s = lines[i].strip()
        if not s or NOISE_ONLY_RE.match(s):
            end = i
        else:
            break
    return '\n'.join(lines[:end]).rstrip()


def _strip_trailing_incomplete(code: str) -> str:
    lines = code.splitlines()
    removed = 0
    while lines and removed < 3:
        last = lines[-1].rstrip()
        if INCOMPLETE_TRAIL_RE.search(last):
            lines.pop()
            removed += 1
        else:
            break
    return '\n'.join(lines)


def _remove_cjk_lines(code: str) -> str:

    cleaned = []
    for line in code.splitlines():
        non_space = [c for c in line if not c.isspace()]
        if not non_space:
            cleaned.append(line)
            continue
        cjk_chars = sum(1 for c in non_space if CJK_RE.match(c))
        if cjk_chars / len(non_space) > 0.20:
            continue  
        cleaned.append(line)
    return '\n'.join(cleaned)


def _remove_prose_lines(code: str) -> str:
    py_punct = re.compile(r'[=(){}\[\]:,]|\bdef\b|\bclass\b|\breturn\b|\bimport\b|\bif\b|\bfor\b|\bwhile\b|\bwith\b|\braise\b')
    cleaned = []
    for line in code.splitlines():
        if line and line[0] in (' ', '\t'):
            cleaned.append(line)
            continue
        s = line.strip()
        if NL_PROSE_RE.match(s) and not py_punct.search(s):
            continue 
        cleaned.append(line)
    return '\n'.join(cleaned)


def _try_fenced_block(text: str) -> Optional[str]:
    matches = FENCED_BLOCK_RE.findall(text)
    if not matches:
        return None
    return max(matches, key=len).strip() or None


def _keyword_anchor(text: str) -> Optional[str]:
    m = re.search(
        r'(?:^|\n)(\s*(?:import\s|from\s[\w.]+\s+import|def\s+\w|class\s+\w|@\w))',
        text,
    )
    if m:
        return text[m.start():].strip()
    return None


def _normalise_whitespace(code: str) -> str:
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    code = re.sub(r'\n{4,}', '\n\n\n', code)
    return code.strip()


def clean_generated_code(raw: str) -> str:
    if not raw or not raw.strip():
        return ''

    code = _strip_leading_triple_quote(raw)

    code = ENDOFTEXT_RE.sub('', code)
    code = GENERIC_TOKEN_RE.sub('', code)

    fenced = _try_fenced_block(code)
    if fenced:
        code = fenced

    code = _remove_cjk_lines(code)

    code = _remove_prose_lines(code)

    code = _strip_leading_boilerplate(code)

    code = _strip_trailing_noise(code)

    code = _strip_trailing_incomplete(code)

    code = _normalise_whitespace(code)

    if not _has_real_code(code):
        anchored = _keyword_anchor(raw)
        if anchored:
            anchored = _strip_leading_boilerplate(anchored)
            anchored = _strip_trailing_noise(anchored)
            anchored = _strip_trailing_incomplete(anchored)
            anchored = _remove_cjk_lines(anchored)
            anchored = ENDOFTEXT_RE.sub('', anchored)
            anchored = _normalise_whitespace(anchored)
            if _has_real_code(anchored):
                return anchored

    if not code.strip():
        fallback = ENDOFTEXT_RE.sub('', raw)
        fallback = GENERIC_TOKEN_RE.sub('', fallback)
        fallback = _remove_cjk_lines(fallback)
        fallback = _strip_leading_boilerplate(fallback)
        fallback = _normalise_whitespace(fallback)
        return fallback

    return code



def clean_file(input_file: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        'total':         0,
        'good_python':   0,
        'only_comments': 0,
        'empty_result':  0,
        'had_leading_quote': 0,
        'had_cjk':       0,
    }

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            obj = json.loads(line)
            raw = obj.get('generated_code', '')

            stripped = raw.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                stats['had_leading_quote'] += 1
            if CJK_RE.search(raw):
                stats['had_cjk'] += 1

            cleaned = clean_generated_code(raw)

            stats['total'] += 1
            if not cleaned.strip():
                stats['empty_result'] += 1
            elif not _has_real_code(cleaned):
                stats['only_comments'] += 1
            else:
                stats['good_python'] += 1

            f_out.write(json.dumps({
                'prompt':         obj.get('prompt', ''),
                'generated_code': cleaned,
                'reference':      obj.get('reference', ''),
            }, ensure_ascii=False) + '\n')

    print(f'✔ Processing completed. File saved in: {output_file}')
    print(f'  Total records      : {stats["total"]}')
    print(f'  Good Python        : {stats["good_python"]}')
    print(f'  Only comments      : {stats["only_comments"]}')
    print(f'  Empty result       : {stats["empty_result"]}')
    print(f'  Had leading quote  : {stats["had_leading_quote"]}')
    print(f'  Had CJK chars      : {stats["had_cjk"]}')

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Clean generated_code fields in JSONL files from CodeGPT-small-py. '
            'Handles leading triple-quotes, CJK noise, NL prose, '
            'incomplete truncations, and boilerplate.'
        )
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSONL file')
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

    clean_file(input_file, output_file)


if __name__ == '__main__':
    main()