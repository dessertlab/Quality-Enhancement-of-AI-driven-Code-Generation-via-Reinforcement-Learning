import json
import re
import argparse
from pathlib import Path
from typing import Optional


LANG_TAG_RE = re.compile(
    r'<\|(?:python|java|javascript|cpp|c\+\+|go|c|rust|ruby|php|swift|kotlin|'
    r'ts|typescript|scala|r|bash|shell|sql|html|css|cs|csharp|haskell|'
    r'controller/config\.js|app|driver\.\w+|hil\.\w+)[^|>]{0,40}(?:\|>|>)',
    re.IGNORECASE,
)

PARTIAL_TAG_RE = re.compile(r'<\|[^\s<>]{1,60}(?:\|>|(?=\s))')
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

NON_PYTHON_PATTERNS = {
    'java': re.compile(
        r'\bpublic\s+(?:class|void|static|abstract|interface)\b'
        r'|import\s+(?:java|javax|org\.apache|org\.springframework)\.'
        r'|package\s+[\w.]+\s*;'
        r'|\bSystem\.out\.print'
    ),
    'cpp': re.compile(
        r'#include\s*[<"]'
        r'|\bstd::\w+'
        r'|::\w+\s*\('
        r'|\buint(?:8|16|32|64)_t\b'
    ),
    'go': re.compile(
        r'^package\s+\w+\s*$'
        r'|\bfunc\s+\w+\s*\('
        r'|\bfmt\.(?:Print|Println|Sprintf|Errorf)\b',
        re.MULTILINE,
    ),
    'javascript': re.compile(
        r'\bconst\s+\w+\s*='
        r'|\bvar\s+\w+\s*='
        r'|\bfunction\s*\w*\s*\('
        r'|\bexport\s+(?:default\s+)?(?:function|class|const)\b'
        r'|\brequire\s*\('
        r'|import\s+\{[^}]+\}\s+from\s+'
        r'|\bPromise\s*\.',
    ),
}

PYTHON_SIGNALS_RE = re.compile(
    r'\bdef\s+\w+\s*\('
    r'|\bclass\s+\w+(?:\s*\([^)]*\))?\s*:'
    r'|^from\s+[\w.]+\s+import\b'
    r'|^import\s+[\w.]+\b'
    r'|@\w+(?:\.\w+)*(?:\([^)]*\))?'
    r'|\bif\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:',
    re.MULTILINE,
)

FENCED_BLOCK_RE = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)

def _detect_language(seg: str) -> str:
    for lang, pat in NON_PYTHON_PATTERNS.items():
        if pat.search(seg):
            return lang
    return 'python'


def _score_segment(seg: str) -> float:
    if not seg.strip():
        return -100.0

    score = 0.0
    lines     = seg.splitlines()
    non_empty = [l for l in lines if l.strip()]

    score += min(len(non_empty), 40) * 0.15
    py_hits = len(PYTHON_SIGNALS_RE.findall(seg))
    score += py_hits * 3.0

    real_code = [
        l for l in non_empty
        if l.strip()
        and not l.strip().startswith('#')
        and not l.strip().startswith('"""')
        and not l.strip().startswith("'''")
    ]
    score += min(len(real_code), 20) * 0.3

    lang = _detect_language(seg)
    if lang != 'python':
        score -= 20.0

    copyright_count = sum(1 for l in non_empty if COPYRIGHT_LINE_RE.match(l))
    score -= copyright_count * 2.5

    if not real_code:
        score -= 10.0

    stripped = seg.strip()
    if (stripped.startswith(('"""', "'''"))
            and stripped.endswith(('"""', "'''"))
            and stripped.count('"""') + stripped.count("'''") == 2):
        score -= 8.0

    first_non_empty = next((l.strip() for l in lines if l.strip()), '')
    if first_non_empty and first_non_empty[0] in ')]},:':
        score -= 6.0

    if len(stripped) < 40:
        score -= 5.0

    noise_lines = sum(1 for l in non_empty if NOISE_ONLY_RE.match(l))
    score -= noise_lines * 0.5

    return score


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


def _remove_c_style_comments(text: str) -> str:
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    lines = []
    for line in text.splitlines():
        if re.match(r'^\s*//', line):
            continue
        lines.append(line)
    return '\n'.join(lines)


def _normalise_whitespace(code: str) -> str:
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    code = re.sub(r'\n{4,}', '\n\n\n', code)
    return code.strip()


def _has_real_code(s: str) -> bool:
    return any(
        l.strip()
        and not l.strip().startswith('#')
        and not l.strip().startswith('"""')
        and not l.strip().startswith("'''")
        for l in s.splitlines()
    )


def _best_segment(raw: str) -> str:
    parts = LANG_TAG_RE.split(raw)
    if len(parts) == 1:
        return raw

    candidates = []
    for i, part in enumerate(parts):
        if i == 0:
            if PYTHON_SIGNALS_RE.search(part):
                candidates.append(part)
        else:
            candidates.append(part)

    if not candidates:
        return raw

    return max(candidates, key=_score_segment)


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


def clean_generated_code(raw: str) -> str:
    if not raw or not raw.strip():
        return ''

    code = _best_segment(raw)

    code = _remove_c_style_comments(code)

    fenced = _try_fenced_block(code)
    if fenced:
        code = fenced

    code = _strip_leading_boilerplate(code)

    code = _strip_trailing_noise(code)

    code = LANG_TAG_RE.sub('', code)
    code = PARTIAL_TAG_RE.sub('', code)
    code = GENERIC_TOKEN_RE.sub('', code)

    code = _normalise_whitespace(code)

    if not _has_real_code(code):
        anchored = _keyword_anchor(raw)
        if anchored:
            anchored = _strip_leading_boilerplate(anchored)
            anchored = _strip_trailing_noise(anchored)
            anchored = LANG_TAG_RE.sub('', anchored)
            anchored = PARTIAL_TAG_RE.sub('', anchored)
            anchored = _normalise_whitespace(anchored)
            if _has_real_code(anchored):
                return anchored
    if not code.strip():
        fallback = _remove_c_style_comments(raw)
        fallback = LANG_TAG_RE.sub(' ', fallback)
        fallback = PARTIAL_TAG_RE.sub('', fallback)
        fallback = GENERIC_TOKEN_RE.sub('', fallback)
        fallback = _strip_leading_boilerplate(fallback)
        fallback = _normalise_whitespace(fallback)
        return fallback

    return code


def clean_file(input_file: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        'total':           0,
        'good_python':     0,
        'non_python_lang': 0,
        'only_comments':   0,
        'empty_result':    0,
    }

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            obj     = json.loads(line)
            raw     = obj.get('generated_code', '')
            cleaned = clean_generated_code(raw)

            stats['total'] += 1
            if not cleaned.strip():
                stats['empty_result'] += 1
            elif not _has_real_code(cleaned):
                stats['only_comments'] += 1
            elif _detect_language(cleaned) != 'python':
                stats['non_python_lang'] += 1
            else:
                stats['good_python'] += 1

            f_out.write(json.dumps({
                'prompt':         obj.get('prompt', ''),
                'generated_code': cleaned,
                'reference':      obj.get('reference', ''),
            }, ensure_ascii=False) + '\n')

    print(f'✔ Processing completed. File saved in: {output_file}')
    print(f'  Total records   : {stats["total"]}')
    print(f'  Good Python     : {stats["good_python"]}')
    print(f'  Non-Python lang : {stats["non_python_lang"]}')
    print(f'  Only comments   : {stats["only_comments"]}')
    print(f'  Empty result    : {stats["empty_result"]}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Clean generated_code fields in a JSONL file. '
            'Handles language tags, copyright boilerplate, '
            'non-Python noise, C-style comments, and more.'
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