#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Characterization of the post-processed model generations.

Decomposes the *error* category into empty outputs, prompt-template
echo, truncation-suspect generations, and other syntax errors, and
compares output lengths and empty/trivial rates between quality-only
and composite PPO reward configurations.

Expected input layout (paths are resolved relative to this script):
    results/inference_cleaned/**/
        {Model}_{config}_inference[_test_{secure|insecure}][_{reward}]_cleaned.jsonl
with config in {pretrained, finetuned, dpo, sft_dpo, ppo, sft_ppo}
(dpo/ppo denote standalone RL applied to the pre-trained model) and
reward in {pylint, semgrep, custom_pylint, custom_semgrep}.

Outputs (inventory.csv, error_breakdown.csv, length_by_reward.csv,
summary.txt) are written to results/.
Note: parseability is assessed with Python's `ast` module, a stricter
criterion than the tolerant parsing of the evaluation pipeline; error
totals are therefore upper bounds of the error rates in the paper.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd

# =====================================================================
# CONFIGURATION
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent     # results/scripts
RESULTS_DIR = SCRIPT_DIR.parent                   # results
GENERATIONS_DIR = RESULTS_DIR / "inference_cleaned"
FILES_GLOB = "*_cleaned.jsonl"        # searched recursively
CLEANED = True

OUT_DIR = RESULTS_DIR
MAX_TARGET_TOKENS = 512
TRIVIAL_TOKEN_THRESHOLD = 10


USE_HF_TOKENIZERS = False
HF_TOKENIZER_IDS = {
    "CodeGPT":   "microsoft/CodeGPT-small-py",
    "CodeGen":   "Salesforce/codegen-350M-multi",
    "QwenCoder": "Qwen/Qwen2.5-Coder-0.5B",
    "DeepSeek":  "deepseek-ai/deepseek-coder-1.3b-instruct",
}

# --- filename parsing -----------------------------------------------
FILENAME_RE = re.compile(
    r"^(?P<model>CodeGPT|CodeGen|QwenCoder|DeepSeek)_"
    r"(?P<train>.+?)_inference"
    r"(?:_test_(?P<split>secure|insecure))?"
    r"(?:_(?P<reward>custom_pylint|custom_semgrep|pylint|semgrep))?"
    r"(?:_cleaned)?\.jsonl$"
)
REWARD_MAP = {"pylint": "Rp", "semgrep": "Rs",
              "custom_pylint": "Rcp", "custom_semgrep": "Rcs"}
TRAIN_ALIASES = {
    "pretrained": "pretrained",
    "finetuned": "sft", "sft": "sft",
    "dpo": "dpo", "ppo": "ppo",          # standalone RL
    "sft_dpo": "sft_dpo", "sft_ppo": "sft_ppo",
}
GENERATION_KEYS = ("generated_code", "generated", "output")

EXCLUDE_SUBSTRINGS = ("bleu", "correctness", "edit", "bertscore", "qiqo",
                      "similarity")

QUALITY_ONLY_REWARDS = {"Rp", "Rs"}
COMPOSITE_REWARDS = {"Rcp", "Rcs"}
PPO_CONFIG_RE = re.compile(r"^(sft_)?ppo_(Rcp|Rcs|Rp|Rs)$")


# =====================================================================
# Loading
# =====================================================================

def parse_filename(name: str):
    m = FILENAME_RE.match(name)
    if not m:
        return None
    train = TRAIN_ALIASES.get(m.group("train"))
    if train is None:
        print(f"  WARNING: unknown training prefix in '{name}' "
              f"('{m.group('train')}') -- add it to TRAIN_ALIASES")
        return None
    reward = m.group("reward")
    if reward:
        config = f"{train}_{REWARD_MAP[reward]}" if train in ("ppo", "sft_ppo") \
            else None
        if config is None:
            print(f"  WARNING: reward on a non-PPO config in '{name}'")
            return None
    else:
        config = train
    split = m.group("split")
    test_set = "ppo" if split else "dpo"
    return dict(model=m.group("model"), config=config,
                test_set=test_set, split=split or "")


def load_generations(directory: Path | None = None,
                     pattern: str | None = None) -> pd.DataFrame:
    directory = Path(directory) if directory is not None else GENERATIONS_DIR
    pattern = pattern if pattern is not None else FILES_GLOB
    rows, inventory = [], []
    files = sorted(Path(directory).rglob(pattern))
    if not files:
        sys.exit(f"No '{pattern}' files in {directory} (recursive search)")
    for f in files:
        if any(s in f.name.lower() for s in EXCLUDE_SUBSTRINGS):
            print(f"  EXCLUDED (correctness-only reward, out of the paper's "
                  f"scope): {f.relative_to(directory)}")
            continue
        meta = parse_filename(f.name)
        if meta is None:
            print(f"  SKIP (unrecognized name): {f.name}")
            continue
        n = 0
        with f.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                gen = next((d[k] for k in GENERATION_KEYS if k in d), None)
                if gen is None:
                    sys.exit(f"Generation field not found in {f.name} "
                             f"(keys: {list(d)})")
                pid = f"{meta['split']}_{i}" if meta["split"] else str(i)
                rows.append(dict(**{k: meta[k] for k in
                                    ("model", "config", "test_set", "split")},
                                 prompt_id=pid,
                                 prompt=d.get("prompt", ""),
                                 generation=gen,
                                 reference=d.get("reference", "")))
                n += 1
        inventory.append(dict(file=f.name, n_rows=n, **meta))

    inv = pd.DataFrame(inventory)
    print("\nFILE INVENTORY (verify the mapping!):")
    print(inv.to_string(index=False))
    dup = inv.duplicated(subset=["model", "config", "test_set", "split"],
                         keep=False)
    if dup.any():
        print("\n  ERROR: (model, config, test_set, split) combinations "
              "DUPLICATED across different files:")
        print(inv.loc[dup, ["file", "model", "config", "test_set",
                            "split"]].to_string(index=False))
        sys.exit("Resolve the duplicates (is the distinction perhaps only "
                 "in the folder? Rename the files or adapt the parser).")
    for _, r in inv.iterrows():
        exp = 600 if r["test_set"] == "ppo" else 100
        if r["n_rows"] != exp:
            print(f"  WARNING: {r['file']} has {r['n_rows']} rows "
                  f"(expected {exp})")
    OUT_DIR.mkdir(exist_ok=True)
    inv.to_csv(OUT_DIR / "inventory.csv", index=False)
    return pd.DataFrame(rows)


def check_prompt_alignment(df: pd.DataFrame) -> None:
    print("\nPROMPT ALIGNMENT CHECK (required by the paired "
          "statistical analysis):")
    ok = True
    for (model, test_set, split), grp in df.groupby(
            ["model", "test_set", "split"]):
        seqs = {cfg: tuple(g.sort_values("prompt_id", key=lambda s: s.map(
                    lambda x: int(str(x).split("_")[-1])))["prompt"])
                for cfg, g in grp.groupby("config")}
        if len(seqs) < 2:
            continue
        base_cfg, base_seq = next(iter(seqs.items()))
        for cfg, seq in seqs.items():
            if seq != base_seq:
                ok = False
                print(f"  MISALIGNED: {model}/{test_set}/{split or '-'} "
                      f"{cfg} vs {base_cfg}")
    if ok:
        print("  OK: prompt order identical across the loaded configs "
              "(within each model).")


# =====================================================================
# Token counting
# =====================================================================

_TOKENIZERS: dict = {}
_APPROX_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str, model: str) -> int:
    if USE_HF_TOKENIZERS:
        if model not in _TOKENIZERS:
            from transformers import AutoTokenizer
            _TOKENIZERS[model] = AutoTokenizer.from_pretrained(
                HF_TOKENIZER_IDS[model], trust_remote_code=True)
        return len(_TOKENIZERS[model](text)["input_ids"])
    return len(_APPROX_TOKEN_RE.findall(text))


# =====================================================================
# Classification
# =====================================================================

_FULL_MARKERS = ("[CODE TEMPLATE", "### Instruction", "### Response", "```")
_PARTIAL_MARKER_TAIL = re.compile(r"\[CODE(\s+T[A-Z]*)?\s*$")


def _has_prompt_echo(text: str) -> bool:
    if any(m in text for m in _FULL_MARKERS):
        return True
    return bool(_PARTIAL_MARKER_TAIL.search(text[-30:]))


def _ends_abruptly(text: str) -> bool:
    for o, c in (("(", ")"), ("[", "]"), ("{", "}")):
        if text.count(o) != text.count(c):
            return True
    if text.count('"""') % 2 or text.count("'''") % 2:
        return True
    return False


def classify_row(text: str, model: str) -> dict:
    if text is None or not text.strip():
        return dict(err_class="empty", n_tokens=0, is_trivial=False)
    n_tok = count_tokens(text, model)
    trivial = n_tok <= TRIVIAL_TOKEN_THRESHOLD
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(text)
        parseable = True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        parseable = False
    if parseable:
        return dict(err_class="valid", n_tokens=n_tok, is_trivial=trivial)
    if CLEANED:
        if _has_prompt_echo(text):
            cls = "prompt_echo"
        elif _ends_abruptly(text) or n_tok >= MAX_TARGET_TOKENS:
            cls = "truncation_suspect"
        else:
            cls = "syntax_error"
    else:
        if "```" in text:
            cls = "markdown_fence"
        elif n_tok >= MAX_TARGET_TOKENS:
            cls = "truncated"
        else:
            cls = "syntax_error"
    return dict(err_class=cls, n_tokens=n_tok,
                is_trivial=trivial if cls != "empty" else False)


ERROR_CLASSES_CLEANED = ["empty", "prompt_echo", "truncation_suspect",
                         "syntax_error"]
ERROR_CLASSES_RAW = ["empty", "markdown_fence", "truncated", "syntax_error"]


# =====================================================================
# Per-configuration breakdown
# =====================================================================

def error_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    err_classes = ERROR_CLASSES_CLEANED if CLEANED else ERROR_CLASSES_RAW
    g = (df.groupby(["model", "config", "test_set", "err_class"])
           .size().rename("n").reset_index())
    tot = (df.groupby(["model", "config", "test_set"])
             .size().rename("total").reset_index())
    g = g.merge(tot, on=["model", "config", "test_set"])
    g["pct"] = (100 * g["n"] / g["total"]).round(1)
    wide = g.pivot_table(index=["model", "config", "test_set"],
                         columns="err_class", values="pct",
                         fill_value=0.0).reset_index()
    for col in err_classes + ["valid"]:
        if col not in wide.columns:
            wide[col] = 0.0
    wide["error_total"] = wide[err_classes].sum(axis=1).round(1)
    return wide[["model", "config", "test_set"] + err_classes
                + ["error_total", "valid"]]


# =====================================================================
# Lengths/empty outputs by reward type
# =====================================================================

def _parse_ppo_config(config: str):
    m = PPO_CONFIG_RE.match(config)
    if not m:
        return None
    pipeline = "sft_ppo" if m.group(1) else "ppo"
    reward = m.group(2)
    rtype = "quality_only" if reward in QUALITY_ONLY_REWARDS else "composite"
    return pipeline, reward, rtype


def reward_hacking_analysis(df: pd.DataFrame) -> pd.DataFrame:
    ppo = df[df["config"].str.match(PPO_CONFIG_RE)].copy()
    if ppo.empty:
        return pd.DataFrame()
    parsed = ppo["config"].map(_parse_ppo_config)
    ppo["pipeline"] = parsed.map(lambda t: t[0])
    ppo["reward_type"] = parsed.map(lambda t: t[2])
    rows = []
    keys = ["model", "pipeline", "reward_type", "test_set"]
    for key, grp in ppo.groupby(keys):
        nonempty = grp[grp["err_class"] != "empty"]
        rows.append(dict(zip(keys, key),
                         n=len(grp),
                         pct_empty=round(100 * (grp["err_class"] == "empty")
                                         .mean(), 1),
                         pct_trivial=round(100 * grp["is_trivial"].mean(), 1),
                         median_tokens=(float(nonempty["n_tokens"].median())
                                        if len(nonempty) else 0.0),
                         mean_tokens=(round(float(nonempty["n_tokens"].mean()),
                                            1) if len(nonempty) else 0.0)))
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def length_test_quality_vs_composite(df: pd.DataFrame) -> list:
    lines = []
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return ["  [scipy not installed: Mann-Whitney skipped]"]
    ppo = df[df["config"].str.match(PPO_CONFIG_RE)].copy()
    if ppo.empty:
        return ["  [no PPO configs loaded]"]
    parsed = ppo["config"].map(_parse_ppo_config)
    ppo["pipeline"] = parsed.map(lambda t: t[0])
    ppo["reward_type"] = parsed.map(lambda t: t[2])
    ppo = ppo[ppo["err_class"] != "empty"]
    for (model, pipeline), grp in ppo.groupby(["model", "pipeline"]):
        q = grp.loc[grp["reward_type"] == "quality_only", "n_tokens"]
        c = grp.loc[grp["reward_type"] == "composite", "n_tokens"]
        if len(q) < 5 or len(c) < 5:
            continue
        stat, p = mannwhitneyu(q, c, alternative="two-sided")
        lines.append(f"  {model:<10} {pipeline:<8} "
                     f"median q-only={q.median():>6.0f}  "
                     f"median composite={c.median():>6.0f}  "
                     f"U={stat:.0f}  p={p:.4g}")
    return lines or ["  [both quality-only and composite configs are needed]"]


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load_generations()
    check_prompt_alignment(df)

    print(f"\nClassifying {len(df)} generations "
          f"({'CLEANED' if CLEANED else 'RAW'} taxonomy)...")
    cls = df.apply(lambda r: classify_row(r["generation"], r["model"]),
                   axis=1, result_type="expand")
    df = pd.concat([df, cls], axis=1)

    bd = error_breakdown(df)
    bd.to_csv(OUT_DIR / "error_breakdown.csv", index=False)
    rh = reward_hacking_analysis(df)
    if len(rh):
        rh.to_csv(OUT_DIR / "length_by_reward.csv", index=False)

    lines = ["=" * 72,
             "ERROR BREAKDOWN (percentages of the total):",
             bd.to_string(index=False), "",
             "=" * 72,
             "LENGTHS/EMPTY OUTPUTS BY REWARD TYPE:",
             (rh.to_string(index=False) if len(rh)
              else "  [no PPO configs loaded]"), "",
             "Mann-Whitney (lengths, quality-only vs composite):"]
    lines += length_test_quality_vs_composite(df)
    summary = "\n".join(str(x) for x in lines)
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nWrote: {OUT_DIR/'inventory.csv'}, "
          f"{OUT_DIR/'error_breakdown.csv'}, "
          f"{OUT_DIR/'length_by_reward.csv'}, {OUT_DIR/'summary.txt'}")


if __name__ == "__main__":
    main()