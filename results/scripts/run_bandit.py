#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bandit re-analysis of the Semgrep-rewarded configurations.

Re-analyzes the outputs of the Semgrep-based reward configurations
(R_s, R_cs) and their baselines with Bandit, an independent
security-oriented static analyzer never used during training,
following the same classification protocol as the paper
(empty/non-parseable = error; 0 issues = clean; >=1 = defective).
Outputs (bandit_reanalysis.csv, bandit_trend_check.csv) are written
to results/.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# =====================================================================
# CONFIGURATION
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent     # results/scripts
RESULTS_DIR = SCRIPT_DIR.parent                   # results
OUT_DIR = RESULTS_DIR

CONFIGS_TO_ANALYZE = {
    "pretrained", "sft",
    "ppo_Rs", "ppo_Rcs",
    "sft_ppo_Rs", "sft_ppo_Rcs",
}


TREND_CHECKS = [
    ("ppo_Rs",      "pretrained"),
    ("ppo_Rs",      "sft"),
    ("ppo_Rcs",     "pretrained"),
    ("ppo_Rcs",     "sft"),
    ("sft_ppo_Rs",  "sft"),
    ("sft_ppo_Rcs", "sft"),
]

BANDIT_CMD = ["bandit", "-r", "-f", "json", "-q"]


import characterize_outputs as _co


def load_generations() -> pd.DataFrame:
    # uses the repository-relative default (results/inference_cleaned)
    return _co.load_generations()


# =====================================================================
# Bandit analysis
# =====================================================================

def is_parseable(text: str) -> bool:
    if text is None or not text.strip():
        return False
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False


def run_bandit_on_group(snippets: dict[str, str]) -> dict[str, int]:
    if not snippets:
        return {}
    with tempfile.TemporaryDirectory(prefix="bandit_") as tmp:
        tmpdir = Path(tmp)
        for pid, src in snippets.items():
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(pid))
            (tmpdir / f"{safe}.py").write_text(src, encoding="utf-8")

        proc = subprocess.run(
            BANDIT_CMD + [str(tmpdir)],
            capture_output=True, text=True,
        )
        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError:
            sys.exit(
                "Unparseable Bandit output. Check the Bandit installation "
                f"(pip install bandit). stderr:\n{proc.stderr[:2000]}"
            )

        counts: dict[str, int] = {}
        for issue in report.get("results", []):
            fname = Path(issue["filename"]).stem
            counts[fname] = counts.get(fname, 0) + 1

        remap = {}
        for pid in snippets:
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(pid))
            remap[str(pid)] = counts.get(safe, 0)
        return remap


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    try:
        subprocess.run(["bandit", "--version"], capture_output=True,
                       check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit("Bandit not found: pip install bandit")

    df = load_generations()
    df = df[df["config"].isin(CONFIGS_TO_ANALYZE)].copy()
    if df.empty:
        sys.exit("No generations for the requested configs: "
                 f"{sorted(CONFIGS_TO_ANALYZE)}")

    rows = []
    for (model, config, test_set), grp in df.groupby(
            ["model", "config", "test_set"]):
        parseable_mask = grp["generation"].map(is_parseable)
        n_total = len(grp)
        n_error = int((~parseable_mask).sum())

        snippets = dict(zip(grp.loc[parseable_mask, "prompt_id"].astype(str),
                            grp.loc[parseable_mask, "generation"]))
        print(f"Bandit analysis for {model}/{config}/{test_set} "
              f"({len(snippets)} parseable snippets)...")
        issue_counts = run_bandit_on_group(snippets)

        n_clean = sum(1 for v in issue_counts.values() if v == 0)
        n_defective = sum(1 for v in issue_counts.values() if v > 0)

        rows.append(dict(
            model=model, config=config, test_set=test_set,
            n=n_total,
            bandit_clean_pct=round(100 * n_clean / n_total, 1),
            bandit_defective_pct=round(100 * n_defective / n_total, 1),
            error_pct=round(100 * n_error / n_total, 1),
        ))

    res = pd.DataFrame(rows).sort_values(["model", "test_set", "config"])
    res.to_csv(OUT_DIR / "bandit_reanalysis.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'bandit_reanalysis.csv'}\n")
    print(res.to_string(index=False))

    print("\n" + "=" * 72)
    print("Checking trend under Bandit "
          "(delta percentage points of clean-rate):")
    idx = res.set_index(["model", "test_set", "config"])["bandit_clean_pct"]
    trend_rows = []
    for (model, test_set) in res[["model", "test_set"]]\
            .drop_duplicates().itertuples(index=False):
        for config, baseline in TREND_CHECKS:
            try:
                delta = (idx[(model, test_set, config)]
                         - idx[(model, test_set, baseline)])
            except KeyError:
                continue
            trend_rows.append(dict(model=model, test_set=test_set,
                                   config=config, baseline=baseline,
                                   delta_clean_pp=round(delta, 1)))
            print(f"  {model:<10} {test_set:<4} {config:<13} vs "
                  f"{baseline:<11} : {delta:+.1f} pp")
    pd.DataFrame(trend_rows).to_csv(OUT_DIR / "bandit_trend_check.csv",
                                    index=False)
    print(f"\nSaved {OUT_DIR / 'bandit_trend_check.csv'}\n")

if __name__ == "__main__":
    main()