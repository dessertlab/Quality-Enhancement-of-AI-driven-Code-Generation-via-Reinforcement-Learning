#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired statistical analysis (Sect. 5 of the paper).

Quality: each prompt is binarized as clean vs. non-clean and each
selected configuration is compared against its baselines with
McNemar's test (exact variant when discordant pairs are fewer than
25), with Wilson 95% confidence intervals. Correctness proxy:
per-sample Edit Distance values are compared with the Wilcoxon
signed-rank test, with Cliff's delta as effect size. P-values are
Holm-adjusted (alpha = 0.05) within each metric family over the
pre-specified comparison family (COMPARISONS below); inference is
anchored to the PPO test set (n = 1,200).

Inputs: per-sample quality labels (results/scripts/quality_labels.csv)
and the post-processed generations (results/inference_cleaned, loaded
via characterize_outputs.py). Empty outputs are reclassified as errors
prior to the analysis, following the same procedure as the paper.
Outputs (stats_quality.csv, stats_ed.csv) are written to results/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

# =====================================================================
# CONFIGURATION
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent     # results/scripts
RESULTS_DIR = SCRIPT_DIR.parent                   # results
QUALITY_LABELS_CSV = SCRIPT_DIR / "quality_labels.csv"
OUT_DIR = RESULTS_DIR
ALPHA = 0.05

COMPARISONS = [
    # CodeGPT: best = DPO standalone and SFT&PPO R_cs
    ("CodeGPT",   "ppo", "dpo",         "sft"),
    ("CodeGPT",   "ppo", "dpo",         "pretrained"),
    ("CodeGPT",   "ppo", "sft_ppo_Rcs", "sft"),
    # CodeGen: best = SFT&DPO and PPO R_p standalone
    ("CodeGen",   "ppo", "sft_dpo",     "sft"),
    ("CodeGen",   "ppo", "ppo_Rp",      "sft"),
    ("CodeGen",   "ppo", "ppo_Rp",      "pretrained"),
    # QwenCoder: best = PPO R_cp and R_p standalone
    ("QwenCoder", "ppo", "ppo_Rcp",     "sft"),
    ("QwenCoder", "ppo", "ppo_Rcp",     "pretrained"),
    ("QwenCoder", "ppo", "ppo_Rp",      "sft"),
    ("QwenCoder", "ppo", "ppo_Rp",      "pretrained"),
    # DeepSeek: best = DPO standalone, PPO R_p and R_s standalone
    ("DeepSeek",  "ppo", "dpo",         "sft"),
    ("DeepSeek",  "ppo", "dpo",         "pretrained"),
    ("DeepSeek",  "ppo", "ppo_Rp",      "sft"),
    ("DeepSeek",  "ppo", "ppo_Rp",      "pretrained"),
    ("DeepSeek",  "ppo", "ppo_Rs",      "sft"),
    ("DeepSeek",  "ppo", "ppo_Rs",      "pretrained"),
]

TRAIN_ALIASES = {
    "pretrained": "pretrained", "sft": "sft", "finetuned": "sft",
    "dpo": "dpo", "ppo": "ppo",          # standalone RL
    "sft_dpo": "sft_dpo", "sft_ppo": "sft_ppo",
}
REWARDS = {"pylint": "Rp", "semgrep": "Rs",
           "custom_pylint": "Rcp", "custom_semgrep": "Rcs"}
CORRECTNESS_REWARDS = {"bertscore", "codebleu", "crystalbleu", "edit"}

PAPER_ERROR = {
 ('CodeGPT','pretrained','dpo'):99.0, ('CodeGPT','pretrained','ppo'):91.2,
 ('CodeGPT','sft','dpo'):100.0, ('CodeGPT','sft','ppo'):94.3,
 ('CodeGPT','dpo','dpo'):94.0, ('CodeGPT','dpo','ppo'):91.8,
 ('CodeGPT','sft_dpo','dpo'):99.0, ('CodeGPT','sft_dpo','ppo'):94.8,
 ('CodeGPT','ppo_Rcp','dpo'):99.0, ('CodeGPT','ppo_Rcp','ppo'):96.0,
 ('CodeGPT','ppo_Rcs','dpo'):97.0, ('CodeGPT','ppo_Rcs','ppo'):97.2,
 ('CodeGPT','ppo_Rp','dpo'):97.0, ('CodeGPT','ppo_Rp','ppo'):97.0,
 ('CodeGPT','ppo_Rs','dpo'):99.0, ('CodeGPT','ppo_Rs','ppo'):96.2,
 ('CodeGPT','sft_ppo_Rcp','dpo'):100.0, ('CodeGPT','sft_ppo_Rcp','ppo'):98.8,
 ('CodeGPT','sft_ppo_Rcs','dpo'):94.0, ('CodeGPT','sft_ppo_Rcs','ppo'):91.2,
 ('CodeGPT','sft_ppo_Rp','dpo'):98.0, ('CodeGPT','sft_ppo_Rp','ppo'):94.8,
 ('CodeGPT','sft_ppo_Rs','dpo'):96.0, ('CodeGPT','sft_ppo_Rs','ppo'):92.0,
 ('CodeGen','pretrained','dpo'):84.0, ('CodeGen','pretrained','ppo'):73.4,
 ('CodeGen','sft','dpo'):72.0, ('CodeGen','sft','ppo'):83.3,
 ('CodeGen','dpo','dpo'):93.0, ('CodeGen','dpo','ppo'):70.5,
 ('CodeGen','sft_dpo','dpo'):78.0, ('CodeGen','sft_dpo','ppo'):60.2,
 ('CodeGen','ppo_Rcp','dpo'):99.0, ('CodeGen','ppo_Rcp','ppo'):98.4,
 ('CodeGen','ppo_Rcs','dpo'):96.0, ('CodeGen','ppo_Rcs','ppo'):88.6,
 ('CodeGen','ppo_Rp','dpo'):93.0, ('CodeGen','ppo_Rp','ppo'):73.4,
 ('CodeGen','ppo_Rs','dpo'):84.0, ('CodeGen','ppo_Rs','ppo'):77.9,
 ('CodeGen','sft_ppo_Rcp','dpo'):71.0, ('CodeGen','sft_ppo_Rcp','ppo'):76.5,
 ('CodeGen','sft_ppo_Rcs','dpo'):76.0, ('CodeGen','sft_ppo_Rcs','ppo'):78.2,
 ('CodeGen','sft_ppo_Rp','dpo'):88.0, ('CodeGen','sft_ppo_Rp','ppo'):91.9,
 ('CodeGen','sft_ppo_Rs','dpo'):95.0, ('CodeGen','sft_ppo_Rs','ppo'):79.6,
 ('QwenCoder','pretrained','dpo'):97.0, ('QwenCoder','pretrained','ppo'):79.2,
 ('QwenCoder','sft','dpo'):96.0, ('QwenCoder','sft','ppo'):91.8,
 ('QwenCoder','dpo','dpo'):100.0, ('QwenCoder','dpo','ppo'):90.2,
 ('QwenCoder','sft_dpo','dpo'):98.0, ('QwenCoder','sft_dpo','ppo'):95.7,
 ('QwenCoder','ppo_Rcp','dpo'):83.0, ('QwenCoder','ppo_Rcp','ppo'):74.6,
 ('QwenCoder','ppo_Rcs','dpo'):96.0, ('QwenCoder','ppo_Rcs','ppo'):77.8,
 ('QwenCoder','ppo_Rp','dpo'):96.0, ('QwenCoder','ppo_Rp','ppo'):75.6,
 ('QwenCoder','ppo_Rs','dpo'):98.0, ('QwenCoder','ppo_Rs','ppo'):88.5,
 ('QwenCoder','sft_ppo_Rcp','dpo'):95.0, ('QwenCoder','sft_ppo_Rcp','ppo'):90.2,
 ('QwenCoder','sft_ppo_Rcs','dpo'):98.0, ('QwenCoder','sft_ppo_Rcs','ppo'):94.3,
 ('QwenCoder','sft_ppo_Rp','dpo'):97.0, ('QwenCoder','sft_ppo_Rp','ppo'):94.9,
 ('QwenCoder','sft_ppo_Rs','dpo'):95.0, ('QwenCoder','sft_ppo_Rs','ppo'):94.2,
 ('DeepSeek','pretrained','dpo'):34.0, ('DeepSeek','pretrained','ppo'):51.2,
 ('DeepSeek','sft','dpo'):61.0, ('DeepSeek','sft','ppo'):64.4,
 ('DeepSeek','dpo','dpo'):35.0, ('DeepSeek','dpo','ppo'):52.1,
 ('DeepSeek','sft_dpo','dpo'):65.0, ('DeepSeek','sft_dpo','ppo'):68.1,
 ('DeepSeek','ppo_Rcp','dpo'):89.0, ('DeepSeek','ppo_Rcp','ppo'):44.8,
 ('DeepSeek','ppo_Rcs','dpo'):68.0, ('DeepSeek','ppo_Rcs','ppo'):49.0,
 ('DeepSeek','ppo_Rp','dpo'):54.0, ('DeepSeek','ppo_Rp','ppo'):49.5,
 ('DeepSeek','ppo_Rs','dpo'):81.0, ('DeepSeek','ppo_Rs','ppo'):47.4,
 ('DeepSeek','sft_ppo_Rcp','dpo'):88.0, ('DeepSeek','sft_ppo_Rcp','ppo'):80.2,
 ('DeepSeek','sft_ppo_Rcs','dpo'):80.0, ('DeepSeek','sft_ppo_Rcs','ppo'):82.5,
 ('DeepSeek','sft_ppo_Rp','dpo'):89.0, ('DeepSeek','sft_ppo_Rp','ppo'):78.2,
 ('DeepSeek','sft_ppo_Rs','dpo'):88.0, ('DeepSeek','sft_ppo_Rs','ppo'):82.3,
}


# =====================================================================
# Label loading
# =====================================================================

def _parse_label_row(config_raw: str, ts_raw: str):
    train = TRAIN_ALIASES.get(config_raw)
    if train is None or ts_raw == "QIQO":
        return None
    split, reward = "", None
    if ts_raw.startswith("test_secure") or ts_raw.startswith("test_insecure"):
        parts = ts_raw.split("_")
        split = parts[1]                       # secure | insecure
        reward = "_".join(parts[2:]) or None   # '' -> None
        test_set = "ppo"
    else:
        test_set = "dpo"
        reward = None if ts_raw == "full" else ts_raw
    if reward in CORRECTNESS_REWARDS:
        return None                            
    if train in ("ppo", "sft_ppo"):
        if reward not in REWARDS:
            return None
        config = f"{train}_{REWARDS[reward]}"
    else:
        if reward is not None:
            return None
        config = train
    return config, test_set, split


def load_quality_labels() -> pd.DataFrame:
    df = pd.read_csv(QUALITY_LABELS_CSV)
    need = {"model", "config", "test_set", "prompt_id", "label"}
    if need - set(df.columns):
        sys.exit(f"Missing columns in {QUALITY_LABELS_CSV}: "
                 f"{need - set(df.columns)}")

    parsed = df.apply(lambda r: _parse_label_row(r["config"], r["test_set"]),
                      axis=1)
    keep = parsed.notna()
    out = df[keep].copy()
    trip = parsed[keep]
    out["config"] = trip.map(lambda t: t[0])
    out["ts"] = trip.map(lambda t: t[1])
    out["split"] = trip.map(lambda t: t[2])
    out["prompt_id"] = [
        f"{s}_{p}" if s else str(p)
        for s, p in zip(out["split"], out["prompt_id"])
    ]
    out = out[["model", "config", "ts", "prompt_id", "label"]] \
        .rename(columns={"ts": "test_set"})

    print(f"Label in scope: {len(out)} "
          f"(discarded {len(df) - len(out)}: QIQO / reward correctness-only)")
    sizes = out.groupby(["model", "config", "test_set"]).size()
    n_combo = len(sizes)
    print(f"Combination (model, config, test_set): {n_combo} (expected 96)")
    bad = sizes[~sizes.isin([100, 1200])]
    if len(bad):
        print("  WARNING anomalous sizes:\n", bad.to_string())
    dup = out.duplicated(subset=["model", "config", "test_set", "prompt_id"])
    if dup.any():
        sys.exit(f"prompt_id duplicated ({dup.sum()}): impossible pairing.")

    err = (out.assign(is_err=out["label"].eq("error"))
              .groupby(["model", "config", "test_set"])["is_err"]
              .mean().mul(100).round(1))
    rows = []
    for k, v in err.items():
        p = PAPER_ERROR.get(k)
        if p is not None:
            rows.append((k, v, p, round(v - p, 1)))
    dev = pd.DataFrame(rows, columns=["combo", "labels_err", "paper_err",
                                      "delta"])
    print(f"Error-rate validation vs figures: mean |delta| = "
          f"{dev['delta'].abs().mean():.2f} pp, max = "
          f"{dev['delta'].abs().max():.1f} pp")
    big = dev[dev["delta"].abs() > 1.0]
    if len(big):
        print("  Combinations with |delta| > 1 pp (verify!):")
        print(big.to_string(index=False))
    else:
        print("  OK: the labels reproduce the paper figures (within 1 pp).")
    return out


def apply_empty_override(labels: pd.DataFrame,
                         gens: pd.DataFrame) -> pd.DataFrame:
    empty_keys = set(map(tuple, gens.loc[
        gens["generation"].fillna("").str.strip().eq(""),
        ["model", "config", "test_set", "prompt_id"]
    ].values))
    key = list(zip(labels["model"], labels["config"],
                   labels["test_set"], labels["prompt_id"]))
    mask = pd.Series([k in empty_keys for k in key], index=labels.index)
    n_over = int((mask & ~labels["label"].eq("error")).sum())
    labels = labels.copy()
    labels.loc[mask, "label"] = "error"
    print(f"Empty->error override applied to {n_over} samples.")

    err = (labels.assign(is_err=labels["label"].eq("error"))
                 .groupby(["model", "config", "test_set"])["is_err"]
                 .mean().mul(100).round(1))
    devs = [(k, v, PAPER_ERROR[k], round(v - PAPER_ERROR[k], 1))
            for k, v in err.items() if k in PAPER_ERROR]
    dd = pd.DataFrame(devs, columns=["combo", "labels_err", "paper_err",
                                     "delta"])
    print(f"Post-override GATE: mean |delta| = "
          f"{dd['delta'].abs().mean():.2f} pp, max = "
          f"{dd['delta'].abs().max():.1f} pp")
    big = dd[dd["delta"].abs() > 1.0]
    if len(big):
        print("  WARNING combos beyond 1 pp -- do NOT proceed without understanding why:")
        print(big.to_string(index=False))
    else:
        print("  OK: the labels reproduce the paper figures (within 1 pp).")
    return labels


# =====================================================================
# Per-sample ED (paper implementation: pylcs + normalization)
# =====================================================================

try:
    import pylcs

    def _edit_distance_raw(a: str, b: str) -> int:
        return pylcs.edit_distance(a, b)
except ImportError:
    print("[WARN] pylcs not installed: using pure-Python Levenshtein "
          "(slower, same result). pip install pylcs")

    def _edit_distance_raw(a: str, b: str) -> int:
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            cur = [i] + [0] * lb
            ca = a[i - 1]
            for j in range(1, lb + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                             prev[j - 1] + (ca != b[j - 1]))
            prev = cur
        return prev[lb]


def edit_dist(hyp: str, ref: str) -> float:
    denom = max(len(hyp), len(ref))
    if denom == 0:
        return 1.0
    return 1.0 - _edit_distance_raw(hyp, ref) / denom


def load_ed_scores(gens: pd.DataFrame | None = None) -> pd.DataFrame:
    if gens is None:
        import characterize_outputs as _co
        gens = _co.load_generations()
    df = gens
    print("Computing per-sample ED (pylcs)...")
    df["ed"] = [100.0 * edit_dist(g or "", r or "")
                for g, r in zip(df["generation"], df["reference"])]
    means = (df.groupby(["model", "config", "test_set"])["ed"]
               .mean().round(2))
    print("\nED VALIDATION -- means per config (compare with Table 3):")
    print(means.to_string())
    return df[["model", "config", "test_set", "prompt_id", "ed"]]


# =====================================================================
# Statistical tests (McNemar, Wilcoxon, Cliff's delta)
# =====================================================================

def cliffs_delta(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gt = int((x[:, None] > y[None, :]).sum())
    lt = int((x[:, None] < y[None, :]).sum())
    d = (gt - lt) / (len(x) * len(y))
    ad = abs(d)
    mag = ("negligible" if ad < 0.147 else "small" if ad < 0.33
           else "medium" if ad < 0.474 else "large")
    return float(d), mag


def _paired(df, model, test_set, config, baseline, col):
    a = df[(df["model"] == model) & (df["test_set"] == test_set)
           & (df["config"] == config)][["prompt_id", col]]
    b = df[(df["model"] == model) & (df["test_set"] == test_set)
           & (df["config"] == baseline)][["prompt_id", col]]
    m = a.merge(b, on="prompt_id", suffixes=("_cfg", "_base"))
    exp = min(len(a), len(b))
    if len(m) == 0:
        raise ValueError(f"no pairing for {model}/{test_set} "
                         f"{config} vs {baseline}")
    if len(m) != exp or len(a) != len(b):
        print(f"  WARNING pairing {model}/{test_set} {config} vs "
              f"{baseline}: {len(m)} pairs from {len(a)}/{len(b)} samples")
    return m[f"{col}_cfg"], m[f"{col}_base"]


def quality_tests(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, test_set, config, baseline in COMPARISONS:
        try:
            cfg, base = _paired(labels, model, test_set, config, baseline,
                                "label")
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue
        c = cfg.eq("clean").to_numpy()
        b_ = base.eq("clean").to_numpy()
        n = len(c)
        n11 = int((b_ & c).sum());   n10 = int((b_ & ~c).sum())
        n01 = int((~b_ & c).sum());  n00 = int((~b_ & ~c).sum())
        exact = (n10 + n01) < 25
        res = mcnemar([[n11, n10], [n01, n00]], exact=exact, correction=True)
        k_c, k_b = int(c.sum()), int(b_.sum())
        lo_c, hi_c = proportion_confint(k_c, n, ALPHA, method="wilson")
        lo_b, hi_b = proportion_confint(k_b, n, ALPHA, method="wilson")
        rows.append(dict(
            metric="quality_clean", model=model, test_set=test_set,
            config=config, baseline=baseline, n=n,
            clean_cfg_pct=round(100 * k_c / n, 1),
            clean_base_pct=round(100 * k_b / n, 1),
            delta_pp=round(100 * (k_c - k_b) / n, 1),
            wilson_cfg=f"[{100*lo_c:.1f}, {100*hi_c:.1f}]",
            wilson_base=f"[{100*lo_b:.1f}, {100*hi_b:.1f}]",
            discordant_b=n10, discordant_c=n01,
            test="McNemar exact" if exact else "McNemar chi2",
            p_raw=float(res.pvalue),
        ))
    out = pd.DataFrame(rows)
    if len(out):
        rej, p_adj, _, _ = multipletests(out["p_raw"], ALPHA, method="holm")
        out["p_holm"] = [f"{p:.2e}" for p in p_adj]
        out["significant"] = rej
        out["p_raw"] = [f"{p:.2e}" for p in out["p_raw"]]
    return out


def ed_tests(ed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, test_set, config, baseline in COMPARISONS:
        try:
            cfg, base = _paired(ed, model, test_set, config, baseline, "ed")
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue
        x = cfg.astype(float).to_numpy()
        y = base.astype(float).to_numpy()
        if np.allclose(x - y, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(x, y, zero_method="pratt", method="approx")
        d, mag = cliffs_delta(x, y)
        rows.append(dict(
            metric="ed", model=model, test_set=test_set,
            config=config, baseline=baseline, n=len(x),
            ed_cfg_mean=round(float(x.mean()), 2),
            ed_base_mean=round(float(y.mean()), 2),
            delta=round(float(x.mean() - y.mean()), 2),
            statistic=None if np.isnan(stat) else round(float(stat), 1),
            p_raw=float(p), cliffs_delta=round(d, 3), magnitude=mag,
        ))
    out = pd.DataFrame(rows)
    if len(out):
        rej, p_adj, _, _ = multipletests(out["p_raw"], ALPHA, method="holm")
        out["p_holm"] = [f"{p:.2e}" for p in p_adj]
        out["significant"] = rej
        out["p_raw"] = [f"{p:.2e}" for p in out["p_raw"]]
    return out


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 72)
    print("0) GENERATIONS (needed for the empty override and for ED)")
    import characterize_outputs as _co
    gens = _co.load_generations()

    print("\n1) QUALITY LABELS")
    labels = load_quality_labels()
    labels = apply_empty_override(labels, gens)

    print("\n2) QUALITY TESTS (McNemar + Wilson, Holm)")
    q = quality_tests(labels)
    q.to_csv(OUT_DIR / "stats_quality.csv", index=False)
    cols = ["model", "config", "baseline", "clean_cfg_pct", "clean_base_pct",
            "delta_pp", "p_holm", "significant"]
    print(q[cols].to_string(index=False))

    print("\n3) ED PER-SAMPLE")
    ed = load_ed_scores(gens)

    print("\n4) ED TESTS (Wilcoxon + Cliff's delta, Holm)")
    e = ed_tests(ed)
    e.to_csv(OUT_DIR / "stats_ed.csv", index=False)
    cols = ["model", "config", "baseline", "ed_cfg_mean", "ed_base_mean",
            "delta", "p_holm", "cliffs_delta", "magnitude", "significant"]
    print(e[cols].to_string(index=False))

    print(f"\nWrote {OUT_DIR/'stats_quality.csv'} and {OUT_DIR/'stats_ed.csv'}")


if __name__ == "__main__":
    main()