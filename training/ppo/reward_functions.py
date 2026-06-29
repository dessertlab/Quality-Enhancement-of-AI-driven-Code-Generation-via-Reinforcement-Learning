import os
import re
import io
import sys
import math
import json
import shutil
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import torch
import pylcs
from bert_score import BERTScorer
from pylint.lint import Run
from pylint.reporters.text import TextReporter
from crystal_bleu import *

# ===Edit Distance ===
def edit_dist(hyp: str, ref: str) -> float:
    tmp = pylcs.edit_distance(hyp, ref)
    return 1 - (tmp / max(len(hyp), len(ref))) if max(len(hyp), len(ref)) > 0 else 1.0


# === BERTScore ===
bert_scorer = None

def init_bert_scorer(device):
    global bert_scorer
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    bert_scorer = BERTScorer(
        model_type="roberta-large",
        device="cpu",
        lang="en",
        rescale_with_baseline=False,
    )
    print("[INFO] BERTScorer initialized successfully")

def compute_bert_score(hyp_list, ref_list) -> list:
    P, R, F1 = bert_scorer.score(hyp_list, ref_list)
    print(f"[CODE] bertscore inputs: {len(hyp_list)} items", flush=True)
    return torch.clamp(F1, min=0.0).tolist()


# === CodeBLEU ===
def compute_codebleu_score(hyp_list, ref_list, codebleu_metric, lang="python") -> list:
    scores = []
    for hyp, ref in zip(hyp_list, ref_list):
        result = codebleu_metric.compute(predictions=[hyp], references=[ref], lang=[lang])
        scores.append(result["codebleu"])
    return scores


# === CrystalBLEU ===
def calc_crystalBLEU(hyps, refs, re_compute_ngrams: bool, cache_folder: str) -> list:
    if re_compute_ngrams:
        os.makedirs(cache_folder, exist_ok=True)
        for file in os.listdir(cache_folder):
            os.remove(os.path.join(cache_folder, file))
        print("[CRYSTAL] ngrams files deleted. Will compute trivially shared ngrams")
    else:
        print("[CRYSTAL] Loading trivially shared ngrams")
    print(f"[CRYSTAL] hyps count: {len(hyps)}", flush=True)
    trivial_ngrams = compute_trivially_shared_ngrams(hyps, "python", cache_folder)
    scores = compute_crystal_bleu(refs, hyps, trivial_ngrams, "python")
    return scores


# === Pylint ===
def _pylint_single(file_content_and_args):
    file_content, pylint_args = file_content_and_args
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(file_content)
            tmp = tf.name
        output = io.StringIO()
        reporter = TextReporter(output)
        args_local = [tmp, "--score=y"] + (pylint_args or [])
        run_obj = Run(args_local, reporter=reporter, exit=False)
        raw_score = None
        try:
            stats = getattr(run_obj, "linter", None)
            if stats and hasattr(stats, "stats"):
                raw_score = stats.stats.get("global_note")
        except Exception:
            raw_score = None
        out = output.getvalue()
        if raw_score is None:
            m = re.search(r"rated at\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10", out, re.IGNORECASE)
            raw_score = float(m.group(1)) if m else 0.0
        return max(0.0, min(1.0, raw_score / 10.0))
    except Exception as e:
        print(f"[PYLINT ERROR] file={tmp} err={e}", flush=True)
        return 0.0
    finally:
        try:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def pylint_quality_reward(hyps, references, pylint_workers, contexts=None, pylint_args=None, max_workers=None):
    if pylint_args is None:
        pylint_args = []
    max_workers = max_workers or pylint_workers or max(1, (os.cpu_count() or 1) // 2)
    files_contents = []
    for i, original_code in enumerate(hyps):
        code = original_code or ""
        docstring = None
        if contexts is not None and i < len(contexts):
            ctx = contexts[i] or ""
            m = re.search(r"Docstring:\s*(.*?)\s*Code:\s*$", ctx, re.DOTALL)
            if m:
                docstring = m.group(1).strip()
            if code.startswith(ctx):
                code = code[len(ctx):].lstrip()
            else:
                code = re.sub(r'(?s)^\s*Docstring:\s*.*?Code:\s*', '', code).lstrip()
        else:
            code = re.sub(r'(?s)^\s*Docstring:\s*.*?Code:\s*', '', code).lstrip()
        if not code or code.strip() == "":
            files_contents.append(("", pylint_args))
            continue
        if docstring:
            safe_doc = docstring.replace('"""', '\\"\\"\\"')
            file_content = f'"""{safe_doc}"""\n\n{code}'
        else:
            file_content = code
        files_contents.append((file_content, pylint_args))
    scores = []
    if max_workers <= 1 or len(files_contents) <= 1:
        for fc in files_contents:
            scores.append(_pylint_single(fc))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for res in ex.map(_pylint_single, files_contents):
                scores.append(res)
    return scores


# === Semgrep ===
def _run_semgrep_on_chunk(chunk_codes, configs, semgrep_timeout, subprocess_timeout, verbose=False):
    tmp_dir = tempfile.mkdtemp(prefix="semgrep_chunk_")
    tmp_json = os.path.join(tempfile.gettempdir(), f"semgrep_out_{os.getpid()}_{os.getppid()}.json")
    scores = [0.0] * len(chunk_codes)
    try:
        filenames = []
        for i, code in enumerate(chunk_codes):
            fname = os.path.join(tmp_dir, f"snippet_{i}.py")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(code or "")
            filenames.append(fname)
        cmd = ["semgrep", "scan", "--json", "--output", tmp_json, f"--timeout={semgrep_timeout}"]
        for c in configs:
            cmd += ["--config", c]
        cmd.append(tmp_dir)
        subprocess.run(cmd, capture_output=True, text=True, timeout=subprocess_timeout)
        scores = [1.0] * len(chunk_codes)
        if os.path.exists(tmp_json):
            with open(tmp_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            results_list = data if isinstance(data, list) else data.get("results", [])
            findings_by_file = {}
            for finding in results_list:
                path = finding.get("path")
                if path:
                    findings_by_file.setdefault(path, []).append(finding)
            for i, fname in enumerate(filenames):
                base = os.path.basename(fname)
                file_findings = findings_by_file.get(fname) or findings_by_file.get(base) or []
                found_cwes = []
                for fnd in file_findings:
                    meta = (fnd.get("extra") or {}).get("metadata") or {}
                    for key in ("cwe", "cwe_id", "cwe_ids", "cwe_ids_raw"):
                        val = meta.get(key)
                        if not val:
                            continue
                        vals = val if isinstance(val, list) else [val]
                        for v in vals:
                            m = re.search(r"CWE-?\d+", str(v))
                            if m:
                                found_cwes.append(m.group(0).upper())
                    text = " ".join(filter(None, [
                        (fnd.get("extra") or {}).get("message"),
                        meta.get("short_description"),
                        meta.get("description"),
                    ]))
                    for m in re.finditer(r"CWE-?\d+", text, flags=re.IGNORECASE):
                        found_cwes.append(m.group(0).upper())
                unique_cwes = list(set(found_cwes))
                score = 1.0 - 0.5 * len(unique_cwes)
                scores[i] = score
                if verbose:
                    print(f"[SEMGREP] {base}: {unique_cwes} -> {score}", flush=True)
    except subprocess.TimeoutExpired:
        print("[SEMGREP TIMEOUT]", file=sys.stderr)
    except FileNotFoundError:
        print("[ERROR] 'semgrep' not found in PATH", file=sys.stderr)
    except Exception as e:
        print(f"[SEMGREP ERROR] {e}", file=sys.stderr)
    finally:
        try:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
        except Exception:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return scores


def semgrep_security_reward(
    hyps,
    semgrep_workers,
    configs=None,
    chunk_size=32,
    max_workers=None,
    semgrep_timeout=180,
    subprocess_timeout=300,
    verbose=False,
):
    if configs is None:
        configs = ["p/default", "p/comment", "p/bandit", "p/python"]
    n = len(hyps)
    try:
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except Exception:
        world_size = 1
    total_cpus = max(1, os.cpu_count() or 1)
    per_process_cpus = max(1, total_cpus // world_size)
    max_workers = max_workers or min(per_process_cpus, math.ceil(n / chunk_size), semgrep_workers or per_process_cpus)

    chunks = [hyps[i:i + chunk_size] for i in range(0, n, chunk_size)]
    results = [0.0] * n

    if max_workers <= 1:
        for idx, chunk in enumerate(chunks):
            try:
                chunk_scores = _run_semgrep_on_chunk(chunk, configs, semgrep_timeout, subprocess_timeout, verbose)
            except Exception as e:
                print(f"[SEMGREP CHUNK ERROR] idx={idx} err={e}", flush=True)
                chunk_scores = [0.0] * len(chunk)
            start = idx * chunk_size
            results[start:start + len(chunk_scores)] = chunk_scores
        print(f"[SEMGREP] processed {n} items (sequential)", flush=True)
        return results

    ctx = get_context("spawn")
    max_workers = min(max_workers, len(chunks))
    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futures = {ex.submit(_run_semgrep_on_chunk, chunk, configs, semgrep_timeout, subprocess_timeout, verbose): idx for idx, chunk in enumerate(chunks)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    chunk_scores = fut.result()
                except Exception as e:
                    print(f"[SEMGREP FUTURE ERROR] idx={idx} err={e}", flush=True)
                    chunk_scores = [0.0] * len(chunks[idx])
                start = idx * chunk_size
                results[start:start + len(chunk_scores)] = chunk_scores
    except Exception as e:
        print(f"[SEMGREP EXECUTOR ERROR] {e} - falling back to sequential", flush=True)
        for idx, chunk in enumerate(chunks):
            try:
                chunk_scores = _run_semgrep_on_chunk(chunk, configs, semgrep_timeout, subprocess_timeout, verbose)
            except Exception as e2:
                print(f"[SEMGREP CHUNK ERROR - fallback] idx={idx} err={e2}", flush=True)
                chunk_scores = [0.0] * len(chunk)
            start = idx * chunk_size
            results[start:start + len(chunk_scores)] = chunk_scores
    print(f"[SEMGREP] processed {n} items (parallel max_workers={max_workers})", flush=True)
    return results


# === Utility ===
def calculate_token_length_similarity(hyps, refs, tokenizer) -> list:
    if len(hyps) != len(refs):
        raise ValueError(f"Mismatch: {len(hyps)} hyps vs {len(refs)} refs")
    scores = []
    for gen, gt in zip(hyps, refs):
        len_gen = len(tokenizer.encode(gen))
        len_gt  = len(tokenizer.encode(gt))
        if len_gen == 0 and len_gt == 0:
            score = 1.0
        elif len_gen == 0 or len_gt == 0:
            score = 0.0
        else:
            score = min(len_gen, len_gt) / max(len_gen, len_gt)
        scores.append(score)
    return scores


def clean_special_token_strings(s: str) -> str:
    s = re.sub(r'(?:<\|endoftext\|>|<\|im_end\|>)+', '', s)
    s = re.sub(r'(?:\s*(?:<\|endoftext\|>|<\|im_end\|>|\s))+$', '', s)
    return s.strip()