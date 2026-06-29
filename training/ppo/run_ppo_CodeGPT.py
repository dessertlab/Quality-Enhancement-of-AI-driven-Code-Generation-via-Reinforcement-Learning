# accelerate launch --num_processes 2 training/ppo/run_ppo_CodeGPT.py
import os
os.environ["HF_HUB_ENABLE_HF_XET"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_CACHE", "/tmp/hfcache")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/tmp/hfhome")
os.environ["TMPDIR"] = os.getenv("TMPDIR", "/tmp")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

import multiprocessing as mp
import torch

cpu_count = os.cpu_count()
world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
threads_per_proc = max(1, cpu_count // world_size)
os.environ["OMP_NUM_THREADS"] = str(threads_per_proc)
os.environ["MKL_NUM_THREADS"] = str(threads_per_proc)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_proc)
torch.set_num_threads(threads_per_proc)
torch.set_num_interop_threads(threads_per_proc)
print(f"[INFO] CPU threads per process: {threads_per_proc}", flush=True)

import torch.distributed as dist
import argparse
import evaluate
from transformers import AutoTokenizer, GenerationConfig
from trl.trainer import PPOTrainer, PPOConfig
from trl import create_reference_model
from trl.models import AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

try:
    from crystal_bleu import *
except ModuleNotFoundError:
    import importlib.util, pathlib
    for candidate_dir in [REPO_ROOT, SCRIPT_DIR]:
        candidate = pathlib.Path(candidate_dir) / "crystal_bleu.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("crystal_bleu", str(candidate))
            cb = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cb)
            globals().update({k: getattr(cb, k) for k in dir(cb) if not k.startswith("_")})
            break

from reward_functions import (
    edit_dist, init_bert_scorer, compute_bert_score,
    compute_codebleu_score, calc_crystalBLEU,
    pylint_quality_reward,
    semgrep_security_reward,
    calculate_token_length_similarity,
)

CRYSTAL_CACHE = os.path.join(REPO_ROOT, "training/ppo/crystal_cache")

# === PARAMETERS ===
parser = argparse.ArgumentParser(description="PPO Training (multi-GPU via accelerate)")
parser.add_argument("--model_name", type=str, default=os.environ.get("MODEL_NAME",  os.path.join(REPO_ROOT, "models/sft/CodeGPT-finetuned")))
parser.add_argument("--metric", type=str, choices=["bertscore", "codebleu", "edit", "crystalbleu", "pylint", "semgrep", "custom_pylint", "custom_semgrep"], default=os.environ.get("METRIC", "pylint"))
parser.add_argument("--finetuned", type=str, choices=["yes", "no"], default=os.environ.get("FINETUNED", "yes"))
parser.add_argument("--train_epochs", type=int, default=int(os.environ.get("TRAIN_EPOCHS", "10")))
parser.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "32")))
parser.add_argument("--grad_acc", type=int, default=int(os.environ.get("GRAD_ACC", "8")))
parser.add_argument("--learning_rate", type=float, default=float(os.environ.get("LEARNING_RATE", "1e-6")))
parser.add_argument("--max_source_length", type=int, default=int(os.environ.get("MAX_SOURCE_LENGTH", "256")))
parser.add_argument("--max_target_length", type=int, default=int(os.environ.get("MAX_TARGET_LENGTH", "512")))
parser.add_argument("--pylint_workers", type=int, default=1)
parser.add_argument("--semgrep_workers", type=int, default=1)
args = parser.parse_args()

# === OUTPUT PATH ===
if args.finetuned == "yes":
    OUTPUT_DIR = os.path.join(REPO_ROOT, "models", "sft_ppo", f"CodeGPT-sft_ppo-{args.metric}")
else:
    OUTPUT_DIR = os.path.join(REPO_ROOT, "models", "ppo", f"CodeGPT-ppo-{args.metric}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print(f"[INFO] tokenizer special_tokens_map: {tokenizer.special_tokens_map}", flush=True)

# === LOAD & PREPARE DATASET ===
data_files = {
    "train": [
        os.path.join(REPO_ROOT, "datasets", "ppo", "train_datasets", "ppo_train_secure.jsonl"),
        os.path.join(REPO_ROOT, "datasets", "ppo", "train_datasets", "ppo_train_insecure.jsonl"),
    ],
    "validation": [
        os.path.join(REPO_ROOT, "datasets", "ppo", "validation_datasets", "ppo_validation_secure.jsonl"),
        os.path.join(REPO_ROOT, "datasets", "ppo", "validation_datasets", "ppo_validation_insecure.jsonl"),
    ],
}
dataset = load_dataset("json", data_files=data_files)

def extract_prompts(example):
    msgs = example["messages"]
    doc       = next((m["content"] for m in msgs if m["role"] == "user"),      "")
    reference = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    return {"prompt": doc, "reference": reference}

dataset = dataset.map(extract_prompts, load_from_cache_file=False)
for split in ["train", "validation"]:
    if "messages" in dataset[split].column_names:
        dataset[split] = dataset[split].remove_columns(["messages"])

def preprocess_function(examples):
    prompts    = examples["prompt"]    if isinstance(examples["prompt"],    list) else [examples["prompt"]]
    references = examples["reference"] if isinstance(examples["reference"], list) else [examples["reference"]]

    formatted_prompts = prompts

    tokens = tokenizer(formatted_prompts, truncation=True, padding="max_length",
                       max_length=args.max_source_length)
    return {
        "input_ids":       tokens["input_ids"],
        "attention_mask":  tokens["attention_mask"],
        "prompt":          formatted_prompts,
        "reference":       references,
    }

train_dataset = dataset["train"].shuffle(seed=42).map(preprocess_function, batched=True, batch_size=64, load_from_cache_file=False)
eval_dataset  = dataset["validation"].shuffle(seed=42).map(preprocess_function, batched=True, batch_size=64, load_from_cache_file=False)

# === GENERATION CONFIG ===
try:
    generation_config = GenerationConfig.from_pretrained(args.model_name)
except Exception:
    from transformers import GenerationConfig as GenCfg
    generation_config = GenCfg()
generation_config.max_new_tokens = args.max_target_length
generation_config.pad_token_id   = tokenizer.eos_token_id
generation_config.top_k          = 0
generation_config.top_p          = 1.0
generation_config.do_sample      = True
generation_config.min_new_tokens = 2

# === PPO CONFIG ===
ppo_config = PPOConfig(
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    mini_batch_size=4,
    num_ppo_epochs=args.train_epochs,
    gradient_accumulation_steps=args.grad_acc,
    logging_steps=50,
    fp16=False,
    bf16=False,
    max_grad_norm=1,
    save_strategy="no",
    save_steps=None,
    eval_strategy="no",
    kl_coef=0.2,
    temperature=0.5,
    missing_eos_penalty=1.0,
    output_dir=OUTPUT_DIR,
)

# === METRIC SETUP ===
codebleu_metric = None
if args.metric == "codebleu":
    codebleu_metric = evaluate.load("k4black/codebleu")
    
semgrep_configs = [
                "p/trailofbits", "p/default", "p/bandit", "p/comment", "p/python", "p/cwe-top-25",
                "p/owasp-top-ten", "p/r2c-security-audit", "p/insecure-transport", "p/secrets",
                "r/python.attr.correctness.mutable-initializer.attr-mutable-initializer",
                "r/python.bokeh.maintainability.deprecated.deprecated_apis.bokeh-deprecated-apis",
                "r/python.click.best-practice.echo-style.use-click-secho",
                "r/python.correctness.socket-shutdown-close.socket-shutdown-close",
                "r/python.correctness.suppressed-exception-handling-finally-break.suppressed-exception-handling-finally-break",
                "r/python.django.best-practice.json_response.use-json-response",
                "r/python.django.best-practice.upsell_django_environ.use-django-environ",
                "r/python.django.best-practice.use-onetoonefield.use-onetoonefield",
                "r/python.django.correctness.model-save.django-db-model-save-super",
                "r/python.django.correctness.nontext-field-must-set-null-true.nontext-field-must-set-null-true",
                "r/python.django.correctness.string-field-null-checks.no-null-string-field",
                "r/python.django.correctness.string-field-null-checks.string-field-must-set-null-true",
                "r/python.django.correctness.use-decimalfield-for-money.use-decimalfield-for-money",
                "r/python.django.maintainability.duplicate-path-assignment.conflicting-path-assignment",
                "r/python.django.maintainability.duplicate-path-assignment.duplicate-name-assignment",
                "r/python.django.maintainability.duplicate-path-assignment.duplicate-path-assignment",
                "r/python.django.maintainability.duplicate-path-assignment.duplicate-path-assignment-different-names",
                "r/python.django.performance.access-foreign-keys.access-foreign-keys",
                "r/python.django.performance.upsell-count.use-count-method",
                "r/python.django.performance.upsell_earliest_latest.use-earliest-or-latest",
                "r/python.flask.best-practice.get-class-method-with-side-effects.flask-class-method-get-side-effects",
                "r/python.flask.best-practice.use-jsonify.use-jsonify",
                "r/python.flask.correctness.access-request-in-wrong-handler.avoid-accessing-request-in-wrong-handler",
                "r/python.flask.correctness.same-handler-name.flask-duplicate-handler-name",
                "r/python.flask.maintainability.deprecated.deprecated-apis.flask-deprecated-apis",
                "r/python.lang.best-practice.hardcoded-tmp-path.hardcoded-tmp-path",
                "r/python.lang.best-practice.logging-error-without-handling.logging-error-without-handling",
                "r/python.lang.best-practice.manual-collections-create.manual-counter-create",
                "r/python.lang.best-practice.manual-collections-create.manual-defaultdict-dict-create",
                "r/python.lang.best-practice.manual-collections-create.manual-defaultdict-list-create",
                "r/python.lang.best-practice.manual-collections-create.manual-defaultdict-set-create",
                "r/python.lang.best-practice.missing-hash-with-eq.missing-hash-with-eq",
                "r/python.lang.best-practice.open-never-closed.open-never-closed",
                "r/python.lang.best-practice.pass-body.pass-body-fn",
                "r/python.lang.best-practice.pass-body.pass-body-range",
                "r/python.lang.best-practice.pdb.python-debugger-found",
                "r/python.lang.best-practice.sleep.arbitrary-sleep",
                "r/python.lang.best-practice.unspecified-open-encoding.unspecified-open-encoding",
                "r/python.lang.correctness.baseclass-attribute-override.baseclass-attribute-override",
                "r/python.lang.correctness.cannot-cache-generators.cannot-cache-generators",
                "r/python.lang.correctness.common-mistakes.default-mutable-dict.default-mutable-dict",
                "r/python.lang.correctness.common-mistakes.default-mutable-list.default-mutable-list",
                "r/python.lang.correctness.common-mistakes.is-comparison-string.identical-is-comparison",
                "r/python.lang.correctness.common-mistakes.is-comparison-string.string-is-comparison",
                "r/python.lang.correctness.common-mistakes.is-not-is-not.is-not-is-not",
                "r/python.lang.correctness.common-mistakes.string-concat-in-list.string-concat-in-list",
                "r/python.lang.correctness.concurrent.uncaught-executor-exceptions",
            ]

# === REWARD MODEL ===
class MyRewardModel(torch.nn.Module):
    def __init__(self, tokenizer, metric="bertscore"):
        super().__init__()
        self.tokenizer = tokenizer
        self.metric    = metric

    def forward(self, query_responses, contexts, references):
        device = query_responses.device
        completions = []
        for i, resp_ids in enumerate(query_responses):
            resp_list  = resp_ids.detach().cpu().tolist() if isinstance(resp_ids, torch.Tensor) else list(resp_ids)
            prompt_len = 0
            try:
                prompt_len = len(self.tokenizer.encode(contexts[i] or "", add_special_tokens=False))
            except Exception:
                pass
            completion = self.tokenizer.decode(resp_list[prompt_len:], skip_special_tokens=True)
            completion = re.sub(r'^\s*Code:\s*', '', completion).strip()
            completions.append(completion)

        if self.metric == "edit":
            rewards = [edit_dist(c, r) for c, r in zip(completions, references)]
        elif self.metric == "bertscore":
            rewards = compute_bert_score(completions, references)
        elif self.metric == "codebleu":
            rewards = compute_codebleu_score(completions, references, codebleu_metric)
        elif self.metric == "crystalbleu":
            rewards = calc_crystalBLEU(completions, references, re_compute_ngrams=False, cache_folder=CRYSTAL_CACHE)
        elif self.metric == "pylint":
            pylint_args = ['--disable=all', '--enable=E,W', '--disable=syntax-error', "--disable=undefined-variable"]
            rewards = pylint_quality_reward(completions, references, args.pylint_workers, contexts, pylint_args=pylint_args)
        elif self.metric == "semgrep":
            rewards = semgrep_security_reward(completions, args.semgrep_workers, configs=semgrep_configs)
        elif self.metric == "custom_pylint":
            WEIGHT_LEN, WEIGHT_PYLINT, WEIGHT_CRYSTAL = 0.5, 0.3, 0.2
            pylint_args = ['--disable=all', '--enable=E,W', '--disable=syntax-error', "--disable=undefined-variable"]
            len_scores      = calculate_token_length_similarity(completions, references, self.tokenizer)
            crystal_rewards = calc_crystalBLEU(completions, references, re_compute_ngrams=False, cache_folder=CRYSTAL_CACHE)
            pylint_rewards  = pylint_quality_reward(completions, references, args.pylint_workers, contexts, pylint_args=pylint_args)
            rewards = [WEIGHT_LEN*l + WEIGHT_CRYSTAL*c + WEIGHT_PYLINT*p for l,c,p in zip(len_scores, crystal_rewards, pylint_rewards)]
        elif self.metric == "custom_semgrep":
            WEIGHT_LEN, WEIGHT_SEMGREP, WEIGHT_CRYSTAL = 0.5, 0.3, 0.2
            len_scores      = calculate_token_length_similarity(completions, references, self.tokenizer)
            crystal_rewards = calc_crystalBLEU(completions, references, re_compute_ngrams=False, cache_folder=CRYSTAL_CACHE)
            semgrep_rewards = semgrep_security_reward(completions, args.semgrep_workers, configs=semgrep_configs)
            rewards = [WEIGHT_LEN*l + WEIGHT_CRYSTAL*c + WEIGHT_SEMGREP*p for l,c,p in zip(len_scores, crystal_rewards, semgrep_rewards)]
        else:
            raise ValueError(f"Unknown metric {self.metric}")

        import re as _re
        print(f"[REWARD] Metric={self.metric} Scores (len={len(rewards)}). First 5: {rewards[:5]}", flush=True)
        return torch.tensor(rewards, dtype=torch.float32, device=device)

reward_model = MyRewardModel(tokenizer, metric=args.metric)

# === DATA COLLATOR ===
def ppo_data_collator(batch):
    input_ids      = pad_sequence([torch.tensor(ex["input_ids"],      dtype=torch.long) for ex in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch], batch_first=True, padding_value=0)
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "prompts":        [ex["prompt"]    for ex in batch],
        "references":     [ex["reference"] for ex in batch],
    }

# === CUSTOM PPO TRAINER ===
from trl.trainer import PPOTrainer as _BasePPOTrainer

class PPOTrainerWithRefs(_BasePPOTrainer):
    def compute_rewards(self, samples, query_responses, **kwargs):
        print("[TRAINER] USING CUSTOM REWARD FUNCTION", flush=True)
        return self.reward_model(query_responses, samples["prompts"], samples["references"])

if __name__ == "__main__":
    import re
    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device      = accelerator.device
    print(
        f"[INFO] WORLD_SIZE={accelerator.num_processes} "
        f"LOCAL_RANK={accelerator.local_process_index} "
        f"GLOBAL_RANK={accelerator.process_index} "
        f"PID={os.getpid()}",
        flush=True
    )

    # === LOAD MODEL ===
    model       = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    ref_model   = create_reference_model(model)
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

    if hasattr(model, "v_head"):
        for param in model.v_head.parameters():
            param.requires_grad = False

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    for m in [model, ref_model, value_model]:
        if not hasattr(m, "base_model_prefix"):
            m.base_model_prefix = "pretrained_model"

    base_prefix  = value_model.base_model_prefix if hasattr(value_model, value_model.base_model_prefix) else "pretrained_model"
    pretrained_vm = getattr(value_model, base_prefix)
    for param in pretrained_vm.parameters():
        param.requires_grad = False
    for param in value_model.v_head.parameters():
        param.requires_grad = True

    model.generation_config     = generation_config
    ref_model.generation_config = generation_config

    if args.metric == "bertscore":
        init_bert_scorer(device)

    # === PPO TRAINER ===
    ppo_trainer = PPOTrainerWithRefs(
        args=ppo_config,
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ppo_data_collator,
        reward_model=reward_model,
        value_model=value_model,
    )

    # === TRAIN ===
    ppo_trainer.train()

    # === SAVE ===
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(ppo_trainer.model)
        if hasattr(unwrapped, "policy"):
            unwrapped.policy.save_pretrained(OUTPUT_DIR)
        else:
            unwrapped.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        if generation_config is not None:
            generation_config.save_pretrained(OUTPUT_DIR)
        print(f"[INFO] Model saved to {OUTPUT_DIR}", flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()