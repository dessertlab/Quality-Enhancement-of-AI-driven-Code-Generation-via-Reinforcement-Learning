# torchrun --nproc_per_node=2 training/dpo/run_dpo_DeepSeek.py
import os
os.environ["HF_HUB_ENABLE_HF_XET"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_CACHE", "/tmp/hfcache")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/tmp/hfhome")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/tmp/hftransformers")
os.environ["TMPDIR"] = os.getenv("TMPDIR", "/tmp")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import sys
import argparse
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR)) 

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import DPOTrainer, DPOConfig

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description="DPO training for DeepSeek")
parser.add_argument(
    "--finetuned",
    type=str,
    choices=["yes", "no"],
    default="yes",
    help=(
        "Whether to start from the finetuned checkpoint (yes) "
        "or from the pretrained base model (no). "
        "Ignored if --model_path is provided."
    ),
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Explicit path to the model to use. Overrides --finetuned.",
)
args = parser.parse_args()

# === MODEL PATH ===
PRETRAINED_MODEL  = "deepseek-ai/deepseek-coder-1.3b-instruct"
FINETUNED_MODEL   = os.path.join(REPO_ROOT, "models", "sft", "DeepSeek-finetuned")
 
if args.model_path is not None:
    MODEL_NAME = args.model_path
    print(f"[INFO] Using explicit model path: {MODEL_NAME}")
elif args.finetuned == "yes":
    MODEL_NAME = FINETUNED_MODEL
    print(f"[INFO] Using finetuned model: {MODEL_NAME}")
else:
    MODEL_NAME = PRETRAINED_MODEL
    print(f"[INFO] Using pretrained model: {MODEL_NAME}")
    
# === OUTPUT PATH ===
if args.finetuned == "yes":
    OUTPUT_DIR = os.path.join(REPO_ROOT, "models/sft_dpo/DeepSeek-sft_dpo")
else:
    OUTPUT_DIR = os.path.join(REPO_ROOT, "models/dpo/DeepSeek-dpo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_EPOCHS = 10
BATCH_SIZE = 4   
GRAD_ACC = 16
LEARNING_RATE = 5e-6
WARMUP_RATIO = 0.01
BETA = 0.1
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 512
MAX_LENGTH = MAX_SOURCE_LENGTH + MAX_TARGET_LENGTH

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# === LOAD DPO DATASET ===
def load_dpo_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

dpo_dataset = load_dpo_dataset(
    os.path.join(REPO_ROOT, "datasets", "dpo", "dpo_training_dataset.json")
)
print(f"[INFO] DPO dataset size: {len(dpo_dataset)}")

# === FORMAT PROMPT ===
def format_dpo_prompt(example):
    example["prompt"] = f"### Instruction:\n{example['prompt']}\n### Response:\n"
    return example

train_dataset = dpo_dataset.map(format_dpo_prompt, num_proc=8)

# === MODEL ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()

# === DPO CONFIG ===
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,

    # --- Training schedule ---
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,

    # --- DPO-specific ---
    beta=BETA,
    loss_type="sigmoid",
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_SOURCE_LENGTH,

    # --- Precision & Memory ---
    fp16=False,
    bf16=True, # Attivato per DeepSeek
    gradient_checkpointing=True,

    # --- Evaluation & checkpointing ---
    eval_strategy="no", 
    save_strategy="no",
    load_best_model_at_end=False,
    greater_is_better=False,

    # --- Logging ---
    logging_dir="./logs_dpo_deepseek",
    logging_steps=50,
    report_to="none",

    # --- Misc ---
    remove_unused_columns=False,
    dataloader_num_workers=4,
)

# === DPO TRAINER ===
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

# === TRAIN ===
print("[INFO] Starting DPO training for DeepSeek...", flush=True)
dpo_trainer.train()

# === SAVE ===
dpo_trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[INFO] Model saved to {OUTPUT_DIR}", flush=True)