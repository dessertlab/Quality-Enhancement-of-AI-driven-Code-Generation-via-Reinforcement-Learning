# torchrun --nproc_per_node=2 training/sft/run_sft_QwenCoder.py
import torch
import json
import os
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, 
)
from datasets import Dataset

# === CONFIG ===
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR)) 
OUTPUT_DIR = os.path.join(REPO_ROOT, "models", "sft", "QwenCoder-finetuned")
TRAIN_EPOCHS      = 10
BATCH_SIZE        = 8
GRAD_ACC          = 8
LEARNING_RATE     = 5e-6
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 512
SYSTEM_PROMPT = "You are a helpful coding assistant."

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# === MODEL ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)

# === LOAD DATASET ===
def load_json_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

train_dataset = load_json_dataset(os.path.join(REPO_ROOT, "datasets", "sft", "sft_train_dataset.json"))
eval_dataset  = load_json_dataset(os.path.join(REPO_ROOT, "datasets", "sft", "sft_validation_dataset.json"))

print("Train set length:", len(train_dataset))
print("Eval set length:",  len(eval_dataset))

# === PREPROCESSING ===
def build_prompt_prefix(instruction: str) -> str:
    """Costruisce il prefisso prompt (senza la risposta) con add_generation_prompt=True."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def build_full_text(instruction: str, output: str) -> str:
    """Testo completo: prompt + risposta + token di chiusura."""
    prefix = build_prompt_prefix(instruction)
    return prefix + output + "<|im_end|>"

def preprocess_function(examples):
    full_texts = [
        build_full_text(instr, out)
        for instr, out in zip(examples["instruction"], examples["output"])
    ]

    tokenized = tokenizer(
        full_texts,
        max_length=MAX_SOURCE_LENGTH + MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )

    labels = []
    for input_ids, instr in zip(tokenized["input_ids"], examples["instruction"]):
        prefix_text = build_prompt_prefix(instr)
        prefix_ids  = tokenizer.encode(prefix_text, add_special_tokens=False)
        mask_len    = len(prefix_ids)
        mask = [-100] * mask_len + input_ids[mask_len:]
        labels.append(mask[: len(input_ids)])

    tokenized["labels"] = labels
    return tokenized

train_dataset = train_dataset.map(
    preprocess_function, batched=True, num_proc=32,
    load_from_cache_file=False, remove_columns=train_dataset.column_names
)
eval_dataset = eval_dataset.map(
    preprocess_function, batched=True, num_proc=32,
    load_from_cache_file=False, remove_columns=eval_dataset.column_names
)

print("Train set size:", len(train_dataset), train_dataset)
print("Validation set size:", len(eval_dataset), eval_dataset)

# === DATA COLLATOR ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === TRAINING ARGS ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=False,
    bf16=True,
    logging_dir="./logs_finetuning_qwen",
    logging_steps=50,
    report_to="none",
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
)

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === TRAIN ===
trainer.train()

# === SAVE ===
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)