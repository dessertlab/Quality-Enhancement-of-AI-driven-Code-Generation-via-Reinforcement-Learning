# python run_ppo_inference_QwenCoder.py --model_path models/ppo/QwenCoder-ppo-pylint --batch_size 8 --max_source_length 256 --max_target_length 512 --metric pylint --train_type ppo --testset_type secure

import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead

SYSTEM_PROMPT = "You are a helpful coding assistant."

def extract_prompts(example):
    msgs = example["messages"]
    doc = next((m["content"] for m in msgs if m["role"] == "user"), "")
    reference = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    
    return {"prompt": doc, "reference": reference}

def main():
    parser = argparse.ArgumentParser(description="Inference script for QwenCoder model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=os.environ.get(
            "MODEL_PATH", 
            "models/ppo/QwenCoder-ppo-pylint"
        ),
        help="Path to the saved trained model"
    )
    
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_source_length", type=int, default=256, help="Max length of input prompt")
    parser.add_argument("--max_target_length", type=int, default=512, help="Max tokens to generate")
    
    parser.add_argument(
        "--metric",
        type=str,
        choices=["bertscore", "codebleu", "edit", "crystalbleu", "pylint","semgrep", "custom_pylint","custom_semgrep"],
        default=os.environ.get("METRIC", "bertscore"),
        help="Reward metric to use: bertscore, codebleu or edit"
    )
    parser.add_argument(
        "--train_type",
        type=str,
        choices=["pretrained", "finetuned", "ppo", "sft_ppo", "sft_dpo", "dpo"],
        default=os.environ.get("TRAIN_TYPE", "pretrained"),
        help="Model type to use: pretrained, finetuned, ppo, sft_ppo, dpo, sft_dpo"
    )
    parser.add_argument(
        "--testset_type",
        type=str,
        choices=["secure", "insecure"],
        default=os.environ.get("TESTSET_TYPE", "insecure"),
        help="Test type to use: secure or insecure"
    )
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # === CONFIG ===
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))    
    REPO_ROOT    = os.path.dirname(os.path.dirname(SCRIPT_DIR))  

    model_name   = "QwenCoder"
    testset_type = args.testset_type
    metric       = args.metric
    train_type   = args.train_type

    TEST_DATA_PATH = os.path.join(
        REPO_ROOT, "datasets", "ppo", "test_datasets",
        f"ppo_test_{testset_type}.jsonl"
    )
    if train_type in ("ppo", "sft_ppo"):
        OUTPUT_PATH = os.path.join(
            REPO_ROOT, "results", "inference", train_type,
            f"{model_name}_{train_type}_inference_test_{testset_type}_{metric}.jsonl"
        )
    elif train_type == "finetuned":
        OUTPUT_PATH = os.path.join(
            REPO_ROOT, "results", "inference", "sft",
            f"{model_name}_{train_type}_inference_test_{testset_type}.jsonl"
        )
    else:
        OUTPUT_PATH = os.path.join(
            REPO_ROOT, "results", "inference", train_type,
            f"{model_name}_{train_type}_inference_test_{testset_type}.jsonl"
        )
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    MAX_SOURCE_LENGTH = args.max_source_length
    MAX_TARGET_LENGTH = args.max_target_length
    
    # === 1. LOAD TOKENIZER ===
    print(f"[INFO] Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=False, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    
    # === 2. LOAD MODEL ===
    print(f"[INFO] Loading model from {args.model_path}...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_path, trust_remote_code=True).to(device)
    
    model.eval() 

    # === 3. GENERATION CONFIG ===
    try:
        generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception:
        generation_config = GenerationConfig()
    
    generation_config.update(
        max_new_tokens=MAX_TARGET_LENGTH,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        top_k=0,
        top_p=1.0,
        min_new_tokens=2
    )

    # === 4. LOAD DATASET ===
    print("[INFO] Loading test dataset...")
    dataset = load_dataset("json", data_files={"test":[TEST_DATA_PATH]})["test"]
    dataset = dataset.map(extract_prompts, remove_columns=["messages"])

    # === 5. RUN INFERENCE ===
    print(f"[INFO] Starting inference on {len(dataset)} examples...")
    
    results = []
    
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[i : i + args.batch_size]
        prompts = batch["prompt"]
        references = batch["reference"]
        
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": p},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        inputs = tokenizer(
            formatted_prompts, 
            truncation=True, 
            padding=True, 
            max_length=MAX_SOURCE_LENGTH, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )

        # === 6. SAVE RESULTS ===
        for j, output_ids in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated_ids = output_ids[input_len:]
            
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            results.append({
                "prompt": prompts[j],
                "reference": references[j],
                "generated_code": generated_text.strip()
            })

    print(f"\n[INFO] Saving results to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print("[INFO] Inference completed!")

if __name__ == "__main__":
    main()