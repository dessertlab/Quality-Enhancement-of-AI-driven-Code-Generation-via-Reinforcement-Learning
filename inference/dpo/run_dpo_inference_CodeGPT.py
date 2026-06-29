# python run_dpo_inference_CodeGPT.py --model_path models/dpo/CodeGPT-dpo --model_type dpo --batch_size 8 --max_source_length 256 --max_target_length 512

import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def main():
    parser = argparse.ArgumentParser(description="Inference script for CodeGPT model")

    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get(
            "MODEL_PATH",
            "models/dpo/CodeGPT-dpo"
        ),
        help="Path to the saved DPO-trained model"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["pretrained", "finetuned", "dpo", "sft_dpo", "ppo", "sft_ppo"],
        default=os.environ.get("MODEL_TYPE", "dpo"),
    )   
    
    parser.add_argument(
        "--ppo_metric",
        type=str,
        choices=["bertscore", "codebleu", "edit", "crystalbleu", "pylint","semgrep", "custom_pylint","custom_semgrep"],
        default=os.environ.get("METRIC", "pylint"),
        help="Reward metric to use: bertsore, codebleu, crystralbleu, edit, pylint, semgrep, custom_pylint, custom_semgrep"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_source_length", type=int, default=256, help="Max length of input prompt")
    parser.add_argument("--max_target_length", type=int, default=512, help="Max new tokens to generate")

    args = parser.parse_args()

    # === PATHS ===
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    TEST_DATA_PATH = os.path.join(REPO_ROOT, "datasets", "dpo", "dpo_test_dataset.json")
    if args.model_type in ("ppo", "sft_ppo"):
        OUTPUT_PATH = os.path.join(
            REPO_ROOT, "results", "inference", args.model_type,
            f"CodeGPT_{args.model_type}_inference_{args.ppo_metric}.jsonl"
        )
    elif args.train_type == "finetuned":
        OUTPUT_PATH = os.path.join(
            REPO_ROOT, "results", "inference", "sft",
            f"CodeGPT_{args.train_type}_inference_test_{args.testset_type}.jsonl"
        )
    else:
        OUTPUT_PATH = os.path.join(
            REPO_ROOT, "results", "inference", args.model_type,
            f"CodeGPT_{args.model_type}_inference.jsonl"
        )
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # === 1. LOAD TOKENIZER ===
    print(f"[INFO] Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  

    # === 2. LOAD MODEL ===
    print(f"[INFO] Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded on {device}")

    # === 3. GENERATION CONFIG ===
    try:
        generation_config = GenerationConfig.from_pretrained(args.model_path)
        generation_config.max_new_tokens = args.max_target_length
        generation_config.pad_token_id   = tokenizer.eos_token_id
        generation_config.top_k          = 0
        generation_config.top_p          = 1.0
        generation_config.do_sample      = True
        generation_config.min_new_tokens = 2
    except Exception:
        generation_config = GenerationConfig(
            max_new_tokens=args.max_target_length,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_k=0,
            top_p=1.0,
            min_new_tokens=2,
        )

    # === 4. LOAD DATASET ===
    print("[INFO] Loading DPO test dataset...")
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    prompts    = [ex["prompt"]    for ex in raw_data]
    references = [ex["reference"] for ex in raw_data]
    print(f"[INFO] Total examples: {len(prompts)}")

    # === 5. RUN INFERENCE ===
    print(f"[INFO] Starting inference on {len(prompts)} examples...")
    results = []

    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts    = prompts[i : i + args.batch_size]
        batch_references = references[i : i + args.batch_size]

        inputs = tokenizer(
            batch_prompts,
            truncation=True,
            padding=True,
            max_length=args.max_source_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
            )

        # === 6. DECODE & COLLECT ===
        for j, output_ids in enumerate(outputs):
            input_len     = inputs["input_ids"][j].shape[0]
            generated_ids = output_ids[input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            results.append({
                "prompt":         batch_prompts[j],
                "reference":      batch_references[j],
                "generated_code": generated_text.strip(),
            })

    # === 7. SAVE RESULTS ===
    print(f"[INFO] Saving results to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("[INFO] Inference completed!")

if __name__ == "__main__":
    main()