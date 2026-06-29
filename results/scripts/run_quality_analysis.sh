#!/usr/bin/env bash
# run_quality_pipeline.sh
# Usage: bash results/scripts/run_quality_pipeline.sh results/inference/path/to/file.jsonl
#
# Steps:
#   1. Detect model from filename prefix (CodeGen/CodeGPT/DeepSeek/QwenCoder)
#   2. Clean the file with the matching cleaning script → inference_cleaned/
#   3. Run quality_analyze_code.py on the cleaned file
#   4. Run quality_process_results.py with --empty_files captured from step 3

set -eo pipefail
PS1="${PS1:-}"

if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <relative_path_to_input.jsonl>"
    exit 1
fi

INPUT_FILE="$(realpath "$1")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   
RESULTS_DIR="$(dirname "$SCRIPT_DIR")"                        
INFERENCE_CLEANED="$RESULTS_DIR/inference_cleaned"

BASENAME="$(basename "$INPUT_FILE")"

if [[ "$BASENAME" == CodeGen* ]]; then
    CLEANER="$SCRIPT_DIR/CodeGen_cleaning_results.py"
    MODEL="CodeGen"
elif [[ "$BASENAME" == CodeGPT* ]]; then
    CLEANER="$SCRIPT_DIR/CodeGPT_cleaning_results.py"
    MODEL="CodeGPT"
elif [[ "$BASENAME" == DeepSeek* ]]; then
    CLEANER="$SCRIPT_DIR/DeepSeek_cleaning_results.py"
    MODEL="DeepSeek"
elif [[ "$BASENAME" == QwenCoder* ]]; then
    CLEANER="$SCRIPT_DIR/QwenCoder_cleaning_results.py"
    MODEL="QwenCoder"
else
    echo "ERROR: cannot detect model from filename '$BASENAME'."
    echo "  Expected prefix: CodeGen / CodeGPT / DeepSeek / QwenCoder"
    exit 1
fi

echo "========================================================"
echo " Model   : $MODEL"
echo " Input   : $INPUT_FILE"
echo " Cleaner : $CLEANER"
echo "========================================================"

echo ""
echo "[1/3] Cleaning inference file..."
python "$CLEANER" --input_file "$INPUT_FILE"

INFERENCE_DIR="$RESULTS_DIR/inference"
REL_PATH="${INPUT_FILE#$INFERENCE_DIR/}"           
STEM="${BASENAME%.jsonl}"
CLEANED_FILE="$INFERENCE_CLEANED/$(dirname "$REL_PATH")/${STEM}_cleaned.jsonl"
CLEANED_FILE="$(realpath --canonicalize-missing "$CLEANED_FILE")"

echo "  Cleaned file: $CLEANED_FILE"

if [ ! -f "$CLEANED_FILE" ]; then
    echo "ERROR: cleaned file not found at $CLEANED_FILE" >&2
    exit 1
fi

echo ""
echo "[2/3] Running Semgrep quality analysis..."
ANALYZE_LOG="$(mktemp)"
python "$SCRIPT_DIR/quality_analyze_code.py" "$CLEANED_FILE" 2>&1 | tee "$ANALYZE_LOG"

EMPTY_FILES=$(grep -oP '(?<=Empty files encountered: )\d+' "$ANALYZE_LOG" || echo "0")
rm -f "$ANALYZE_LOG"
echo "  Empty files: $EMPTY_FILES"

QUALITY_OUT="$SCRIPT_DIR/quality_outputs"
CLEANED_STEM="$(basename "$CLEANED_FILE" .jsonl)"
BATCH_PREFIX="$QUALITY_OUT/${CLEANED_STEM}_semgrep_results_batch"

BATCH_COUNT=$(ls "${BATCH_PREFIX}"_*.json 2>/dev/null | wc -l)
if [ "$BATCH_COUNT" -eq 0 ]; then
    echo "ERROR: no semgrep batch JSON files found matching ${BATCH_PREFIX}_*.json" >&2
    exit 1
fi

echo ""
echo "[3/3] Processing Semgrep results (batches=$BATCH_COUNT, empty_files=$EMPTY_FILES)..."
python "$SCRIPT_DIR/quality_process_results.py" \
    "$BATCH_PREFIX" \
    "$BATCH_COUNT" \
    --empty_files "$EMPTY_FILES"

echo ""
echo "========================================================"
echo " Pipeline complete."
echo "========================================================"