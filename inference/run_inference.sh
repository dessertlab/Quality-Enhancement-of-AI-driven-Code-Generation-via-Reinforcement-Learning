#!/usr/bin/env bash

# run_inference.sh
# Usage:
#   bash inference/run_inference.sh <model_path> <mode>
#
#   <model_path> : path to the model folder — training type and metric are
#                  extracted automatically from the folder name
#                  e.g. models/ppo/CodeGen-ppo-custom_semgrep
#   <mode>       : "dpo" or "ppo"
#                  dpo → single inference run
#                  ppo → runs secure test set, then insecure test set

set -eo pipefail
PS1="${PS1:-}"

if [ $# -lt 2 ]; then
    echo "Usage: bash $0 <model_path> <mode: dpo|ppo>"
    exit 1
fi

MODEL_PATH="$1"
MODE="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   
REPO_ROOT="$(dirname "$SCRIPT_DIR")"                          


if [[ "$MODEL_PATH" = /* ]]; then
    ABS_MODEL_PATH="$MODEL_PATH"
else
    ABS_MODEL_PATH="$REPO_ROOT/$MODEL_PATH"
fi

if [ ! -d "$ABS_MODEL_PATH" ]; then
    echo "ERROR: Model directory not found: $ABS_MODEL_PATH" >&2
    exit 1
fi

MODEL_BASENAME="$(basename "$ABS_MODEL_PATH")"

if [[ "$MODEL_BASENAME" == CodeGen* ]]; then
    MODEL_NAME="CodeGen"
elif [[ "$MODEL_BASENAME" == CodeGPT* ]]; then
    MODEL_NAME="CodeGPT"
elif [[ "$MODEL_BASENAME" == DeepSeek* ]]; then
    MODEL_NAME="DeepSeek"
elif [[ "$MODEL_BASENAME" == QwenCoder* || "$MODEL_BASENAME" == Qwen* ]]; then
    MODEL_NAME="QwenCoder"
else
    echo "ERROR: Cannot detect model name from '$MODEL_BASENAME'."
    echo "  Expected prefix: CodeGen / CodeGPT / DeepSeek / QwenCoder"
    exit 1
fi

if [[ "$MODEL_BASENAME" == *sft_ppo* ]]; then
    TRAIN_TYPE="sft_ppo"
elif [[ "$MODEL_BASENAME" == *ppo* ]]; then
    TRAIN_TYPE="ppo"
elif [[ "$MODEL_BASENAME" == *sft_dpo* ]]; then
    TRAIN_TYPE="sft_dpo"
elif [[ "$MODEL_BASENAME" == *dpo* ]]; then
    TRAIN_TYPE="dpo"
elif [[ "$MODEL_BASENAME" == *finetuned* ]]; then
    TRAIN_TYPE="finetuned"
elif [[ "$MODEL_BASENAME" == *pretrained* ]]; then
    TRAIN_TYPE="pretrained"
else
    echo "ERROR: Cannot detect training type from '$MODEL_BASENAME'."
    echo "  Expected one of: pretrained, finetuned, dpo, sft_dpo, ppo, sft_ppo"
    exit 1
fi

# Format: ModelName-train_type-metric  e.g. CodeGen-ppo-custom_semgrep
# For dpo/sft_dpo there is no metric suffix — default to bertscore (unused in output path)
METRIC=""
if [[ "$TRAIN_TYPE" == *ppo* ]]; then
    METRIC="${MODEL_BASENAME##*-}"
    if [ -z "$METRIC" ]; then
        echo "ERROR: Cannot extract metric from '$MODEL_BASENAME'."
        echo "  Expected format: ModelName-train_type-metric"
        exit 1
    fi
fi

echo "========================================================"
echo " Model path  : $ABS_MODEL_PATH"
echo " Model name  : $MODEL_NAME"
echo " Train type  : $TRAIN_TYPE"
echo " Mode        : $MODE"
if [ -n "$METRIC" ]; then
    echo " Metric      : $METRIC"
fi
echo "========================================================"

if [ "$MODE" = "dpo" ]; then
    SCRIPT="$SCRIPT_DIR/dpo/run_dpo_inference_${MODEL_NAME}.py"

    if [ ! -f "$SCRIPT" ]; then
        echo "ERROR: Script not found: $SCRIPT" >&2
        exit 1
    fi

    echo ""
    METRIC_ARG=()
    if [ -n "$METRIC" ]; then
        METRIC_ARG=(--ppo_metric "$METRIC")
    fi

    echo "[DPO] Running inference..."
    python "$SCRIPT" \
        --model_path  "$ABS_MODEL_PATH" \
        --model_type  "$TRAIN_TYPE" \
        "${METRIC_ARG[@]}"

elif [ "$MODE" = "ppo" ]; then
    SCRIPT="$SCRIPT_DIR/ppo/run_ppo_inference_${MODEL_NAME}.py"

    if [ ! -f "$SCRIPT" ]; then
        echo "ERROR: Script not found: $SCRIPT" >&2
        exit 1
    fi

    echo ""
    echo "[PPO] Running inference on SECURE test set..."
    python "$SCRIPT" \
        --model_path   "$ABS_MODEL_PATH" \
        --train_type   "$TRAIN_TYPE" \
        --metric       "$METRIC" \
        --testset_type "secure"

    echo ""
    echo "[PPO] Running inference on INSECURE test set..."
    python "$SCRIPT" \
        --model_path   "$ABS_MODEL_PATH" \
        --train_type   "$TRAIN_TYPE" \
        --metric       "$METRIC" \
        --testset_type "insecure"

else
    echo "ERROR: mode must be 'dpo' or 'ppo', got '$MODE'" >&2
    exit 1
fi

echo ""
echo "========================================================"
echo " Inference completed."
echo "========================================================"