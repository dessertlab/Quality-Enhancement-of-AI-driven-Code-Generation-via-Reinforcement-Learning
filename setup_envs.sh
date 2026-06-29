#!/usr/bin/env bash
# setup_envs.sh
# Creates two conda environments for the project.
#
# Usage (from code_repository/):
#   bash setup_envs.sh
#
# Environments created:
#   myEnv     — Python 3.9.21 — main training/inference env
#   myPyTEnv  — Python 3.7.16 — python-taint static analysis env

set -eo pipefail
PS1="${PS1:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" 
TRL_DIR="$SCRIPT_DIR/trl"

# === Verify conda is available ===
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "========================================================"
echo " Setting up project environments"
echo " Repo root : $SCRIPT_DIR"
echo "========================================================"

# === myEnv (Python 3.9.21) ===
echo ""
echo "[1/2] Creating myEnv (Python 3.9.21)..."

if conda env list | grep -q "^myEnv "; then
    echo "  myEnv already exists — skipping creation, updating packages only."
    conda activate myEnv
else
    conda create -y -n myEnv python=3.9.21
    conda activate myEnv
fi

echo "  Installing packages from requirements.txt..."
pip install -r "$SCRIPT_DIR/requirements.txt"


if [ -d "$TRL_DIR" ]; then
    echo "  Installing custom trl from $TRL_DIR..."
    pip install -e "$TRL_DIR" --no-deps
else
    echo "  WARNING: trl/ folder not found at $TRL_DIR — skipping trl install."
    echo "           Copy your modified trl/ folder to the repo root and re-run."
fi

conda deactivate
echo "  myEnv ready."

# === myPyTEnv (Python 3.7.16) ===
echo ""
echo "[2/2] Creating myPyTEnv (Python 3.7.16)..."

if conda env list | grep -q "^myPyTEnv "; then
    echo "  myPyTEnv already exists — skipping creation, updating packages only."
    conda activate myPyTEnv
else
    conda create -y -n myPyTEnv python=3.7.16
    conda activate myPyTEnv
fi

echo "  Installing packages from requirements_pyt.txt..."
pip install -r "$SCRIPT_DIR/requirements_pyt.txt"

conda deactivate
echo "  myPyTEnv ready."

echo ""
echo "========================================================"
echo " Setup complete."
echo ""
echo " Activate environments with:"
echo "   conda activate myEnv      # main training/inference"
echo "   conda activate myPyTEnv   # python-taint analysis"
echo ""
echo " To use the custom trl in myEnv, scripts insert"
echo " $(realpath "$TRL_DIR") into sys.path at runtime."
echo "========================================================"