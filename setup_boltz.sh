#!/usr/bin/env bash
# =============================================================================
# One-time setup: create a conda env on PROJECT storage and install Boltz-2.
# Run this ONCE on a LOGIN node (needs internet).
#
# Why project storage and not $HOME: a conda env is tens of thousands of small
# files, and NAISS home dirs usually have a low file-count (inode) quota.
# =============================================================================
set -euo pipefail

# ---- EDIT THESE (Arrhenius-specific) ---------------------------------------
# Absolute path on project/scratch storage where the env should live:
ENV_PREFIX="${ENV_PREFIX:-/path/to/project/storage/envs/boltz2}"      # <-- EDIT
#
# If Arrhenius ships a conda/miniforge/mamba module, load it. Find it with:
#   module spider conda      (also try: miniforge / mambaforge / anaconda)
# module load <conda-module>                                          # <-- maybe EDIT
# ----------------------------------------------------------------------------

# Prefer mamba (fast solver) if present, else conda.
CONDA_EXE="$(command -v mamba || command -v conda || true)"
if [[ -z "$CONDA_EXE" ]]; then
  echo "ERROR: no conda/mamba on PATH. Load the conda module first (module spider conda)." >&2
  exit 1
fi
echo "Using solver: $CONDA_EXE"

# Create the env at an explicit prefix. Boltz supports Python 3.10-3.12.
"$CONDA_EXE" create -y -p "$ENV_PREFIX" python=3.11 pip

# Activate the prefix env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Install Boltz-2 (PyPI package name is 'boltz'; 2.x == Boltz-2).
# The [cuda] extra pulls NVIDIA cuEquivariance kernels for GPU acceleration.
pip install --upgrade pip
pip install "boltz[cuda]" -U

echo
echo "---- versions ----"
pip show boltz | grep -E 'Name|Version' || true
python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
echo
echo "Done. Env prefix: $ENV_PREFIX"
echo "Next: edit and run prepare_msa.sh on a node with internet."
