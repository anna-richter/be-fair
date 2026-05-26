#!/usr/bin/env bash
set -euo pipefail

# Paths
ENV_DIR="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/48-addition_2/node_4/conda_env"
REQ_COMMON="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_ML="/home/anri21/.conda/envs/mlzero_env/lib/python3.11/site-packages/autogluon/assistant/tools_registry/machine learning/requirements.txt"
# start change
# PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/48-addition_2/node_4/generated_code.py"  # original
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/48-addition_2/48-addition_2_generated_code_DDI.py"
# end change

# 1. Create conda env if not exists
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" python=3.11
fi

# 2. Activate env
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv if not present
if ! python -m uv --version >/dev/null 2>&1; then
    pip install uv
fi

# 4. Install requirements with uv (prerelease allowed)
python -m uv pip install --prerelease=allow -r "$REQ_ML" -r "$REQ_COMMON"

# 5. Install extra packages needed by the script
pip install pandas scikit-learn pillow tqdm joblib

# 6. Run the Python script
python "$PY_SCRIPT"