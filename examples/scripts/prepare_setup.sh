#!/usr/bin/env bash
set -Eeuo pipefail

# Launch the full YOLO SHAP-BPT-SAM experiment in a persistent tmux session.
# Usage: bash examples/scripts/prepare_setup.sh

SESSION_NAME="shapbpt"
PROJECT_DIR="/beegfs/home/mrashid/repos/XAI/shap_bpt_sam"
VENV_ACTIVATE="/beegfs/home/mrashid/pt_312/bin/activate"
PYTORCH_PACKAGES="/opt/pytorch-v2.7.1/lib/python3.12/site-packages"
CONFIG="examples/configs/MSCOCO_epito.yaml"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session '${SESSION_NAME}' already exists."
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    exit 1
fi

# Everything after bash -lc runs on the allocated A100 node. Keeping the
# tmux window after completion makes the final output and errors inspectable.
tmux new-session -d -s "${SESSION_NAME}" \
    "srun -p epito --gres=gpu:a100:1 bash -lc '\
        source ${VENV_ACTIVATE}; \
        export PYTHONPATH=${PYTORCH_PACKAGES}; \
        cd ${PROJECT_DIR}; \
        python3 examples/scripts/run_yolo_full.py --config ${CONFIG}'"

tmux set-option -t "${SESSION_NAME}" remain-on-exit on

echo "Started experiment in tmux session '${SESSION_NAME}'."
echo "Detach with: Ctrl-b d"
echo "Reattach with: tmux attach -t ${SESSION_NAME}"
tmux attach-session -t "${SESSION_NAME}"
