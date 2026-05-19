#!/bin/bash
#PBS -S /bin/bash
#
# Single-experiment GPU training job for DA6401 Assignment 3.
# Resource directives (queue, walltime, GPUs) are set by submit_all_experiments.sh.
#
# Submit with EXPERIMENT set, e.g.:
#   qsub -N da6401-exp1 -v EXPERIMENT=exp1 \
#        -o logs/exp1.log -e logs/exp1.err scripts/pbs_train_job.sh
#
# Or use: bash scripts/submit_all_experiments.sh

set -euo pipefail

if [[ -z "${EXPERIMENT:-}" ]]; then
    echo "ERROR: EXPERIMENT not set. Pass via qsub -v EXPERIMENT=exp1" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpce_config.sh
source "${SCRIPT_DIR}/hpce_config.sh"

PROJECT_DIR="${PBS_O_WORKDIR:-${PROJECT_DIR}}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "=== DA6401 training job ==="
echo "Job ID:    ${PBS_JOBID:-local}"
echo "Node:      ${HOSTNAME:-unknown}"
echo "Experiment: ${EXPERIMENT}"
echo "Project:   ${PROJECT_DIR}"
echo "Date:      $(date)"
echo "PBS_NODEFILE:"
cat "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || true

_run_training() {
    local workdir="$1"
    cd "${workdir}"

    # --- Python environment ---
    if [[ -n "${CONDA_ENV}" ]]; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
    elif [[ -n "${VENV_PATH}" && -f "${VENV_PATH}/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${VENV_PATH}/bin/activate"
    fi

    export PYTHONUNBUFFERED=1
    export WANDB_PROJECT

    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi
    fi

    python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"

    python train.py --experiment "${EXPERIMENT}"
}

_copy_project_to_scratch() {
    local dest="$1"
    mkdir -p "${dest}"
    rsync -a --exclude='.git' --exclude='__pycache__' --exclude='wandb' \
        --exclude='*.pt' --exclude='logs' --exclude='scratch' \
        "${PROJECT_DIR}/" "${dest}/"
}

_copy_results_back() {
    local src="$1"
    mkdir -p "${PROJECT_DIR}/logs"
    # Checkpoints and run artifacts
    shopt -s nullglob
    for f in "${src}"/best_checkpoint_"${EXPERIMENT}".pt; do
        [[ -f "$f" ]] && cp -f "$f" "${PROJECT_DIR}/"
    done
    if [[ -d "${src}/wandb" ]]; then
        mkdir -p "${PROJECT_DIR}/wandb"
        rsync -a "${src}/wandb/" "${PROJECT_DIR}/wandb/"
    fi
    shopt -u nullglob
}

if [[ "${USE_SCRATCH}" == "1" && -n "${PBS_JOBID:-}" ]]; then
    tpdir="$(echo "${PBS_JOBID}" | cut -f1 -d.)"
    tempdir="${HOME}/scratch/job${tpdir}"
    mkdir -p "${tempdir}"
    echo "Staging project at ${tempdir}"
    _copy_project_to_scratch "${tempdir}"
    _run_training "${tempdir}"
    echo "Copying results back to ${PROJECT_DIR}"
    _copy_results_back "${tempdir}"
    cd "${PROJECT_DIR}"
    rm -rf "${tempdir}"
else
    _run_training "${PROJECT_DIR}"
fi

echo "=== Job finished: ${EXPERIMENT} ==="
