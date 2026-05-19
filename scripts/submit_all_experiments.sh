#!/bin/bash
# Submit exp1 through exp25b as separate PBS GPU jobs on HPCE.
#
# Usage (from project root or any directory):
#   bash scripts/submit_all_experiments.sh
#   bash scripts/submit_all_experiments.sh --dry-run
#
# Prerequisites:
#   - Edit scripts/hpce_config.sh (conda env, walltime, queue, etc.)
#   - wandb login on the cluster (if logging to W&B)
#   - qsub available and access to gpuq

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpce_config.sh
source "${SCRIPT_DIR}/hpce_config.sh"

PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_DIR

PBS_SCRIPT="${SCRIPT_DIR}/pbs_train_job.sh"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# All experiments from exp1 to exp25b (order preserved for readability)
EXPERIMENTS=(
    exp1
    exp21a
    exp21b
    exp22a
    exp22b
    exp23
    exp24a
    exp24b
    exp25a
    exp25b
)

echo "Project:     ${PROJECT_DIR}"
echo "PBS script:  ${PBS_SCRIPT}"
echo "Queue:       ${PBS_QUEUE}"
echo "Walltime:    ${PBS_WALLTIME}"
echo "Resources:   ${PBS_SELECT}"
echo "Experiments: ${#EXPERIMENTS[@]}"
echo ""

SUBMITTED=()
FAILED=()

for exp in "${EXPERIMENTS[@]}"; do
    job_name="da6401-${exp}"
    stdout="${LOG_DIR}/${exp}.log"
    stderr="${LOG_DIR}/${exp}.err"

    qsub_cmd=(
        qsub
        -N "${job_name}"
        -q "${PBS_QUEUE}"
        -l "walltime=${PBS_WALLTIME}"
        -l "${PBS_SELECT}"
        -o "${stdout}"
        -e "${stderr}"
        -v "EXPERIMENT=${exp},PROJECT_DIR=${PROJECT_DIR}"
        "${PBS_SCRIPT}"
    )

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[dry-run] ${qsub_cmd[*]}"
        SUBMITTED+=("${exp}")
        continue
    fi

    if job_id="$("${qsub_cmd[@]}")"; then
        echo "Submitted ${exp} → ${job_id} (logs: ${stdout})"
        SUBMITTED+=("${exp}")
        echo "${job_id}" >> "${LOG_DIR}/submitted_jobs.txt"
    else
        echo "FAILED to submit ${exp}" >&2
        FAILED+=("${exp}")
    fi
done

echo ""
echo "Summary: ${#SUBMITTED[@]} submitted, ${#FAILED[@]} failed"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo "Failed: ${FAILED[*]}"
    exit 1
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
    echo "Track jobs: qstat -u \$USER"
    echo "Job IDs saved to: ${LOG_DIR}/submitted_jobs.txt"
fi
