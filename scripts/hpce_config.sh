# HPCE cluster settings — edit before submitting jobs.
# Sourced by pbs_train_job.sh and submit_all_experiments.sh.

# Absolute path to the assignment repo (submit scripts override via PROJECT_DIR)
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# PBS queue and resources
PBS_QUEUE="${PBS_QUEUE:-gpuq}"
PBS_WALLTIME="${PBS_WALLTIME:-24:00:00}"
PBS_SELECT="${PBS_SELECT:-select=1:ncpus=4:ngpus=1:mem=32gb}"

# Python environment — uncomment/adapt for your cluster
# module load python/3.10
# module load cuda/11.8
CONDA_ENV="${CONDA_ENV:-}"          # e.g. da6401
VENV_PATH="${VENV_PATH:-}"          # e.g. $PROJECT_DIR/.venv

# Weights & Biases (optional; leave empty to use existing login)
export WANDB_PROJECT="${WANDB_PROJECT:-da6401-a3}"
# export WANDB_API_KEY="..."        # or run `wandb login` once on login node

# Use scratch staging (recommended on HPCE); set 0 to run in-place in PROJECT_DIR
USE_SCRATCH="${USE_SCRATCH:-1}"
