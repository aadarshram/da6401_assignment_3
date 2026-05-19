# HPCE batch submission (PBS)

Launch all DA6401 experiments (`exp1` … `exp25b`) as **separate GPU jobs** on the HPCE cluster.

## Setup

1. Edit `scripts/hpce_config.sh`:
   - Set `CONDA_ENV` or `VENV_PATH` for your Python environment
   - Adjust `PBS_QUEUE`, `PBS_WALLTIME`, `PBS_SELECT` if needed
   - Optionally set `WANDB_API_KEY` or run `wandb login` on the cluster

2. Install dependencies in that environment (`pip install -r requirements.txt`, `python -m spacy download en_core_web_sm de_core_news_sm`).

## Submit all jobs

From the project root on the **login node**:

```bash
bash scripts/submit_all_experiments.sh
```

Preview commands without submitting:

```bash
bash scripts/submit_all_experiments.sh --dry-run
```

## Submit one experiment

```bash
qsub -N da6401-exp1 -q gpuq -l walltime=24:00:00 -l select=1:ncpus=4:ngpus=1 \
  -v EXPERIMENT=exp1,PROJECT_DIR=$PWD \
  -o logs/exp1.log -e logs/exp1.err \
  scripts/pbs_train_job.sh
```

## Outputs

| Path | Description |
|------|-------------|
| `logs/<exp>.log` | stdout |
| `logs/<exp>.err` | stderr |
| `logs/submitted_jobs.txt` | PBS job IDs |
| `best_checkpoint_<exp>.pt` | Best model (+ `train_config` inside) |

Monitor: `qstat -u $USER`

Jobs use `$HOME/scratch/job<PBS_JOBID>` for staging (see `USE_SCRATCH` in `hpce_config.sh`), matching the HPCE CUDA/GROMACS workflow.
