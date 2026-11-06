#!/bin/bash

# Slurm job for the NUS SoC cluster, adapted from job_template.sbatch in
# https://github.com/isaacNingLee/Unofficial_SoC_Cluster_Guide.
#
# The first argument is the pipeline script; the rest are passed on to it.
# Submit from the repository root, naming the job after the run:
#   sbatch -J bart-base.s1 cluster/job.sbatch cluster/train.sh training/configs/bart-base.stage1.json
#
# --time is a starting value: check the partition limit with `sinfo -o "%P %l"`
# and override it per job (sbatch --time=...). Training resumes from its last
# checkpoint, so a job cut off by the time limit can simply be submitted again.

#SBATCH --job-name=gec
#SBATCH --partition=gpu-long
#SBATCH --time=12:00:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-gec-train}"

# Checkpoints and the Hugging Face cache live on scratch (500 GB), not in home.
SCRATCH="${SCRATCH:-/mnt/scratch/${USER:0:1}/$USER}"
export CKPT_ROOT="${CKPT_ROOT:-$SCRATCH/cs4248-gec/checkpoints}"
export HF_HOME="${HF_HOME:-$SCRATCH/cs4248-gec/hf_cache}"
# Models are downloaded beforehand on xlogin (cluster/download_models.sh).
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# Slurm runs a spooled copy of this script; SLURM_SUBMIT_DIR is the repository root.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR" "$CKPT_ROOT"
exec > "$LOG_DIR/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-$$}.out" 2>&1
cd "$REPO_ROOT"

# Non-interactive jobs do not source ~/.bashrc; some environments touch unset variables.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date), commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

PIPELINE_SCRIPT="$1"
shift
bash "$PIPELINE_SCRIPT" "$@"
echo "finished at $(date)"
