#!/bin/bash
# Generic launcher: sbatch -J name run.sbatch script.py [args]
#SBATCH --account=bcse-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bcrv/asasli/later/dp2_transients/logs/%x_%j.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /projects/bcrv/asasli/later/dp2_transients
[ -f ~/.rsp_env ] && . ~/.rsp_env
exec /u/asasli/miniconda3/envs/applecider/bin/python -u "$@"
