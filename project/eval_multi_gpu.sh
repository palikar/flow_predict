#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=16gb
#SBATCH --partition=visu
#SBATCH --output=speed_model

cd $(ws_find arnaud)
. setenv.sh
cd code/flow_pred/

module load devel/cuda/10.1

echo "Job ID: " ${SLURM_JOB_ID}

CUDA_VISIBLE_DEVICES=0 ./eval_models.sh results_c/* &

CUDA_VISIBLE_DEVICES=1 ./eval_models.sh results_s/* & 

CUDA_VISIBLE_DEVICES=2 ./eval_models.sh results_vd/*
