#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=16gb
#SBATCH --partition=visu
#SBATCH --output=constant_models

cd $(ws_find arnaud)
. setenv.sh
cd code/flow_pred/

module load devel/cuda/10.1

echo "Job ID: " ${SLURM_JOB_ID}


# MASK=" "          EPOCHS=40 CUDA_VISIBLE_DEVICES=0 ./train_models.sh --unet -c -n 32 -l 5 -p -np -m 2 &
# MASK=" "          EPOCHS=40 CUDA_VISIBLE_DEVICES=1 ./train_models.sh --unet -c -n 32 -l 5 -p -np -m 2 & 
# MASK=" "          EPOCHS=40 CUDA_VISIBLE_DEVICES=1 ./train_models.sh --unet -c -n 32 -l 5 -p -np -m 2 &
# MASK=" "          EPOCHS=40 CUDA_VISIBLE_DEVICES=1 ./train_models.sh --unet -c -n 32 -l 5 -p -np -m 2

# MASK="--no-mask"  EPOCHS=40 CUDA_VISIBLE_DEVICES=2 ./train_models.sh --unet -c -n 32 -l 5 -p -np -m 2 &
# MASK="--no-mask"  EPOCHS=40 CUDA_VISIBLE_DEVICES=3 ./train_models.sh --unet -c -n 32 -l 5 -p -np -m 2
# MASK="--no-mask"  EPOCHS=45 CUDA_VISIBLE_DEVICES=2 ./train_models.sh --unet -c -n 32 -l 5 -p -np -m 5 & 
# MASK="--no-mask"  EPOCHS=45 CUDA_VISIBLE_DEVICES=3 ./train_models.sh --unet -c -n 32 -l 5 -p -np -m 5

MASK=" "          EPOCHS=45 CUDA_VISIBLE_DEVICES=0 ./train_models.sh --unet -s -n 32 -l 5 -p -np -m 2 &
MASK=" "          EPOCHS=45 CUDA_VISIBLE_DEVICES=1 ./train_models.sh --unet -s -n 32 -l 5 -p -np -m 2 &
MASK=" "          EPOCHS=45 CUDA_VISIBLE_DEVICES=2 ./train_models.sh --unet -s -n 32 -l 5 -p -np -m 2 & 
MASK=" "          EPOCHS=45 CUDA_VISIBLE_DEVICES=3 ./train_models.sh --unet -s -n 32 -l 5 -p -np -m 2 
