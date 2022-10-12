#!/bin/bash
#SBATCH --job-name=gihanJob
#SBATCH --ntasks=4
#SBATCH --mem=16gb
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00

#SBATCH --output=gihanOut.txt
#SBATCH --error=gihanError.txt

source ~/.bashrc
conda activate cmsc828c
python simulate-neurons.py


echo "Job finished at $(date)"
touch gihanComplete.txt