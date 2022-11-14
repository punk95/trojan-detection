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
conda activate keras
python create-trojan.py --epochs 50 --trojan True


echo "Job finished at $(date)"
touch gihanComplete.txt