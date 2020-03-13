#!/bin/bash
#
#SBATCH --job-name=ReprTrain
#SBATCH --partition=bigmem
#SBATCH --ntasks=1
#SBATCH --qos=bigmem
#SBATCH --cpus-per-task=2
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:pascal:2
#SBATCH --mail-type=All

srun python run.py -tr -rt
