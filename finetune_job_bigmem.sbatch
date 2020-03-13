#!/bin/bash
#
#SBATCH --job-name=SeqTrain_big
#SBATCH --partition=bigmem
#SBATCH --ntasks=1
#SBATCH --qos=bigmem
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:pascal:1
#SBATCH --mail-type=ALL

srun python run.py -tr -ft
