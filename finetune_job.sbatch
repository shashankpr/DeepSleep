#!/bin/bash
#
#SBATCH --job-name=SeqTrain
#SBATCH --ntasks=1
#SBATCH --qos=long
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:pascal:1
#SBATCH --mail-type=ALL

srun python run.py -tr -ft