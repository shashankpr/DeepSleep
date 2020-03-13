#!/bin/bash
#
#SBATCH --job-name=SeqTrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:pascal:2

srun python run.py -tr -sq
