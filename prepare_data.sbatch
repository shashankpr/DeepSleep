#!/bin/bash
#
#SBATCH --job-name=PrepData
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=16GB

srun python run.py -pp
