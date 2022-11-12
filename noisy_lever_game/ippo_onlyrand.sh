#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --output=./logs/iql_%j.log
#SBATCH --job-name=ippo
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=short
#SBATCH --open-mode=append
#SBATCH --clusters=all

# Load necessary applications
module load Anaconda3

# Load conda environment
source activate $HOME/.conda/envs/pytorch

wandb agent jia_wan/noisy-ZSC-tests/o1yoehuw