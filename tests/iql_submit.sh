#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --output=./logs/odql-48cpus_%j.log
#SBATCH --job-name=odql
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=medium
#SBATCH --open-mode=append
#SBATCH --clusters=all

# Load necessary applications
module load Anaconda3

# Load conda environment
source activate $HOME/.conda/envs/pytorch

wandb agent jia_wan/noisy-ZSC-tests/pirmkiut
