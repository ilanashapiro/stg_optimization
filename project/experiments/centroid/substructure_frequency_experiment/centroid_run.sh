#!/bin/bash
#SBATCH --job-name="centroid_z3_50s_ablation4"
#SBATCH --output=centroid_z3_out_50s_ablation4
#SBATCH --error=centroid_z3_err_50s_ablation4
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --mem=64G
#SBATCH --export=ALL
#SBATCH -t 7:00:00
#SBATCH -A csd887
#SBATCH --mail-user=ilshapiro@ucsd.edu
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate ishapiro

python3 /home/ishapiro/project/experiments/centroid/centroid_z3_repair.py

conda deactivate