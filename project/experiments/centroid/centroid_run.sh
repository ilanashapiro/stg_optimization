#!/bin/bash
#SBATCH --job-name="centroid_z3_50s_ONEPARENT"
#SBATCH --output=centroid_z3_out_50s_ONEPARENT
#SBATCH --error=centroid_z3_error_50s_ONEPARENT
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=108
#SBATCH --mem=64G
#SBATCH --export=ALL
#SBATCH -t 13:00:00
#SBATCH -A csd887
#SBATCH --mail-user=ilshapiro@ucsd.edu
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate ishapiro

python3 /home/ishapiro/project/experiments/centroid/centroid_z3_repair.py

conda deactivate