#!/bin/bash
#SBATCH --job-name=npi-model
#SBATCH --time=30-00:00:00
#SBATCH --partition=unlimited
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=24
#SBATCH --mail-type=end
#SBATCH --mail-user=killeen@jhu.edu


module load python

python scripts/main.py --supercounties --cluster 1
