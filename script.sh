#!/bin/bash
#SBATCH -c 10
#SBATCH --mem-per-cpu=1024
#SBATCH -G 4
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --job-name=bart_finetune
#SBATCH --output=smai_run.txt

# Activate the same environment
conda activate anlp_a2

# Optional: move to your project directory
# cd /path/to/SMAI-A3

# Install project requirements
pip install -r requirements.txt

# finetune_bart.py also needs these (if not already in requirements.txt)
pip install pandas scikit-learn datasets

# Run finetuning
python finetune_bart.py