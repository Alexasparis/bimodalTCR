#!/bin/bash
#SBATCH --job-name="SHAP"
#SBATCH --output="./logs/out/shap_%j.out"
#SBATCH --error="./logs/err/shap_%j.err"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --time=02:00:00
#SBATCH --qos=gp_bscls
#SBATCH -A bsc72
#SBATCH --constraint=highmem

# Load necessary modules
module purge
module load gcc intel openmpi/4.1.5
module load miniforge
source activate esm3

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

embeddings="$1"
dataframe="$2"
model="$3"
type_mod="$4"
out="$5"

echo "Start at $(date)"
echo "---------------------"

# Run Python script
python3 predict_test.py -emb "$embeddings" -df "$dataframe" -model "$model" -type "$type_mod" -out "$out"

echo "End at $(date)"
echo "---------------------"


