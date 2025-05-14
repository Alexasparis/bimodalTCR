#!/bin/bash
#SBATCH --job-name="seq_train"
#SBATCH --output="./logs/out/train_%j.out"
#SBATCH --error="./logs/err/train_%j.err"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --time=15:00:00
#SBATCH --qos=gp_bscls
#SBATCH -A bsc72
#SBATCH --constraint=highmem

# Load necessary modules
module purge
module load gcc intel openmpi/4.1.5
module load miniforge
source activate esm3

# Get input arguments
dataframe=$1
embeddings=$2
type_emb=$3

# Check if all required arguments are provided
if [ -z "$dataframe" ] || [ -z "$embeddings" ] || [ -z "$type_emb" ]; then
  echo "Error: Missing one or more arguments."
  echo "Usage: sbatch your_script.sh <input_file> <model_file> <output_file> <type_emb>"
  exit 1
fi

# Export SLURM environment variable
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

echo "Start at $(date)"
echo "---------------------"

# Run Python script
python3 train_xgb.py -df "$dataframe" -emb "$embeddings" -t "$type_emb"

echo "End at $(date)"
echo "---------------------"

conda deactivate

