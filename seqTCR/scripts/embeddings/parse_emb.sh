#!/bin/bash
#SBATCH --job-name="ESM_cdrs"
#SBATCH --output="./logs/out/cdrs_esm3_emb_%j.out"
#SBATCH --error="./logs/err/cdrs_esm3_emb_%j.err"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=04:00:00
#SBATCH --qos=gp_bscls
#SBATCH -A bsc72

# Load all the necessary modules for run the tas
module purge
module load gcc intel openmpi/4.1.5
module load miniforge
source activate anarci

# Get input and output file paths from command line arguments
input_file=$1
embed_file=$2
output_file=$3

# Check if all three arguments are provided
if [ -z "$input_file" ] || [ -z "$embed_file" ] || [ -z "$output_file" ]; then
  echo "Error: Missing one or more arguments."
  echo "Usage: sbatch your_script.sh <input_file> <embed_file> <output_file>"
  exit 1
fi

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

echo "Start at $(date)"
echo "---------------------"
python3 parse_emb.py -df "$input_file" -emb "$embed_file" -o "$output_file"


echo "End at $(date)"
echo "---------------------"

conda deactivate
