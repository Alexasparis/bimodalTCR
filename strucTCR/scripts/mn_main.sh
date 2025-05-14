#!/bin/bash
#SBATCH --job-name="StrucTCR"
#SBATCH --output="./logs/structcr_%j.out"
#SBATCH --error="./logs/structcr_%j.err"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --time=02:00:00
#SBATCH --qos=gp_bscls
#SBATCH -A bsc72

# Load all the necessary modules for run the task
module load miniforge
source activate anarci
source ~/env/bin/activate

# Args
input=$1
model=$2
output=$3   

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

echo "Start $(date)"
echo "---------------"
echo "Starting job with the following parameters:"
echo "Input: $input"
echo "Model: $model"
echo "Output: $output"

python main.py -i "$input" -tcrp "$model" -tcrm ../models/TCR_MHC.csv -o "$output" -w "$SRUN_CPUS_PER_TASK" -v

echo "End $(date)"
echo "--------------"
