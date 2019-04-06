#!/bin/bash
## SLURM scripts for running Dream Debate Jobs

## Job Metadata
#SBATCH --job-name=GPT-Oracle
#SBATCH --output=/checkpoint/%u/debate/logs/GPT-Judge-%j.out
#SBATCH --error=/checkpoint/%u/debate/logs/GPT-Judge-%j.err

## Partition
#SBATCH --partition=priority
#SBATCH --comment="Running for meeting Monday, 4/8"

## Number of Nodes (Number of Tasks to run)
#SBATCH --nodes=1

## Tasks per Node
#SBATCH --ntasks-per-node=1

## CPUS/Rollout Workers per Task
#SBATCH --cpus-per-task=32

## Time Limit - 1 Day
#SBATCH --time=1440

## GPUS - One per Task
#SBATCH --gres=gpu:1

# Start clean
module purge

# Load what we need
module load anaconda3

### Section 3:
### Run your job.
srun --label /private/home/siddk/.conda/envs/allennlp/bin/python3.6 /private/home/siddk/allennlp/dream_experiments/slurm/dream_runner.py -m gpt