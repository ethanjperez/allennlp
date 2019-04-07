#!/bin/bash
## SLURM scripts for running Dream Debate Jobs

## Job Metadata
#SBATCH --job-name=SL-Sweep
#SBATCH --output=/checkpoint/%u/debate/logs/SL-Sweep-%j.out
#SBATCH --error=/checkpoint/%u/debate/logs/SL-Sweep-%j.err

## Partition
#SBATCH --partition=priority
#SBATCH --comment="Meeting on Monday 4/8!"

## Number of Nodes (Number of Tasks to run)
#SBATCH --nodes=20

## Tasks per Node
#SBATCH --ntasks-per-node=1

## CPUS/Rollout Workers per Task
#SBATCH --cpus-per-task=8

## Time Limit - 1 Day
#SBATCH --time=720

## GPUS - One per Task
#SBATCH --gres=gpu:1

# Start clean
module purge

# Load what we need
module load anaconda3

### Section 3:
### Run your job.
srun --label /private/home/siddk/.conda/envs/allennlp/bin/python3.6 /private/home/siddk/allennlp/dream_experiments/slurm/dream_runner.py -m supervised -s 0