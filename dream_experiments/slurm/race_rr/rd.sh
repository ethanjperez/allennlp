#!/bin/bash
## SLURM scripts for running Race Debate Jobs

## Job Metadata
#SBATCH --job-name=RACE-Middle-High-Round-Robin
#SBATCH --output=/checkpoint/%u/debate/logs/Race-RR-Dev-%j.out
#SBATCH --error=/checkpoint/%u/debate/logs/Race-RR-Dev-%j.err

## Partition
#SBATCH --partition=priority
#SBATCH --comment="Meeting 5/6 in prep for EMNLP Submission!"

## Number of Nodes (Number of Tasks to run)
#SBATCH --nodes=6
#SBATCH --constraint=volta32gb

## Tasks per Node
#SBATCH --ntasks-per-node=1

## CPUS/Rollout Workers per Task
#SBATCH --cpus-per-task=8

## Time Limit - 1 Day
#SBATCH --time=1440

## GPUS - One per Task
#SBATCH --gres=gpu:1

# Start clean
module purge

# Load what we need
module load anaconda3

# CD To allennlp directory
cd /private/home/siddk/allennlp

### Section 3:
### Run your job.
srun --label /private/home/siddk/.conda/envs/allennlp/bin/python3.6 /private/home/siddk/allennlp/dream_experiments/slurm/race_runner.py -m rd