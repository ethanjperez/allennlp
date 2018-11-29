#!/usr/bin/env bash

## SLURM scripts have a specific format.

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=priority-bidaf-debate-dropout=0.5
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/allennlp_jobs/abj-dropout=0.5-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/checkpoint/%u/allennlp_jobs/abj-dropout=0.5-%j.err

## partition name
#SBATCH --partition=priority
## number of nodes
#SBATCH --nodes=1

## number of tasks per node
#SBATCH --ntasks-per-node=1

## number of CPUs
#SBATCH --cpus-per-task=16

## time limit
#SBATCH --time=1440

## GPUS
#SBATCH --gres=gpu:volta32gb:1

## Working directory
#SBATCH --workdir="/private/home/siddk/allennlp"


### Section 2: Setting environment variables for the job
### Remember that all the module command does is set environment
### variables for the software you need to. Here I am assuming I
### going to run something with python.
### You can also set additional environment variables here and
### SLURM will capture all of them for each task

# Start clean
module purge

# Load what we need
module load anaconda3

### Section 3:
### Run your job.
srun /private/home/siddk/.conda/envs/allennlp/bin/allennlp train /private/home/siddk/allennlp/training_config/bidaf.jsonnet -j /private/home/siddk/allennlp/training_config/bidaf.dropout=0.5.jsonnet -u --serialization-dir /checkpoint/siddk/bidaf-debate/abj_debate.dropout=0.5