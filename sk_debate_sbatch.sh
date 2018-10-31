#!/usr/bin/env bash

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=bidaf_debate_train_200
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/bidaf-debate-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/checkpoint/%u/jobs/bidaf-debate-%j.err

## partition name
#SBATCH --partition=uninterrupted
## number of nodes
#SBATCH --nodes=1

## number of tasks per node
#SBATCH --ntasks-per-node=1


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
### Run your job. Note that we are not passing any additional
### arguments to srun since we have already specificed the job
### configuration with SBATCH directives
salloc --label -C  allennlp