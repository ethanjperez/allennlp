#!/usr/bin/env bash

### AllenNLP Commands
# Training J only:
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode rr --serialization-dir tmp/rr.2

# Training A/B with fixed J
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.pt=rr.2 -j tmp/rr.2/model.tar.gz

# Training A/B/J with initialized J (provide model.tar.gz to -j and use -u)
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.pt=rr.2 -j tmp/rr.2/model.tar.gz -u

# Train A/B/J, A/R/J, B/R/J, or GT/B/J from scratch with F1 reward
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.2 -j training_config/bidaf.num_epochs=200.jsonnet -u
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode ar --serialization-dir tmp/ar.2 -j training_config/bidaf.num_epochs=200.jsonnet -u
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode br --serialization-dir tmp/br.2 -j training_config/bidaf.num_epochs=200.jsonnet -u
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode gB --serialization-dir tmp/gB.2 -u

# Evaluate A/B/J (add -e -r, no -u)
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode rr --serialization-dir tmp/ab.pt\=rr -j tmp/rr.2/model.tar.gz -r -e

# Evaluate J with Ground Truth vs. Oracle B
allennlp train training_config/bidaf.mini.debug.jsonnet --serialization-dir tmp/debug."$(uuid)" -e --debate_mode gB
allennlp train training_config/bidaf.num_epochs=200.jsonnet --serialization-dir tmp/rr.2 -e -r --debate_mode gB

### SLURM
# sbatch job
export SAVE_DIR=tmp/ab.v  # NB: Update every run!
if test -e $SAVE_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists. Make sure you wanted to load from an existing checkpoint.\n"; else mkdir -p $SAVE_DIR; fi
sbatch --job-name $SAVE_DIR --mem=20000 -t 2-23:58 --gres=gpu:p40 --open-mode append --requeue --wrap "\
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.v -j training_config/bidaf.num_epochs=200.jsonnet -u -v
"
echo -e "\n${CYAN}${SAVE_DIR}/train.log\n"

# Get a 24GB GPU
srun --pty --mem=20000 -t 2-23:58 --gres=gpu:p40 bash

# Get a dev GPU. Other GPUs: {1080ti,titanxp,titanblack,k40,k20,k20x,m2090}
srun --pty --mem=20000 -t 1-23:58 --gres=gpu:titanxp bash
srun --pty --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 bash

# Live updating dashboard of your jobs:
watch 'squeue -o "%.18i %.40j %.10u %.8T %.10M %.9l %.16b %.6C %.6D %R" -u $USER'

# Copy Prince tensorboard to local:
rsync -rav -e ssh --include '*/' --include 'events.out.tfevents.*' --exclude='*' ejp416@prince.hpc.nyu.edu:~/research/allennlp/tmp/ ~/research/allennlp/tmp

### Python
# iPDB: To run a list comprehension, use this before
globals().update(locals())
