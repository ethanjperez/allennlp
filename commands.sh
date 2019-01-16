#!/usr/bin/env bash

### AllenNLP Commands

# RACE: Training J on full passage (normal supervised training)
allennlp train training_config/bidaf.race.size=half.jsonnet --serialization-dir tmp/race.ans_beg.j.pt=f.size=half --debate_mode f

# Training J only:
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode rr --serialization-dir tmp/rr.2

# Training ab with fixed J
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.3.pt=rr.3 -j tmp/rr.3/model.tar.gz
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.3.pt=gr.2 -j tmp/gr.2/model.tar.gz

# Training bg with fixed J
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode bg --serialization-dir tmp/bg.3.pt=rr.3 -j tmp/rr.3/model.tar.gz

# Training abj with initialized J (provide model.tar.gz to -j and use -u)
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.3.pt=rr.2.u -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.3.pt=gr.2.u -j tmp/gr.2/model.tar.gz -u

# Train abj, arj, brj from scratch with F1 reward
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.3.j.dropout=0.4 -j training_config/bidaf.dropout=0.4.jsonnet -u
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ar --serialization-dir tmp/ar.3.j.dropout=0.4 -j training_config/bidaf.dropout=0.4.jsonnet -u
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode br --serialization-dir tmp/br.3.j.dropout=0.4 -j training_config/bidaf.dropout=0.4.jsonnet -u

# Train j on gB
allennlp train training_config/bidaf.jsonnet --debate_mode gB --serialization-dir tmp/gB.2 -u
allennlp train training_config/bidaf.dropout=0.4.jsonnet --debate_mode gB --serialization-dir tmp/gB.j.dropout=0.4.u -u
# Train j on gB with pre-trained initialization (and maybe frozen layers)
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode gB --serialization-dir tmp/gB.j.pt=rr.2.u -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.num_epochs=200.j.no_grad=cw.jsonnet --debate_mode gB --serialization-dir tmp/gB.j.no_grad=cw.u -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.num_epochs=200.j.no_grad=cwp.jsonnet --debate_mode gB --serialization-dir tmp/gB.j.no_grad=cwp.u -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.num_epochs=200.j.no_grad=cwpa.jsonnet --debate_mode gB --serialization-dir tmp/gB.j.no_grad=cwpa.u -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.num_epochs=200.j.no_grad=cwpam.jsonnet --debate_mode gB --serialization-dir tmp/gB.j.no_grad=cwpam.u -j tmp/rr.2/model.tar.gz -u

# Train b on gb with SL on oracle
allennlp train training_config/bidaf.cpu.mini.debug.jsonnet --serialization-dir tmp/debug -j training_config/bidaf.cpu.mini.debug.jsonnet -u --debate_mode gb -m sl
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --serialization-dir tmp/gb.m=sl.dropout=0.2 -j tmp/rr.3/model.tar.gz --debate_mode gb -m sl

# Evaluate abj (add -e -r, no -u)
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode rr --serialization-dir tmp/ab.pt\=rr -j tmp/rr.2/model.tar.gz -r -e

# Evaluate arj
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ar --serialization-dir tmp/ab.3.pt\=rr.2.u -j training_config/bidaf.num_epochs=200.jsonnet -r -e
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ar --serialization-dir tmp/ab.3.pt\=rr.2.u -j tmp/rr.3/model.tar.gz -r -e

# Evaluate j with gB
allennlp train training_config/bidaf.cpu.mini.debug.jsonnet --serialization-dir tmp/debug -e --debate_mode gB
allennlp train training_config/bidaf.num_epochs=200.jsonnet --serialization-dir tmp/rr.2 -e -r --debate_mode gB
allennlp train training_config/bidaf.num_epochs=200.jsonnet --serialization-dir tmp/rr.2 -e -r --debate_mode gB -o "{'test_data_path': 'datasets/squad/squad-adversarial-add-one-sent.json'}"
allennlp train training_config/bidaf.num_epochs=200.jsonnet --serialization-dir tmp/rr.2 -e -r --debate_mode gB -o "{'test_data_path': 'datasets/squad/squad-adversarial-add-sent.json'}"

# Evaluate j with gb (SL-trained)
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --serialization-dir tmp/gb.m=sl.dropout=0.2.backup -j tmp/rr.3/model.tar.gz --debate_mode gb -m sl -e -r

# Debug
allennlp train training_config/bidaf.cpu.mini.debug.jsonnet --serialization-dir tmp/debug -j training_config/bidaf.cpu.mini.debug.jsonnet -u --debate_mode ab

### SLURM
# sbatch job
# NB: Update SERIALIZATION_DIR every run!
export SERIALIZATION_DIR=tmp/race.j.pt=f.size=half
if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists. Make sure you wanted to load from an existing checkpoint.\n"; else mkdir -p $SERIALIZATION_DIR; fi
sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:titanblack --open-mode append --requeue --wrap "\
allennlp train training_config/bidaf.race.size=half.jsonnet --serialization-dir tmp/race.j.pt=f.size=half --debate_mode f
"
echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"

# Get a 24GB GPU
srun --pty --mem=20000 -t 6-23:58 --gres=gpu:p40 bash

# Get a dev GPU. Other GPUs: {1080ti,titanxp,titanblack,k40,k20,k20x,m2090}
srun --pty --mem=20000 -t 1-23:58 --gres=gpu:titanxp bash
srun --pty --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 bash

# Live updating dashboard of your jobs:
watch 'squeue -o "%.18i %.40j %.10u %.8T %.10M %.9l %.16b %.6C %.6D %R" -u $USER'

# Copy Prince tensorboard to local:
rsync -rav -e ssh --include '*/' --include 'events.out.tfevents.*' --exclude='*' ejp416@prince.hpc.nyu.edu:~/research/allennlp/tmp/ ~/research/allennlp/tmp
rsync -rav -e ssh --include '*/' --include 'events.out.tfevents.*' --exclude='*' ejp416@access.cims.nyu.edu:~/research/allennlp/tmp/ ~/research/allennlp/tmp

### Python
# iPDB: To run a list comprehension, use this before
globals().update(locals())

allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode gB --serialization-dir tmp/br.3.j.dropout=0.4 -j training_config/bidaf.num_epochs=200.jsonnet -r -e -o "{'test_data_path': 'datasets/squad/squad-adversarial-add-sent.json'}"
