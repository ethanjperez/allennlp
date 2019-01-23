#!/usr/bin/env bash

### AllenNLP Commands

# SQuAD+BERT/OpenAI: Training J on full passage (normal supervised training)
allennlp train training_config/bidaf_bert.jsonnet -s tmp/bert.f -d f
allennlp train training_config/bidaf_openai.jsonnet -s tmp/openai.f -d f

# RACE: Training J on full passage (normal supervised training)
allennlp train training_config/bidaf.race.size=half.jsonnet -s tmp/race.f -d f

# RACE: Debate baselines
allennlp train training_config/bidaf.race.size=half.jsonnet -s tmp/race.f -e -r -d BAA

# RACE: SL baselines
allennlp train training_config/bidaf.race.size=half.jsonnet -s tmp/race.a.m=sl.pt=race.f -j tmp/race.f/model.tar.gz -m sl -d a
allennlp train training_config/bidaf.race.size=half.jsonnet -s tmp/race.b.m=sl.pt=race.f -j tmp/race.f/model.tar.gz -m sl -d b
allennlp train training_config/bidaf.race.size=half.jsonnet -s tmp/race.a.m=sl-ssp.pt=race.f -j tmp/race.f/model.tar.gz -m sl-ssp -d a
allennlp train training_config/bidaf.race.size=half.jsonnet -s tmp/race.b.m=sl-spp.pt=race.f -j tmp/race.f/model.tar.gz -m sl-ssp -d b

# RACE: RL: EM Reward (OOM on Titan XP 43% through 1 epoch with br)
allennlp train training_config/bidaf.race.size=half.patience=None.jsonnet -s tmp/race.a.m=em.pt=race.f -j tmp/race.f/model.tar.gz -m em -d a
allennlp train training_config/bidaf.race.size=half.patience=None.jsonnet -s tmp/race.b.m=em.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m em -d b
allennlp train training_config/bidaf.race.size=half.patience=None.jsonnet -s tmp/race.ab.m=em.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m em -d ab
allennlp train training_config/bidaf.race.size=half.patience=None.jsonnet -s tmp/race.ar.m=em.pt=race.f -j tmp/race.f/model.tar.gz -m em -d ar
allennlp train training_config/bidaf.race.size=half.patience=None.jsonnet -s tmp/race.br.m=em.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m em -d br

# RACE: RL: SSP Reward
allennlp train training_config/bidaf.race.size=half.patience=None.jsonnet -s tmp/race.a.m=ssp.pt=race.f -j tmp/race.f/model.tar.gz -m ssp -d a
allennlp train training_config/bidaf.race.size=half.patience=None.jsonnet -s tmp/race.b.m=ssp.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m ssp -d b
allennlp train training_config/bidaf.race.size=half.patience=None.jsonnet -s tmp/race.ab.m=ssp.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m ssp -d ab
allennlp train training_config/bidaf.race.size=half.patience=None.jsonnet -s tmp/race.ar.m=ssp.pt=race.f -j tmp/race.f/model.tar.gz -m ssp -d ar
allennlp train training_config/bidaf.race.size=half.patience=None.jsonnet -s tmp/race.br.m=ssp.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m ssp -d br

# Training J only:
allennlp train training_config/bidaf.num_epochs=200.jsonnet -d rr -s tmp/rr.2
allennlp train training_config/bidaf.squad_xl.num_epochs=200.jsonnet -d f -s tmp/squad_xl.f
allennlp train training_config/bidaf.squad_xl.num_epochs=200.jsonnet -d rr -s tmp/squad_xl.rr

# Training ab with fixed J
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -d ab -s tmp/ab.3.pt=rr.3 -j tmp/rr.3/model.tar.gz
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -d ab -s tmp/ab.3.pt=gr.2 -j tmp/gr.2/model.tar.gz

# Training bg with fixed J
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -d bg -s tmp/bg.3.rb=1-ra.pt=rr.3 -j tmp/rr.3/model.tar.gz
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -d bg -s tmp/bg.3.m=ssp.rb=1-ra.pt=rr.3 -j tmp/rr.3/model.tar.gz -m ssp

# Training abj with initialized J (provide model.tar.gz to -j and use -u)
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -d ab -s tmp/ab.3.pt=rr.2.u -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -d ab -s tmp/ab.3.pt=gr.2.u -j tmp/gr.2/model.tar.gz -u

# Train abj\arj\brj from scratch with F1 reward
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -d ab -s tmp/ab.3.j.dropout=0.4 -j training_config/bidaf.dropout=0.4.jsonnet -u
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -d ar -s tmp/ar.3.j.dropout=0.4 -j training_config/bidaf.dropout=0.4.jsonnet -u
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -d br -s tmp/br.3.j.dropout=0.4 -j training_config/bidaf.dropout=0.4.jsonnet -u

# Train j on gB
allennlp train training_config/bidaf.jsonnet -d gB -s tmp/gB.2 -u
allennlp train training_config/bidaf.dropout=0.4.jsonnet -d gB -s tmp/gB.j.dropout=0.4.u -u
# Train j on gB with pre-trained initialization (and maybe frozen layers)
allennlp train training_config/bidaf.num_epochs=200.jsonnet -d gB -s tmp/gB.j.pt=rr.2.u -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.num_epochs=200.j.no_grad=cw.jsonnet -d gB -s tmp/gB.j.no_grad=cw.u -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.num_epochs=200.j.no_grad=cwp.jsonnet -d gB -s tmp/gB.j.no_grad=cwp.u -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.num_epochs=200.j.no_grad=cwpa.jsonnet -d gB -s tmp/gB.j.no_grad=cwpa.u -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.num_epochs=200.j.no_grad=cwpam.jsonnet -d gB -s tmp/gB.j.no_grad=cwpam.u -j tmp/rr.2/model.tar.gz -u

# Train b on gb with SL on oracle
allennlp train training_config/bidaf.cpu.mini.debug.jsonnet -s tmp/debug -j training_config/bidaf.cpu.mini.debug.jsonnet -u -d gb -m sl
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/gb.m=sl-ssp.dropout=0.2 -j tmp/rr.3/model.tar.gz -d gb -m sl-ssp

# Evaluate abj (add -e -r no -u)
allennlp train training_config/bidaf.num_epochs=200.jsonnet -d rr -s tmp/ab.pt\=rr -j tmp/rr.2/model.tar.gz -r -e

# Evaluate arj
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -d ar -s tmp/ab.3.pt\=rr.2.u -j training_config/bidaf.num_epochs=200.jsonnet -r -e
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -d ar -s tmp/ab.3.pt\=rr.2.u -j tmp/rr.3/model.tar.gz -r -e

# Evaluate j with gB
allennlp train training_config/bidaf.cpu.mini.debug.jsonnet -s tmp/debug -e -d gB
allennlp train training_config/bidaf.num_epochs=200.jsonnet -s tmp/rr.2 -e -r -d gB
allennlp train training_config/bidaf.num_epochs=200.jsonnet -s tmp/rr.2 -e -r -d gB -o "{'test_data_path': 'datasets/squad/squad-adversarial-add-one-sent.json'}"
allennlp train training_config/bidaf.num_epochs=200.jsonnet -s tmp/rr.2 -e -r -d gB -o "{'test_data_path': 'datasets/squad/squad-adversarial-add-sent.json'}"

# Evaluate j with gb (SL-trained)
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/gb.m=sl.dropout=0.2.backup -j tmp/rr.3/model.tar.gz -d gb -m sl -e -r

# Debug
allennlp train training_config/bidaf.cpu.mini.debug.jsonnet -s tmp/debug -j training_config/bidaf.cpu.mini.debug.jsonnet -u -d ab

### SLURM
# sbatch job
# NB: Update SERIALIZATION_DIR every run!
export SERIALIZATION_DIR=tmp/race.j.pt=f.size=half
if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists. Make sure you wanted to load from an existing checkpoint.\n"; else mkdir -p $SERIALIZATION_DIR; fi
sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:titanblack --open-mode append --requeue --wrap "\
allennlp train training_config/bidaf.race.size=half.jsonnet -s tmp/race.j.pt=f.size=half -d f
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
# iPDB: To run a list comprehension\nuse this before
globals().update(locals())

allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -d gB -s tmp/br.3.j.dropout=0.4 -j training_config/bidaf.num_epochs=200.jsonnet -r -e -o "{'test_data_path': 'datasets/squad/squad-adversarial-add-sent.json'}"
