#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.ⅱ.m=sl-sents.i.lr=2e-5.bsz=12.n=1.x=0.5.c=concat -j tmp/race.best.f/model.tar.gz -e -r -b 1 -d ⅰ -m sl -a 12 -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=ⅰ.dev.num_passages=13.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev.num_passages=13'}\" 2>&1 | tee tmp/race.ⅱ.m=sl-sents.i.lr=2e-5.bsz=12.n=1.x=0.5.c=concat/d=ⅰ.c=concat.dev.num_passages=13.txt"
"allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.ⅱ.m=sl-sents.i.lr=2e-5.bsz=12.n=1.x=0.5.c=concat -j tmp/race.best.f/model.tar.gz -e -r -b 1 -d ⅱ -m sl -a 12 -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=ⅱ.dev.num_passages=13.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev.num_passages=13'}\" 2>&1 | tee tmp/race.ⅱ.m=sl-sents.i.lr=2e-5.bsz=12.n=1.x=0.5.c=concat/d=ⅱ.c=concat.dev.num_passages=13.txt"
"allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.ⅱ.m=sl-sents.i.lr=2e-5.bsz=12.n=1.x=0.5.c=concat -j tmp/race.best.f/model.tar.gz -e -r -b 1 -d ⅲ -m sl -a 12 -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=ⅲ.dev.num_passages=13.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev.num_passages=13'}\" 2>&1 | tee tmp/race.ⅱ.m=sl-sents.i.lr=2e-5.bsz=12.n=1.x=0.5.c=concat/d=ⅲ.c=concat.dev.num_passages=13.txt"
"allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.ⅱ.m=sl-sents.i.lr=2e-5.bsz=12.n=1.x=0.5.c=concat -j tmp/race.best.f/model.tar.gz -e -r -b 1 -d ⅳ -m sl -a 12 -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=ⅳ.dev.num_passages=13.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev.num_passages=13'}\" 2>&1 | tee tmp/race.ⅱ.m=sl-sents.i.lr=2e-5.bsz=12.n=1.x=0.5.c=concat/d=ⅳ.c=concat.dev.num_passages=13.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=30000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done
