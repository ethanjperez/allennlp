#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.debate.sl.lr=1e-5.jsonnet -s tmp/race.m=sl-sents.best.e -j tmp/race.best.f/model.tar.gz -e -r -b 1 -d ⅰ -m sl-sents -a 12 -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.m=sl-sents.best.e/d=ⅰ.c=concat.test.txt"
"allennlp train training_config/race.best.debate.sl.lr=1e-5.jsonnet -s tmp/race.m=sl-sents.best.e -j tmp/race.best.f/model.tar.gz -e -r -b 1 -d ⅱ -m sl-sents -a 12 -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.m=sl-sents.best.e/d=ⅱ.c=concat.test.txt"
"allennlp train training_config/race.best.debate.sl.lr=1e-5.jsonnet -s tmp/race.m=sl-sents.best.e -j tmp/race.best.f/model.tar.gz -e -r -b 1 -d ⅲ -m sl-sents -a 12 -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.m=sl-sents.best.e/d=ⅲ.c=concat.test.txt"
"allennlp train training_config/race.best.debate.sl.lr=1e-5.jsonnet -s tmp/race.m=sl-sents.best.e -j tmp/race.best.f/model.tar.gz -e -r -b 1 -d ⅳ -m sl-sents -a 12 -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.m=sl-sents.best.e/d=ⅳ.c=concat.test.txt"
"allennlp train training_config/race.best.debate.sl.lr=1e-5.jsonnet -s tmp/race.m=sl-sents.i.best.e -j tmp/race.best.f/model.tar.gz -e -r -b 1 -d ⅰ -m sl-sents -i -a 12 -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.m=sl-sents.i.best.e/d=ⅰ.c=concat.test.txt"
"allennlp train training_config/race.best.debate.sl.lr=1e-5.jsonnet -s tmp/race.m=sl-sents.i.best.e -j tmp/race.best.f/model.tar.gz -e -r -b 1 -d ⅱ -m sl-sents -i -a 12 -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.m=sl-sents.i.best.e/d=ⅱ.c=concat.test.txt"
"allennlp train training_config/race.best.debate.sl.lr=1e-5.jsonnet -s tmp/race.m=sl-sents.i.best.e -j tmp/race.best.f/model.tar.gz -e -r -b 1 -d ⅲ -m sl-sents -i -a 12 -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.m=sl-sents.i.best.e/d=ⅲ.c=concat.test.txt"
"allennlp train training_config/race.best.debate.sl.lr=1e-5.jsonnet -s tmp/race.m=sl-sents.i.best.e -j tmp/race.best.f/model.tar.gz -e -r -b 1 -d ⅳ -m sl-sents -i -a 12 -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.m=sl-sents.i.best.e/d=ⅳ.c=concat.test.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done
