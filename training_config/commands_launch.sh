#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅰ -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=Ⅰ.test.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.best.f/d=Ⅰ.c=concat.test.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅱ -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=Ⅱ.test.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.best.f/d=Ⅱ.c=concat.test.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅲ -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=Ⅲ.test.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.best.f/d=Ⅲ.c=concat.test.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅳ -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=Ⅳ.test.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.best.f/d=Ⅳ.c=concat.test.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done
