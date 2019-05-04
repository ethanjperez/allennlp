#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_12.best.f -e -r -d Ⅰ -c concat -p tmp/race.num_sents_leq_12.best.f/oracle_outputs.c=concat.d=Ⅰ.train.6.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.6'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f/d=Ⅰ.c=concat.train.6.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_12.best.f -e -r -d Ⅰ -c concat -p tmp/race.num_sents_leq_12.best.f/oracle_outputs.c=concat.d=Ⅰ.train.7.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.7'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f/d=Ⅰ.c=concat.train.7.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_12.best.f -e -r -d Ⅰ -c concat -p tmp/race.num_sents_leq_12.best.f/oracle_outputs.c=concat.d=Ⅰ.train.8.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.8'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f/d=Ⅰ.c=concat.train.8.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_12.best.f -e -r -d Ⅰ -c concat -p tmp/race.num_sents_leq_12.best.f/oracle_outputs.c=concat.d=Ⅰ.train.9.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.9'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f/d=Ⅰ.c=concat.train.9.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_12.best.f -e -r -d Ⅰ -c concat -p tmp/race.num_sents_leq_12.best.f/oracle_outputs.c=concat.d=Ⅰ.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f/d=Ⅰ.c=concat.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_12.best.f -e -r -d Ⅰ -c concat -p tmp/race.num_sents_leq_12.best.f/oracle_outputs.c=concat.d=Ⅰ.test.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f/d=Ⅰ.c=concat.test.txt"

)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done

