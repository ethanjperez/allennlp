#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d ⅱ -c concat -m sl-random -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}\" 2>&1 | tee tmp/race.best.f/d=r.c=concat.dev.ⅠⅡ.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅰ ⅱ -c concat -m sl-random -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}\" 2>&1 | tee tmp/race.best.f/d=Ⅰ_r.c=concat.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅰ Ⅱ ⅰ -c concat -m sl-random -p tmp/race.best.f/oracle_outputs.c=concat.d=2_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}\" 2>&1 | tee tmp/race.best.f/d=Ⅰ_Ⅱ_r.c=concat.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅰ Ⅱ Ⅰ ⅱ -c concat -m sl-random -p tmp/race.best.f/oracle_outputs.c=concat.d=3_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}\" 2>&1 | tee tmp/race.best.f/d=Ⅰ_Ⅱ_Ⅰ_r.c=concat.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅰ Ⅱ Ⅰ Ⅱ ⅰ -c concat -m sl-random -p tmp/race.best.f/oracle_outputs.c=concat.d=4_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}\" 2>&1 | tee tmp/race.best.f/d=Ⅰ_Ⅱ_Ⅰ_Ⅱ_r.c=concat.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅰ Ⅱ Ⅰ Ⅱ Ⅰ ⅰ -c concat -m sl-random -p tmp/race.best.f/oracle_outputs.c=concat.d=5_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}\" 2>&1 | tee tmp/race.best.f/d=Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ_r.c=concat.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅱ ⅰ -c concat -m sl-random -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}\" 2>&1 | tee tmp/race.best.f/d=Ⅱ_r.c=concat.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅱ Ⅰ ⅱ -c concat -m sl-random -p tmp/race.best.f/oracle_outputs.c=concat.d=2_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}\" 2>&1 | tee tmp/race.best.f/d=Ⅱ_Ⅰ_r.c=concat.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅱ Ⅰ Ⅱ ⅰ -c concat -m sl-random -p tmp/race.best.f/oracle_outputs.c=concat.d=3_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}\" 2>&1 | tee tmp/race.best.f/d=Ⅱ_Ⅰ_Ⅱ_r.c=concat.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅱ Ⅰ Ⅱ Ⅰ ⅱ -c concat -m sl-random -p tmp/race.best.f/oracle_outputs.c=concat.d=4_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}\" 2>&1 | tee tmp/race.best.f/d=Ⅱ_Ⅰ_Ⅱ_Ⅰ_r.c=concat.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d Ⅱ Ⅰ Ⅱ Ⅰ Ⅱ ⅰ -c concat -m sl-random -p tmp/race.best.f/oracle_outputs.c=concat.d=5_ⅠⅡ_turns.all.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}\" 2>&1 | tee tmp/race.best.f/d=Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ_r.c=concat.dev.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=30000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done
