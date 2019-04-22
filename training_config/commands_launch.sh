#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f.eval.test.human_eval.Ⅰ -j tmp/race.best.f/model.tar.gz -e -f -d Ⅰ -c concat -p tmp/race.best.f.eval.test.human_eval.Ⅰ/oracle_outputs.c=concat.d=Ⅰ.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.best.f.eval.test.human_eval.Ⅰ/oracle_outputs.c=concat.d=Ⅰ.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f.eval.test.human_eval.Ⅱ -j tmp/race.best.f/model.tar.gz -e -f -d Ⅱ -c concat -p tmp/race.best.f.eval.test.human_eval.Ⅱ/oracle_outputs.c=concat.d=Ⅱ.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.best.f.eval.test.human_eval.Ⅱ/oracle_outputs.c=concat.d=Ⅱ.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f.eval.test.human_eval.Ⅲ -j tmp/race.best.f/model.tar.gz -e -f -d Ⅲ -c concat -p tmp/race.best.f.eval.test.human_eval.Ⅲ/oracle_outputs.c=concat.d=Ⅲ.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.best.f.eval.test.human_eval.Ⅲ/oracle_outputs.c=concat.d=Ⅲ.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f.eval.test.human_eval.Ⅳ -j tmp/race.best.f/model.tar.gz -e -f -d Ⅳ -c concat -p tmp/race.best.f.eval.test.human_eval.Ⅳ/oracle_outputs.c=concat.d=Ⅳ.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race.best.f.eval.test.human_eval.Ⅳ/oracle_outputs.c=concat.d=Ⅳ.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
#    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:titanx --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done
