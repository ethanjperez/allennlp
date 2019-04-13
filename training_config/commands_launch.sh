#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr.eval.Ⅱ_Ⅰ_Ⅱ_Ⅰ -j tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/model.tar.gz -e -f -d Ⅱ Ⅰ Ⅱ Ⅰ -c concat -p tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/oracle_outputs.c=concat.d=Ⅱ_Ⅰ_Ⅱ_Ⅰ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/d=Ⅱ_Ⅰ_Ⅱ_Ⅰ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr.eval.Ⅲ_Ⅰ_Ⅲ_Ⅰ -j tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/model.tar.gz -e -f -d Ⅲ Ⅰ Ⅲ Ⅰ -c concat -p tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/oracle_outputs.c=concat.d=Ⅲ_Ⅰ_Ⅲ_Ⅰ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/d=Ⅲ_Ⅰ_Ⅲ_Ⅰ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr.eval.Ⅳ_Ⅰ_Ⅳ_Ⅰ -j tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/model.tar.gz -e -f -d Ⅳ Ⅰ Ⅳ Ⅰ -c concat -p tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/oracle_outputs.c=concat.d=Ⅳ_Ⅰ_Ⅳ_Ⅰ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/d=Ⅳ_Ⅰ_Ⅳ_Ⅰ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr.eval.Ⅲ_Ⅱ_Ⅲ_Ⅱ -j tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/model.tar.gz -e -f -d Ⅲ Ⅱ Ⅲ Ⅱ -c concat -p tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/oracle_outputs.c=concat.d=Ⅲ_Ⅱ_Ⅲ_Ⅱ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/d=Ⅲ_Ⅱ_Ⅲ_Ⅱ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr.eval.Ⅳ_Ⅱ_Ⅳ_Ⅱ -j tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/model.tar.gz -e -f -d Ⅳ Ⅱ Ⅳ Ⅱ -c concat -p tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/oracle_outputs.c=concat.d=Ⅳ_Ⅱ_Ⅳ_Ⅱ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/d=Ⅳ_Ⅱ_Ⅳ_Ⅱ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr.eval.Ⅳ_Ⅲ_Ⅳ_Ⅲ -j tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/model.tar.gz -e -f -d Ⅳ Ⅲ Ⅳ Ⅲ -c concat -p tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/oracle_outputs.c=concat.d=Ⅳ_Ⅲ_Ⅳ_Ⅲ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.bsz=32.lr=1e-5.c=concat.rrrr/d=Ⅳ_Ⅲ_Ⅳ_Ⅲ.c=concat.race_h.dev.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done
