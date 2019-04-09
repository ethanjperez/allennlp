#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=32.f -e -r -d f -c concat -p tmp/race_m.best.bsz=32.f/oracle_outputs.c=concat.d=f.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=32.f/d=f.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=32.f -e -r -d rrrr rrrr rrrr rrrr -c concat -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=32.f/d=rrrr_rrrr_rrrr_rrrr.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=32.f -e -r -d ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ -c concat -p tmp/race_m.best.bsz=32.f/oracle_outputs.c=concat.d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=32.f/d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=32.f -e -r -d AW AW AW AW -c concat -p tmp/race_m.best.bsz=32.f/oracle_outputs.c=concat.d=AW_AW_AW_AW.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=32.f/d=AW_AW_AW_AW.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=32.f -e -r -d ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ -c concat -p tmp/race_m.best.bsz=32.f/oracle_outputs.c=concat.d=ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=32.f/d=ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=32.f -e -r -d ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ -c concat -p tmp/race_m.best.bsz=32.f/oracle_outputs.c=concat.d=ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=32.f/d=ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=32.f -e -r -d ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ -c concat -p tmp/race_m.best.bsz=32.f/oracle_outputs.c=concat.d=ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=32.f/d=ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=32.f -e -r -d ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ -c concat -p tmp/race_m.best.bsz=32.f/oracle_outputs.c=concat.d=ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=32.f/d=ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=32.f -e -r -d ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ -c concat -p tmp/race_m.best.bsz=32.f/oracle_outputs.c=concat.d=ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=32.f/d=ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=32.f -e -r -d ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ -c concat -p tmp/race_m.best.bsz=32.f/oracle_outputs.c=concat.d=ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=32.f/d=ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.c=concat.race_h.dev.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=30000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done
