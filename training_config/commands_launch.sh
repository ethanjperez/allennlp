#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f.race_h.dev.ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ -j tmp/race_m.best.bsz=12.f/model.tar.gz -e -d ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f.race_h.test.ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ -j tmp/race_m.best.bsz=12.f/model.tar.gz -e -d ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.race_h.test.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/test'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.c=concat.race_h.test.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
#    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done

"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f.test.ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ -j tmp/race_m.best.bsz=12.f/model.tar.gz -e -d ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.c=concat.test.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f.test.ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ -j tmp/race_m.best.bsz=12.f/model.tar.gz -e -d ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.c=concat.test.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f.test.ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ -j tmp/race_m.best.bsz=12.f/model.tar.gz -e -d ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.c=concat.test.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f.test.ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ -j tmp/race_m.best.bsz=12.f/model.tar.gz -e -d ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.c=concat.test.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f.test.ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ -j tmp/race_m.best.bsz=12.f/model.tar.gz -e -d ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.c=concat.test.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f.test.ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ -j tmp/race_m.best.bsz=12.f/model.tar.gz -e -d ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.c=concat.test.txt"


"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f.dev.ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ -j tmp/race_m.best.bsz=12.f/model.tar.gz -e -d ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.c=concat.test.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f.test.ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ -j tmp/race_m.best.bsz=12.f/model.tar.gz -e -d ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.test.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.c=concat.test.txt"



cd ~/research/allennlp/datasets/race_raw/
QTYPES=("A" "B" "C" "D" "E")
for QTYPE in "${QTYPES[@]}"; do
    cp -r $QTYPE $QTYPE.high
    rm -r $QTYPE.high/middle
    cp -r $QTYPE $QTYPE.middle
    rm -r $QTYPE.middle/high
done

mv debate_logs.d=Ⅰ.json Ⅰ.json
mv debate_logs.d=Ⅱ.json Ⅱ.json
mv debate_logs.d=Ⅲ.json Ⅲ.json
