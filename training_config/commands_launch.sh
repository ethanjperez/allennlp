#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f -e -r -d Ⅱ Ⅰ Ⅱ Ⅰ Ⅱ Ⅰ Ⅱ Ⅰ Ⅱ Ⅰ Ⅱ Ⅰ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f -e -r -d Ⅲ Ⅰ Ⅲ Ⅰ Ⅲ Ⅰ Ⅲ Ⅰ Ⅲ Ⅰ Ⅲ Ⅰ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=Ⅲ_Ⅰ_Ⅲ_Ⅰ_Ⅲ_Ⅰ_Ⅲ_Ⅰ_Ⅲ_Ⅰ_Ⅲ_Ⅰ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=Ⅲ_Ⅰ_Ⅲ_Ⅰ_Ⅲ_Ⅰ_Ⅲ_Ⅰ_Ⅲ_Ⅰ_Ⅲ_Ⅰ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f -e -r -d Ⅳ Ⅰ Ⅳ Ⅰ Ⅳ Ⅰ Ⅳ Ⅰ Ⅳ Ⅰ Ⅳ Ⅰ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=Ⅳ_Ⅰ_Ⅳ_Ⅰ_Ⅳ_Ⅰ_Ⅳ_Ⅰ_Ⅳ_Ⅰ_Ⅳ_Ⅰ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=Ⅳ_Ⅰ_Ⅳ_Ⅰ_Ⅳ_Ⅰ_Ⅳ_Ⅰ_Ⅳ_Ⅰ_Ⅳ_Ⅰ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f -e -r -d Ⅲ Ⅱ Ⅲ Ⅱ Ⅲ Ⅱ Ⅲ Ⅱ Ⅲ Ⅱ Ⅲ Ⅱ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=Ⅲ_Ⅱ_Ⅲ_Ⅱ_Ⅲ_Ⅱ_Ⅲ_Ⅱ_Ⅲ_Ⅱ_Ⅲ_Ⅱ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=Ⅲ_Ⅱ_Ⅲ_Ⅱ_Ⅲ_Ⅱ_Ⅲ_Ⅱ_Ⅲ_Ⅱ_Ⅲ_Ⅱ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f -e -r -d Ⅳ Ⅱ Ⅳ Ⅱ Ⅳ Ⅱ Ⅳ Ⅱ Ⅳ Ⅱ Ⅳ Ⅱ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=Ⅳ_Ⅱ_Ⅳ_Ⅱ_Ⅳ_Ⅱ_Ⅳ_Ⅱ_Ⅳ_Ⅱ_Ⅳ_Ⅱ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=Ⅳ_Ⅱ_Ⅳ_Ⅱ_Ⅳ_Ⅱ_Ⅳ_Ⅱ_Ⅳ_Ⅱ_Ⅳ_Ⅱ.c=concat.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f -e -r -d Ⅳ Ⅲ Ⅳ Ⅲ Ⅳ Ⅲ Ⅳ Ⅲ Ⅳ Ⅲ Ⅳ Ⅲ -c concat -p tmp/race_m.best.bsz=12.f/oracle_outputs.c=concat.d=Ⅳ_Ⅲ_Ⅳ_Ⅲ_Ⅳ_Ⅲ_Ⅳ_Ⅲ_Ⅳ_Ⅲ_Ⅳ_Ⅲ.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=Ⅳ_Ⅲ_Ⅳ_Ⅲ_Ⅳ_Ⅲ_Ⅳ_Ⅲ_Ⅳ_Ⅲ_Ⅳ_Ⅲ.c=concat.race_h.dev.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=30000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done








mv debate_logs.d=AW_AW_AW_AW.json race_h.debate_logs.d=AW_AW_AW_AW.json
mv debate_logs.d=f.json race_h.debate_logs.d=f.json
mv debate_logs.d=ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.json race_h.debate_logs.d=ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.json
mv debate_logs.d=ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.json race_h.debate_logs.d=ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.json
mv debate_logs.d=ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.json race_h.debate_logs.d=ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.json
mv debate_logs.d=ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.json race_h.debate_logs.d=ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.json
mv debate_logs.d=ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.json race_h.debate_logs.d=ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.json
mv debate_logs.d=ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.json race_h.debate_logs.d=ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.json
mv debate_logs.d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.json race_h.debate_logs.d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.json
mv debate_logs.d=rrrr_rrrr_rrrr_rrrr.json race_h.debate_logs.d=rrrr_rrrr_rrrr_rrrr.json
