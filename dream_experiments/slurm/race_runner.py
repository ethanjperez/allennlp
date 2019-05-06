"""
race_runner.py

SLURM Helper for quickly running RACE Training jobs
"""
import argparse
import os

PYTHON_PATH = '/private/home/siddk/.conda/envs/allennlp/bin/python3.6'
PROGRAM_PATH = '/private/home/siddk/allennlp/allennlp/run.py'


# Middle/High School Training Commands
mh_commands = [
    """train training_config/bert_mc_gpt.large.race.lr=3e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=16.lr=3e-5.f -d f -a 16 -f -o "{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}" """,
    """train training_config/bert_mc_gpt.large.race.lr=3e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=3e-5.f -d f -a 12 -f -o "{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}" """,
    """train training_config/bert_mc_gpt.large.race.lr=5e-6.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=8.lr=5e-6.f -d f -a 8 -f -o "{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}" """,
    """train training_config/bert_mc_gpt.large.race.lr=1e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=8.lr=1e-5.f -d f -a 8 -f -o "{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}" """,
]

# Length Training Commands
len_commands = [
    """train training_config/bert_mc_gpt.large.race.lr=5e-6.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=32.lr=5e-6.f -d f -a 32 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """
    """train training_config/bert_mc_gpt.large.race.lr=1e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=32.lr=1e-5.f -d f -a 32 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=2e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=32.lr=2e-5.f -d f -a 32 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=3e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=32.lr=3e-5.f -d f -a 32 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=5e-6.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=16.lr=5e-6.f -d f -a 16 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=1e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=16.lr=1e-5.f -d f -a 16 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=2e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=16.lr=2e-5.f -d f -a 16 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=3e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=16.lr=3e-5.f -d f -a 16 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=5e-6.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=12.lr=5e-6.f -d f -a 12 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=1e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=12.lr=1e-5.f -d f -a 12 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=2e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=12.lr=2e-5.f -d f -a 12 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=3e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=12.lr=3e-5.f -d f -a 12 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=5e-6.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=8.lr=5e-6.f -d f -a 8 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=1e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=8.lr=1e-5.f -d f -a 8 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=2e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=8.lr=2e-5.f -d f -a 8 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """,
    """train training_config/bert_mc_gpt.large.race.lr=3e-5.jsonnet -s /checkpoint/siddk/debate/runs/race/race.num_sents_leq_12.bert_mc_gpt.large.bsz=8.lr=3e-5.f -d f -a 8 -o "{'train_data_path': 'datasets/num_sents_leq_12/train', 'validation_data_path': 'datasets/num_sents_leq_12/dev'}" -f """
]

# Middle -> High School
round_robin = [
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.f.ra -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e -d f -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/test'}" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=f.ra.c=concat.race_h.dev.txt""",
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.rr_rr_rr_rr_rr_rr.ra -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e --require_action -d rr rr rr rr rr rr -c concat -p /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/oracle_outputs.c=concat.d=rr_rr_rr_rr_rr_rr.ra.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/test'}\" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=rr_rr_rr_rr_rr_rr.ra.c=concat.race_h.dev.txt""",
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.ra -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e --require_action -d ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ -c concat -p /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/oracle_outputs.c=concat.d=ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.ra.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/test'}\" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.ra.c=concat.race_h.dev.txt""",
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.ra -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e --require_action -d ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ -c concat -p /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/oracle_outputs.c=concat.d=ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.ra.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/test'}\" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.ra.c=concat.race_h.dev.txt""",
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.ra -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e --require_action -d ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ -c concat -p /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/oracle_outputs.c=concat.d=ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.ra.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/test'}\" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.ra.c=concat.race_h.dev.txt""",
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.ra -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e --require_action -d ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ -c concat -p /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/oracle_outputs.c=concat.d=ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.ra.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/test'}\" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.ra.c=concat.race_h.dev.txt""",
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.ra -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e --require_action -d ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ -c concat -p /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/oracle_outputs.c=concat.d=ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.ra.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/test'}\" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.ra.c=concat.race_h.dev.txt""",
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.ra -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e --require_action -d ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ -c concat -p /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/oracle_outputs.c=concat.d=ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.ra.race_h.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/test'}\" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.ra.c=concat.race_h.dev.txt"""
]

round_robin_dev = [
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.ra.dev -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e --require_action -d ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ ⅠⅡ -c concat -p /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/oracle_outputs.c=concat.d=ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.ra.race_h.test.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.ra.c=concat.race_h.test.txt""",
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.ra.dev -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e --require_action -d ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ ⅠⅢ -c concat -p /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/oracle_outputs.c=concat.d=ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.ra.race_h.test.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.ra.c=concat.race_h.test.txt""",
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.ra.dev -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e --require_action -d ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ ⅠⅣ -c concat -p /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/oracle_outputs.c=concat.d=ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.ra.race_h.test.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.ra.c=concat.race_h.test.txt""",
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.ra.dev -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e --require_action -d ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ ⅡⅢ -c concat -p /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/oracle_outputs.c=concat.d=ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.ra.race_h.test.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.ra.c=concat.race_h.test.txt""",
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.ra.dev -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e --require_action -d ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ ⅡⅣ -c concat -p /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/oracle_outputs.c=concat.d=ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.ra.race_h.test.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.ra.c=concat.race_h.test.txt""",
    """train training_config/race/race.large.best.jsonnet -s /checkpoint/siddk/debate/runs/race/race_m.large.best.f.eval.ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.ra.dev -j /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/model.tar.gz -e --require_action -d ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ ⅢⅣ -c concat -p /checkpoint/siddk/debate/runs/race/race_m.bert_mc_gpt.large.bsz=12.lr=1e-5.f/oracle_outputs.c=concat.d=ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.ra.race_h.test.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee /checkpoint/siddk/debate/logs/race/race_m.large.best.f/d=ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.ra.c=concat.race_h.test.txt"""
]


def parse_args():
    p = argparse.ArgumentParser("RACE SLURM Runner")
    p.add_argument("-m", "--mode", default='mh', help='Default run mode in < mh (middle-high), pl (passage-len)>')

    return p.parse_args()


if __name__ == "__main__":
    # Parse Args
    args = parse_args()

    # Lift SLURM Node ID
    s_id = int(os.environ['SLURM_NODEID'])

    # Middle/High Judge Training
    if args.mode == 'mh':
        run_command = "%s %s %s" % (PYTHON_PATH, PROGRAM_PATH, mh_commands[s_id])

        print('Running %s!' % run_command)
        os.system(run_command)

    elif args.mode == 'len':
        run_command = "%s %s %s" % (PYTHON_PATH, PROGRAM_PATH, len_commands[s_id])

        print('Running %s!' % run_command)
        os.system(run_command)

    elif args.mode == 'rr':
        run_command = "%s %s %s" % (PYTHON_PATH, PROGRAM_PATH, round_robin[s_id])

        print('Running %s!' % run_command)
        os.system(run_command)

    elif args.mode == 'rd':
        run_command = "%s %s %s" % (PYTHON_PATH, PROGRAM_PATH, round_robin_dev[s_id])

        print('Running %s!' % run_command)
        os.system(run_command)
