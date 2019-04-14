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
