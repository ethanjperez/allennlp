"""
dream_runner.py

SLURM Helper Function for Quickly Running various DREAM BERT MC Sweeps.
"""
import argparse
import os

PYTHON_PATH = '/private/home/siddk/.conda/envs/allennlp/bin/python3.6'

PROGRAM_PATH = '/private/home/siddk/allennlp/allennlp/run.py'

TRAIN_CONFIG = "{'trainer': { 'num_epochs': 20, 'patience': 3, 'validation_metric': '+start_acc', 'cuda_device': 0, " \
               "'learning_rate_scheduler': {'type': 'reduce_on_plateau', 'factor': 0.67, 'mode': 'max', " \
               "'patience': 1}, 'optimizer': {'lr': %.7f, 'type': 'bert_adam'}}}"

LR = [1e-5, 2e-5, 3e-5, 5e-5, 5e-6]

BASE_BSZ = 8

ACCUMULATION_STEPS = [4]

MODE_CONFIGS = {m: '/private/home/siddk/allennlp/training_config/dream/bert_mc_%s.dream.bsz=8.lr=FILL.jsonnet' % m
                for m in ['pq2a', 'a', 'q2a']}

CKPT_PATH = "/checkpoint/siddk/debate/dream/dream.bert_mc_%s.bsz=%d.lr=%.1e.f"


def parse_args():
    p = argparse.ArgumentParser("Dream SLURM Runner")
    p.add_argument("-m", "--mode", default='pq2a', help='Default BERT Mode - choices <pq2a | a | q2a>')

    return p.parse_args()


if __name__ == "__main__":
    # Parse Args
    args = parse_args()

    # Lift SLURM Node ID
    s_id = int(os.environ['SLURM_NODEID'])

    # Set Up Run Command
    run_command = '%s %s train %s -s %s -d f -a %d -o "%s"' % (
        PYTHON_PATH,
        PROGRAM_PATH,
        MODE_CONFIGS[args.mode],
        CKPT_PATH % (args.mode, BASE_BSZ * ACCUMULATION_STEPS[s_id % len(ACCUMULATION_STEPS)], LR[s_id % len(LR)]),
        ACCUMULATION_STEPS[s_id % len(ACCUMULATION_STEPS)],
        TRAIN_CONFIG % LR[s_id % len(LR)]
    )
    print("Running %s!" % run_command)
    os.system(run_command)
