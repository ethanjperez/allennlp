"""
dream_runner.py

SLURM Helper Function for Quickly Running various DREAM BERT MC Sweeps.
"""
import argparse
import os

PYTHON_PATH = '/private/home/siddk/.conda/envs/allennlp/bin/python3.6'

PROGRAM_PATH = '/private/home/siddk/allennlp/allennlp/run.py'

TRAIN_CONFIG = "{'trainer': { 'num_epochs': 20, 'patience': 10, 'validation_metric': '-loss', 'cuda_device': 0, " \
               "'learning_rate_scheduler': {'type': 'reduce_on_plateau', 'factor': 0.67, 'mode': 'max', " \
               "'patience': 1}, 'optimizer': {'lr': %.7f, 'type': 'bert_adam'}}}"

BASE_BSZ = 8

ACCUMULATION_STEPS = [2, 4]

MODE_CONFIGS = {m: '/private/home/siddk/allennlp/training_config/dream/bert_mc_%s.dream.bsz=8.lr=FILL.jsonnet' % m
                for m in ['pq2a', 'a', 'q2a']}
MODE_CONFIGS['gpt'] = '/private/home/siddk/allennlp/training_config/dream/bert_mc_gpt.dream.bsz=1.lr=FILL.jsonnet'

CKPT_PATH = "/checkpoint/siddk/debate/runs/dream/dream.bert_mc_%s.bsz=%d.lr=%.1e.f"

BEST_CONFIG = '/private/home/siddk/allennlp/training_config/dream/dream.best.jsonnet'

BEST_TRAIN_CONFIG = '/private/home/siddk/allennlp/training_config/dream/dream.best.train.jsonnet'

BEST_CKPT_PATH = '/checkpoint/siddk/debate/runs/dream/dream.bert_mc_gpt.bsz=32.lr=2.0e-05.f'


# SL Search Parameters
BSZ = [8, 12, 16]

LR = [1e-5, 2e-5, 3e-5, 5e-6]

SL_MODE = ['sl', 'sl-sents', 'i-sl-sents']

# QA Aux Parameters
AUX = {
    5e-6: [1, 2, 4, 8],
    1e-5: [.5, 1, 2, 4],
    2e-5: [.25, .5, 1, 2],
    3e-5: [.125, .25, .5, 1]
}


def parse_args():
    p = argparse.ArgumentParser("Dream SLURM Runner")
    p.add_argument("-m", "--mode", default='pq2a', help='Default BERT Mode - choices <pq2a | a | q2a | gpt | oracle>')
    p.add_argument("-o", "--oracle_mode", default='eval', help='Oracle Mode for dumping - choices <eval | train>')
    p.add_argument("-s", "--supervised", default=0, type=int, help='Debate mode to run search over for super training')
    p.add_argument("-b", "--bsz", default=8, type=int, help='Batch size for training')

    return p.parse_args()


if __name__ == "__main__":
    # Parse Args
    args = parse_args()

    # Lift SLURM Node ID
    s_id = int(os.environ['SLURM_NODEID'])

    # Set Up Run Command
    if args.mode in ['pq2a', 'a', 'q2a']:
        run_command = '%s %s train %s -s %s -d f -a %d -o "%s"' % (
            PYTHON_PATH,
            PROGRAM_PATH,
            MODE_CONFIGS[args.mode],
            CKPT_PATH % (args.mode, BASE_BSZ * ACCUMULATION_STEPS[s_id % len(ACCUMULATION_STEPS)], LR[s_id % len(LR)]),
            ACCUMULATION_STEPS[s_id % len(ACCUMULATION_STEPS)],
            TRAIN_CONFIG % LR[s_id % len(LR)]
        )

    elif args.mode in ['gpt']:
        run_command = '%s %s train %s -s %s -d f -a 32' % (
            PYTHON_PATH,
            PROGRAM_PATH,
            BEST_CONFIG,
            CKPT_PATH % (args.mode, 32, LR[1])
        )

    elif args.mode in ['oracle']:
        if args.oracle_mode == 'eval':
            run_command = '%s %s train %s -s %s -e -r -d %s -c concat -p %s' % (
                PYTHON_PATH,
                PROGRAM_PATH,
                BEST_CONFIG,
                BEST_CKPT_PATH,
                DEBATE_MODES[s_id],
                os.path.join(BEST_CKPT_PATH, 'oracle_outputs.d=' + ("".join(DEBATE_MODES[s_id].split()) + '.dev.pkl'))
            )
        elif args.oracle_mode == 'train':
            run_command = '%s %s train %s -s %s -e -r -d %s -c concat -p %s' % (
                PYTHON_PATH,
                PROGRAM_PATH,
                BEST_TRAIN_CONFIG,
                BEST_CKPT_PATH,
                DEBATE_MODES[s_id],
                os.path.join(BEST_CKPT_PATH, 'oracle_outputs.d=' + ("".join(DEBATE_MODES[s_id].split()) + '.train.pkl'))
            )

    elif args.mode in ['supervised']:
        atom_lr = LR[s_id % len(LR)]
        atom_bsz = BSZ[s_id % len(BSZ)]
        sl_mode = SL_MODE[args.supervised]

        config = '/private/home/siddk/allennlp/training_config/dream/sl.dream.gpt.bsz=1.lr=FILL.jsonnet'

        if sl_mode[0] == 'i':
            ckpt_path = '/checkpoint/siddk/debate/runs/dream/dream.sl_gpt.lr=%.1e.bsz=%d.m=sl-sents.i+theory' % (atom_lr,
                                                                                                          atom_bsz)
        else:
            ckpt_path = '/checkpoint/siddk/debate/runs/dream/dream.sl_gpt.lr=%.1e.bsz=%d.m=%s+theory' % (atom_lr, atom_bsz,
                                                                                                  sl_mode)

        judge_path = '/checkpoint/siddk/debate/runs/dream/dream.bert_mc_gpt.bsz=32.lr=2.0e-05.f/model.tar.gz'
        oracle_path = '/checkpoint/siddk/debate/runs/dream/dream.bert_mc_gpt.bsz=32.lr=2.0e-05.f/oracle_outputs.d=I.all.pkl'

        if sl_mode[0] == 'i':
            run_command = '%s %s train %s -s %s -j %s -d e -m sl-sents -i -p %s -a %d -c concat -t' % (
                PYTHON_PATH,
                PROGRAM_PATH,
                config,
                ckpt_path,
                judge_path,
                oracle_path,
                atom_bsz
            )

        else:
            run_command = '%s %s train %s -s %s -j %s -d e -m %s -p %s -a %d -c concat -t' % (
                PYTHON_PATH,
                PROGRAM_PATH,
                config,
                ckpt_path,
                judge_path,
                sl_mode,
                oracle_path,
                atom_bsz
            )

        run_command += ' -o "%s"' % (TRAIN_CONFIG % atom_lr)

    elif args.mode in ['qa_aux']:
        atom_lr = LR[s_id % len(LR)]
        atom_qa = AUX[LR[s_id % len(LR)]][s_id // len(LR)]
        atom_bsz = args.bsz
        sl_mode = SL_MODE[args.supervised]

        config = '/private/home/siddk/allennlp/training_config/dream/sl.dream.gpt.bsz=1.lr=FILL.jsonnet'

        if sl_mode[0] == 'i':
            ckpt_path = '/checkpoint/siddk/debate/runs/dream/dream.sl_gpt.lr=%.1e.bsz=%d.m=sl-sents.i+qa_%.3f' % (
                atom_lr, atom_bsz, atom_qa)
        else:
            ckpt_path = '/checkpoint/siddk/debate/runs/dream/dream.sl_gpt.lr=%.1e.bsz=%d.m=%s+qa_%.3f' % (
                atom_lr, atom_bsz, sl_mode, atom_qa)

        judge_path = '/checkpoint/siddk/debate/runs/dream/dream.bert_mc_gpt.bsz=32.lr=2.0e-05.f/model.tar.gz'
        oracle_path = '/checkpoint/siddk/debate/runs/dream/dream.bert_mc_gpt.bsz=32.lr=2.0e-05.f/oracle_outputs.d=I.all.pkl'

        if sl_mode[0] == 'i':
            run_command = '%s %s train %s -s %s -j %s -d e -m sl-sents -i -p %s -a %d -c concat -q %.3f' % (
                PYTHON_PATH,
                PROGRAM_PATH,
                config,
                ckpt_path,
                judge_path,
                oracle_path,
                atom_bsz,
                atom_qa
            )

        else:
            run_command = '%s %s train %s -s %s -j %s -d e -m %s -p %s -a %d -c concat -q %.3f' % (
                PYTHON_PATH,
                PROGRAM_PATH,
                config,
                ckpt_path,
                judge_path,
                sl_mode,
                oracle_path,
                atom_bsz,
                atom_qa
            )

        run_command += ' -o "%s"' % (TRAIN_CONFIG % atom_lr)

    print("Running %s!" % run_command)
    os.system(run_command)