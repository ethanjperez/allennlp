"""
fetch_results.py

Obtain results (validation loss) and aggregate over sweeps to find best HP settings.
"""
import argparse
import json
import os


ROOT_DIR = "/checkpoint/siddk/debate/runs/dream"

BSZ = [8, 12, 16, 32]

LR = [1e-5, 2e-5, 3e-5, 5e-5, 5e-6]

SL_MODE = ['sl', 'sl-sents', 'i-sl-sents']


def parse_args():
    p = argparse.ArgumentParser("Dream Result Fetcher")
    return p.parse_args()


if __name__ == "__main__":
    # Parse Args
    args = parse_args()

    # Iterate through Full Sweep and Collect Results
    mode_dict = {m: {} for m in SL_MODE}
    for m in SL_MODE:
        for b in BSZ:
            for l in LR:
                # Open Metrics File
                if m[0] == 'i':
                    ckpt = os.path.join(ROOT_DIR, 'dream.sl_gpt.lr=%.1e.bsz=%d.m=sl-sents.i' % (l, b))
                else:
                    ckpt = os.path.join(ROOT_DIR, 'dream.sl_gpt.lr=%.1e.bsz=%d.m=%s' % (l, b, m))

                # if 'model.tar.gz' not in os.listdir(ckpt):
                #     print("Oops: %s" % ckpt)
                #     continue

                metric_file = sorted([metrics for metrics in os.listdir(ckpt) if 'metrics_epoch_' in metrics])[-1]

                with open(os.path.join(ckpt, metric_file), 'r') as f:
                    metric_data = json.load(f)

                best_epoch, best_rew = metric_data['best_epoch'], metric_data['best_validation_reward_turn_0_agent_e']

                # Add to mode_dict
                mode_dict[m][(l, b)] = (best_epoch, best_rew)

    # Write Report
    with open('graphs/report.md', 'w') as f:
        for m in SL_MODE:
            f.write("Mode - %s\n" % m)
            f.write("-" * 30 + "\n")

            for lr, bsz in mode_dict[m]:
                f.write("\tLR: %s\tBSZ: %d\tBest Epoch: %d\tBest Reward: %.5f\n" % (lr, bsz,
                                                                                      mode_dict[m][(lr, bsz)][0],
                                                                                      mode_dict[m][(lr, bsz)][1]))
            f.write("\n")
