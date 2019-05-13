"""
fetch_results.py

Obtain results (validation loss) and aggregate over sweeps to find best HP settings.
"""
import argparse
import json
import os


ROOT_DIR = "/checkpoint/siddk/debate/runs/dream"

BSZ = [8, 12, 16]

LR = [1e-5, 2e-5, 3e-5, 5e-6]

# QA Aux Parameters
AUX = {
    5e-6: [1, 2, 4, 8],
    1e-5: [.5, 1, 2, 4],
    2e-5: [.25, .5, 1, 2],
    3e-5: [.125, .25, .5, 1]
}

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
        for b in BSZ[:1]:
            for l in LR:
                for ax in range(4):
                    # Open Metrics
                    if m[0] == 'i':
                        ckpt = os.path.join(ROOT_DIR, 'dream.sl_gpt.lr=%.1e.bsz=%d.m=sl-sents.i+qa_%.3f' % (l, b,
                                                                                                            AUX[l][ax]))
                    else:
                        ckpt = os.path.join(ROOT_DIR, 'dream.sl_gpt.lr=%.1e.bsz=%d.m=%s+qa_%.3f' % (l, b, m,
                                                                                                    AUX[l][ax]))

                    if 'model.tar.gz' not in os.listdir(ckpt):
                        print("Oops: %s" % ckpt)
                        continue

                    metric_file = sorted([metrics for metrics in os.listdir(ckpt) if 'metrics_epoch_' in metrics])[-1]

                    with open(os.path.join(ckpt, metric_file), 'r') as f:
                        metric_data = json.load(f)

                    best_epoch, best_rew = metric_data['best_epoch'], metric_data['best_validation_reward_turn_0_agent_e']

                    # Add to mode_dict
                    mode_dict[m][(l, b, AUX[l][ax])] = (best_epoch, best_rew)

    # Write Report
    with open('graphs-5-12/report.md', 'w') as f:
        for m in SL_MODE:
            f.write("Mode - %s\n" % m)
            f.write("-" * 30 + "\n")

            for lr, bsz, qw in mode_dict[m]:
                f.write("\tLR: %s\tBSZ: %d\tQA Weight: %.3f\tBest Epoch: %d\tBest Reward: %.5f\n" % (lr, bsz, qw,
                                                                                      mode_dict[m][(lr, bsz, qw)][0],
                                                                                      mode_dict[m][(lr, bsz, qw)][1]))
            f.write("\n")
