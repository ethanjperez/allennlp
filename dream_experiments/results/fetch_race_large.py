"""
fetch_race_large.py

Obtain results (validation accuracy) on RACE Bert Large training jobs.
"""
import json
import os

ROOT_DIR = "/checkpoint/siddk/debate/runs/race"

BSZ = [8, 12, 16, 32]

LR = ["1e-5", "2e-5", "3e-5", "5e-6"]


if __name__ == "__main__":
    # Iterate through Full Sweep and Collect Results
    mode_dict = {}
    for b in BSZ:
        for l in LR:
            cdir = os.path.join(ROOT_DIR, 'race.f.bert_mc_gpt.large.bsz=%d.lr=%s.f' % (b, l))

            if not os.path.exists(cdir):
                print("Oops: %s" % cdir)
                continue

            elif 'model.tar.gz' not in os.listdir(cdir):
                print("Oops: %s" % cdir)
                continue

            metric_file = sorted([metrics for metrics in os.listdir(cdir) if 'metrics_epoch_' in metrics])[-1]
            with open(os.path.join(cdir, metric_file), 'r') as f:
                metric_data = json.load(f)

            if 'best_epoch' not in metric_data:
                continue
            best_epoch, best_rew = metric_data['best_epoch'], metric_data['best_validation_start_acc']

            # Add to mode_dict
            mode_dict[(l, b)] = (best_epoch, best_rew)

    # Write Report
    with open('graphs/report.md', 'w') as f:
        f.write("Mode - BERT Large\n")
        f.write("-" * 30 + "\n")
        for l, b in mode_dict:
            f.write("\tLR: %s\tBSZ: %d\tBest Epoch: %d\tBest Valid Acc: %.5f\n" % (l, b, mode_dict[(l, b)][0],
                                                                                   mode_dict[(l, b)][1]))
