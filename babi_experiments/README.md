# bAbI Experiments

Commands to run bAbI Experiments:

### Full-Passage Judge Experiments

- Single: allennlp train babi_experiments/single_babi.jsonnet -s tmp/babi-single.full -d f -g

- Double: allennlp train babi_experiments/double_babi.jsonnet -s tmp/babi-double.full -d f -g

- Triple: allennlp train babi_experiments/triple_babi.jsonnet -s tmp/babi-triple.full -d f -g

### ORACLE Experiments

- Single: allennlp train babi_experiments/single_babi.jsonnet -s tmp/babi-single.A -j tmp/babi-single.full -g -e -m ssp -d [A/AA/AAA/B/BB/BBB/AB/BA]
- Double: allennlp train babi_experiments/double_babi.jsonnet -s tmp/babi-double.A -j tmp/babi-double.full -g -e -m ssp -d [A/AA/AAA/B/BB/BBB/AB/BA]