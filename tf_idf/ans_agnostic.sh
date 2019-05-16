#!/usr/bin/env bash

python tf_idf/run_ans_agnostic.py -m q-only -t datasets/race_raw/train -v datasets/race_raw/test; python tf_idf/run_ans_agnostic.py -m n-sents -t datasets/race_raw/train -v datasets/race_raw/test; python fasttext/run_ans_agnostic.py -v datasets/race_raw/test; python tf_idf/run_ans_agnostic.py -m q-only -s dream -t datasets/dream/train.json -v datasets/dream/test.json; python tf_idf/run_ans_agnostic.py -m n-sents -s dream -t datasets/dream/train.json -v datasets/dream/test.json; python fasttext/run_ans_agnostic.py -d dream -v datasets/dream/test.json
