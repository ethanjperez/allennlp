#!/usr/bin/env bash
# To be run from allennlp root directory
cp -r datasets/race_raw datasets/race_raw_high
rm -r datasets/race_raw_high/*/middle
cp -r datasets/race_raw datasets/race_raw_middle
rm -r datasets/race_raw_middle/*/high
