#!/usr/bin/env bash
# Run from allennlp base folder
cp -r datasets/race_raw datasets/race_raw_high
rm -r datasets/race_raw_high/*/middle
cp -r datasets/race_raw datasets/race_raw_middle
rm -r datasets/race_raw_middle/*/high
