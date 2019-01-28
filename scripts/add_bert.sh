#!/usr/bin/env bash
# From the allennlp repo root
pip install pytorch-pretrained-bert==0.4.0
pip install matplotlib==2.2.3
mkdir -p datasets/bert
cd datasets/bert
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
cd ../..
