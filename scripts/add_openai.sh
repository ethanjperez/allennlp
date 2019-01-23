#!/usr/bin/env bash
# To be run from allennlp root directory
mkdir -p datasets/openai
cd datasets/openai
wget https://s3-us-west-2.amazonaws.com/allennlp/models/openai-transformer-lm-2018.07.23.tar.gz
