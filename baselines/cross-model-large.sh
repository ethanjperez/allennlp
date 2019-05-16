#!/usr/bin/env bash

allennlp train training_config/race/race.large.best.jsonnet -s tmp/cross.large_tfidf_qa -j /checkpoint/siddk/debate/runs/race/race.f.bert_mc_gpt.large.bsz=12.lr=5e-6.f/model.tar.gz -e -d f -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test_tfidf_qa'}" 2>&1 | tee cross_large_tfidf_qa.txt

allennlp train training_config/race/race.large.best.jsonnet -s tmp/cross.large_tfidf_a -j /checkpoint/siddk/debate/runs/race/race.f.bert_mc_gpt.large.bsz=12.lr=5e-6.f/model.tar.gz -e -d f -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test_tfidf_a'}" 2>&1 | tee cross_large_tfidf_a.txt

allennlp train training_config/race/race.large.best.jsonnet -s tmp/cross.large_fasttext -j /checkpoint/siddk/debate/runs/race/race.f.bert_mc_gpt.large.bsz=12.lr=5e-6.f/model.tar.gz -e -d f -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test_fasttext'}" 2>&1 | tee cross_large_fasttext.txt

allennlp train training_config/race/race.large.best.jsonnet -s tmp/cross.large_base -j /checkpoint/siddk/debate/runs/race/race.f.bert_mc_gpt.large.bsz=12.lr=5e-6.f/model.tar.gz -e -d f -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test_bert_base'}" 2>&1 | tee cross_large_base.txt

allennlp train training_config/race/race.large.best.jsonnet -s tmp/cross.large_large -j /checkpoint/siddk/debate/runs/race/race.f.bert_mc_gpt.large.bsz=12.lr=5e-6.f/model.tar.gz -e -d f -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test_bert_large'}" 2>&1 | tee cross_large_large.txt

allennlp train training_config/race/race.large.best.jsonnet -s tmp/cross.large_human -j /checkpoint/siddk/debate/runs/race/race.f.bert_mc_gpt.large.bsz=12.lr=5e-6.f/model.tar.gz -e -d f -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test_human'}" 2>&1 | tee cross_large_human.txt