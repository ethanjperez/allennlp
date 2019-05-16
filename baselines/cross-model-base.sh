#!/usr/bin/env bash

allennlp train race.best.last_epoch.f/config.json -s race.best.last_epoch.f -e -d f -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test_tfidf_qa'}" -r 2>&1 | tee cross_base_tfidf_qa.txt

allennlp train race.best.last_epoch.f/config.json -s race.best.last_epoch.f -e -d f -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test_tfidf_a'}" -r 2>&1 | tee cross_base_tfidf_a.txt

allennlp train race.best.last_epoch.f/config.json -s race.best.last_epoch.f -e -d f -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test_fasttext'}" -r 2>&1 | tee cross_base_fasttext.txt

allennlp train race.best.last_epoch.f/config.json -s race.best.last_epoch.f -e -d f -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test_bert_base'}" -r 2>&1 | tee cross_base_base.txt

allennlp train race.best.last_epoch.f/config.json -s race.best.last_epoch.f -e -d f -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test_bert_large'}" -r 2>&1 | tee cross_base_large.txt

allennlp train race.best.last_epoch.f/config.json -s race.best.last_epoch.f -e -d f -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test_human'}" -r 2>&1 | tee cross_base_human.txt
