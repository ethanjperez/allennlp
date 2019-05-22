#!/usr/bin/env bash

python baselines/fasttext.py -v eval/dream/human/test/Ⅰ.json eval/dream/human/test/Ⅱ.json eval/dream/human/test/Ⅲ.json 2>&1 | tee fast_human1.txt;
python baselines/fasttext.py -v eval/dream/human.2/test/Ⅰ.json eval/dream/human.2/test/Ⅱ.json eval/dream/human.2/test/Ⅲ.json 2>&1 | tee fast_human2.txt;
python baselines/fasttext.py -v eval/dream/human.3/test/Ⅰ.json eval/dream/human.3/test/Ⅱ.json eval/dream/human.3/test/Ⅲ.json 2>&1 | tee fast_human3.txt;
python baselines/fasttext.py -v eval/dream/tfidf.o_q/test/dream_test_tfidf_Ⅰ_wq.json eval/dream/tfidf.o_q/test/dream_test_tfidf_Ⅱ_wq.json eval/dream/tfidf.o_q/test/dream_test_tfidf_Ⅲ_wq.json 2>&1 | tee fast_tfidf_qa.txt;
python baselines/fasttext.py -v eval/dream/tfidf.o/test/dream_test_tfidf_Ⅰ.json eval/dream/tfidf.o/test/dream_test_tfidf_Ⅱ.json eval/dream/tfidf.o/test/dream_test_tfidf_Ⅲ.json 2>&1 | tee fast_tfidf_a.txt;
python baselines/fasttext.py -v eval/dream/fasttext.o/test/Ⅰ.json eval/dream/fasttext.o/test/Ⅱ.json eval/dream/fasttext.o/test/Ⅲ.json 2>&1 | tee fast_fast.txt;
python baselines/fasttext.py -v eval/dream/cross_ranker.last_epoch/test/Ⅰ.json eval/dream/cross_ranker.last_epoch/test/Ⅱ.json eval/dream/cross_ranker.last_epoch/test/Ⅲ.json 2>&1 | tee fast_base.txt;
python baselines/fasttext.py -v eval/dream/cross_ranker.large.best_epoch/test/Ⅰ.json eval/dream/cross_ranker.large.best_epoch/test/Ⅱ.json eval/dream/cross_ranker.large.best_epoch/test/Ⅲ.json 2>&1 | tee fast_large.txt;
python baselines/fasttext.py -v eval/dream/sl/test/ⅰ.json eval/dream/sl/test/ⅱ.json eval/dream/sl/test/ⅲ.json 2>&1 | tee fast_sl.txt;
python baselines/fasttext.py -v eval/dream/sl-sents/test/ⅰ.json eval/dream/sl-sents/test/ⅱ.json eval/dream/sl-sents/test/ⅲ.json 2>&1 | tee fast_sl_sent.txt;
python baselines/fasttext.py -v eval/dream/sl-sents-influence/test/ⅰ.json eval/dream/sl-sents-influence/test/ⅱ.json eval/dream/sl-sents-influence/test/ⅲ.json 2>&1 | tee fast_sl_sent_i.txt