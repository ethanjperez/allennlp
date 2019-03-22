#!/usr/bin/env bash

wget https://github.com/nlpdata/dream/archive/master.zip
unzip master.zip
mv dream-master/data datasets/dream
rm -rf master.zip
rm -rf dream-master