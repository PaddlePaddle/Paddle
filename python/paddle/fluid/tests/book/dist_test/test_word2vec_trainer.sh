#!/usr/bin/env bash

#export GLOG_v=3
#export GLOG_logtostderr=1

source test_word2vec.env

export TRAINING_ROLE=TRAINER
python test_word2vec.py
