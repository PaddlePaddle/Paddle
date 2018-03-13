#!/usr/bin/env xonsh
import sys
import numpy
import math
from utils import *

class TestError(Exception):
    pass

def test_model(path):
    '''
    test a model located in path
    '''
    cd @(path)

    run_train_cpu()
    test_with_history('./train.cost.txt', './history/train.cost.txt')

    run_train_gpu()
    test_with_history('./valid.cost.txt', './history/valid.cost.txt')

    run_predict()
    test_with_history('./predict.cost.txt', './history/predict.cost.txt')

def run_train_cpu():
    ./train.xsh --train_cost_out ./train.cost.txt --valid_cost_out ./valid.cost.txt

def run_train_gpu():
    ./train.xsh --train_cost_out ./train.cost.gpu.txt --valid_cost_out ./valid.cost.gpu.txt --gpu 0

def run_predict():
    ./predict.xsh --out_path predict.cost.txt

def test_with_history(out, his):
    '''
    format are <duration>\tcost
    '''
    out = np.array(load_log(out))
    his = np.array(load_log(his))

    nfields = len(out[0])
    for i in range(nfields):
        suc, msg = compare_trend(out[:, i], his[:, i], config.diff_thre)
        if not suc:
            raise TestError

def compare_trend(arr0, arr1, diff_thre=0.1):
    CHECK_EQ(len(arr0), len(arr1), "record number not match history")
    for i in range(len(arr0)):
        diff = abs(arr0[i] - arr1[i])
        CHECK_LT(diff / max(arr0[i], arr1[i]), diff_thre, "test trend failed")
    return SUC

def load_log(path):
    res = []
    with open(path) as f:
        for line in f.readlines():
            fs = line.strip().split('\t')
            key = fs[0]
            vs = [float(v) for v in fs[1].split()]
            res.append([key]+vs)
    return res

def ring_alarm(msg):
    print('error')
    print(msg)
    sys.exit(-1)
