#!/usr/bin/env xonsh
import os
import repo
import prepare
import config
from utils import *

url = 'https://github.com/PaddlePaddle/Paddle.git'
dst = './paddle_code'

def test_released_whl():
    prepare.get_whl()
    prepare.install_whl()
    test_models()

def test_latest_source():
    if not os.path.isdir(dst):
    repo.clone(url, dst)
    repo.pull(dst)
    prepare.compile()

    prepare.install_whl()

def test_models():
    cd @(config.workspace)
    for path in $(ls models):
        log.warn('test %s', path)
        model_path = pjoin(config.workspace, path)
        try:
            test_model(model_path)
        except:
            pass


test_latest_source()
