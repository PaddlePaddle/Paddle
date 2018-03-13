#!/usr/bin/env xonsh
from utils import *
import config

def get_whl():
    download(config.whl_url, config.whl_path)

def install_whl():
    cd @(config.tmp_root)
    pip install --upgrade @(config.whl_path)

def compile():
    cd @(config.local_repo_path)
    mkdir -p build
    cd build
    cmake .. -DCUDNN_ROOT=/usr -DCUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu
    make install -j10
    # TODO save the installed whl to paddle.whl
    tmp_whl = None
    cp @(config.default_compiled_whl_path) @(config.whl_path)
