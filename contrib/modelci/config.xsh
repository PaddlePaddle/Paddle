#!/usr/bin/env xonsh
from utils import *

workspace = $(pwd).strip()

repo_url = 'https://github.com/PaddlePaddle/Paddle.git'
whl_url = ''

local_repo_path = pjoin(workspace, 'source_code')

models_path = pjoin(workspace, 'models')

tmp_root = pjoin(workspace, "tmp")


diff_thre = 0.1

default_compiled_whl_path = '/usr/local/opt/paddle/share/wheels/paddlepaddle_gpu-0.11.1a1-cp27-cp27mu-linux_x86_64.whl'

whl_path = pjoin(tmp_root, os.path.basename(default_compiled_whl_path))

mkdir -p @(tmp_root)