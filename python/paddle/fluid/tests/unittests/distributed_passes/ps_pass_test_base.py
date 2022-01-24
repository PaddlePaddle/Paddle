# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import os
import random
import sys
import pickle
import shlex
import shutil
import inspect
import numpy as np
from collections import OrderedDict
from .dist_pass_test_base import prepare_python_path_and_return_module
import paddle.distributed.fleet as fleet


class PsPassTestBase(unittest.TestCase):
    def init(self):
        self.worker_num = 2
        self.server_num = 2
        self.debug_pass = True

    def setUp(self):
        print('Ps setUp...')

    def tearDown(self):
        print('Ps tearDown...')

    def ps_launch(self):
        cmd = [
            sys.executable,
            "-u",
        ] + coverage_args + [
            "-m", "launch", "--worker_num", self.worker_num, "--server_num",
            self.server_num, "../ps/ps_dnn_trainer.py", "-m", "benchmark.yaml",
            "--debug_pass", self.debug_pass
        ]
        cmd = [shlex.quote(c) for c in cmd]
        prepare_python_path_and_return_module(__file__)
        exitcode = os.system(' '.join(cmd))
