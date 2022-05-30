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

import os
import random
import sys
import pickle
import shlex
import shutil
import inspect
import unittest
import numpy as np
from collections import OrderedDict
from paddle.distributed.ps.utils.public import logger
from paddle.fluid.tests.unittests.distributed_passes.dist_pass_test_base import prepare_python_path_and_return_module, remove_path_if_exists
import paddle.distributed.fleet as fleet


class PsPassTestBase(unittest.TestCase):
    def init(self):
        self.config = {}
        self.config['ps_mode_config'] = ""
        self.config['worker_num'] = "1"
        self.config['server_num'] = "1"
        self.config['run_minimize'] = "0"
        self.config['run_single_pass'] = "0"
        self.config['run_the_one_ps'] = '0'
        self.config['debug_new_minimize'] = "0"
        self.config['debug_new_pass'] = "0"
        self.config['debug_the_one_ps'] = '0'
        self.config['log_dir'] = ""
        self.config['applied_pass_name'] = ""

    def setUp(self):
        print('Ps setUp...')

    def tearDown(self):
        print('Ps tearDown...')

    def ps_launch(self, ps_mode="cpu-ps"):
        if ps_mode == "cpu-ps" or ps_mode == 'heter-ps':
            os.environ['WITH_DISTRIBUTE'] = 'ON'

            cmd = [
                sys.executable,
                "-u",
            ] + [
                "-m", "launch", "--log_dir", self.config['log_dir'],
                "--worker_num", self.config['worker_num'], "--server_num",
                self.config['server_num']
            ]
            if ps_mode == 'heter-ps':
                os.environ['FLAGS_START_PORT'] = '12004'
                cmd += [
                    '--heter_worker_num', self.config['heter_worker_num'],
                    '--heter_devices', self.config['heter_devices']
                ]

            cmd += [
                "../ps/ps_dnn_trainer.py", "-m", self.config['ps_mode_config'],
                "--run_minimize", self.config['run_minimize'],
                "--run_single_pass", self.config['run_single_pass'],
                "--run_the_one_ps", self.config['run_the_one_ps'],
                "--debug_new_pass", self.config['debug_new_pass'],
                "--debug_new_minimize", self.config['debug_new_minimize'],
                "--applied_pass_name", self.config['applied_pass_name'],
                "--debug_the_one_ps", self.config['debug_the_one_ps']
            ]
        elif ps_mode == "gpu-ps":
            os.environ['FLAGS_LAUNCH_BARRIER'] = '0'
            os.environ['PADDLE_PSERVER_NUMS'] = '1'
            os.environ['PADDLE_TRAINERS_NUM'] = '1'
            os.environ['POD_IP'] = '127.0.0.1'
            os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:29011'
            os.environ['PADDLE_PORT'] = '29011'
            os.environ['FLAGS_selected_gpus'] = '0,1,2,3,4,5,6,7'
            # pserver
            # os.environ['TRAINING_ROLE'] = 'PSERVER'

            # trainer
            os.environ['TRAINING_ROLE'] = 'TRAINER'
            os.environ['PADDLE_TRAINER_ID'] = '0'

            cmd = [
                sys.executable, "-u", "../ps/ps_dnn_trainer.py", "-m",
                self.config['ps_mode_config'], "--run_minimize",
                self.config['run_minimize'], "--run_single_pass",
                self.config['run_single_pass'], "--run_the_one_ps",
                self.config['run_the_one_ps'], "--debug_new_pass",
                self.config['debug_new_pass'], "--debug_new_minimize",
                self.config['debug_new_minimize'], "--applied_pass_name",
                self.config['applied_pass_name'], "--debug_the_one_ps",
                self.config['debug_the_one_ps']
            ]

        cmd = [shlex.quote(c) for c in cmd]
        prepare_python_path_and_return_module(__file__)
        exitcode = os.system(' '.join(cmd))
