# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import os
import unittest
import numpy as np

import paddle
import paddle.fluid as fluid

import paddle
from ..distributed_passes.ps_pass_test_base import *
from paddle.distributed.ps.utils.public import logger, ps_log_root_dir
from paddle.fluid.tests.unittests.ps.ps_dnn_trainer import DnnTrainer


class TestTheOnePs(PsPassTestBase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def check(self):
        pass

    def test_ps_cpu_async(self):
        self.init()
        self.config['ps_mode_config'] = "../ps/cpu_async_ps_config.yaml"
        self.config['run_the_one_ps'] = '1'

        self.config['debug_the_one_ps'] = '0'
        self.config[
            'log_dir'] = ps_log_root_dir + "async_cpu_log_old_the_one_ps"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()
        '''
        self.config['debug_the_one_ps'] = '1'
        self.config['log_dir'] = ps_log_root_dir + "async_cpu_log_new_the_one_ps"
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()
        '''
        self.check()


if __name__ == '__main__':
    unittest.main()
