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
from ps_pass_test_base import PsPassTestBase
from paddle.fluid.tests.unittests.ps.ps_dnn_trainer import DnnTrainer


class TestPsTrainerPass(PsPassTestBase):
    def init(self):
        self.worker_num = 2
        self.server_num = 2
        self.run_minimize = 0
        self.run_single_pass = 0
        self.debug_new_minimize = 0
        self.debug_new_pass = 0
        self.applied_pass_name = ""
        self.log_dir = ""

    def setUp(self):
        print('setUp...')

    def tearDown(self):
        print('tearDown...')

    def check(self):
        pass

    def test_ps_optimizer_minimize(self):
        self.run_minimize = 1

        self.debug_new_minimize = 0
        self.log_dir = "log_old"
        self.ps_launch()

        self.debug_new_minimize = 1
        self.log_dir = "log_new"
        self.ps_launch()

        self.check()

    def test_append_send_ops_pass(self):
        self.run_single_pass = 1

        self.debug_new_pass = 0
        self.log_dir = "log_old"
        self.ps_launch()

        self.debug_new_pass = 1
        self.applied_pass_name = "append_send_ops_pass"
        self.log_dir = "log_new"
        self.ps_launch()

        self.check()

    def test_distributed_ops_pass(self):
        pass


if __name__ == '__main__':
    unittest.main()
