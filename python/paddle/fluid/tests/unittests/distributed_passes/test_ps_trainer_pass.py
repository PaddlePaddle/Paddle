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


class TestPsTrainerPass(PsPassTestBase):
    def init(self, pass_name):
        self.pass_name = pass_name

    def setUp(self):
        print('setUp...')

    def tearDown(self):
        print('tearDown...')

    def check_rename_program_file(self):
        pass

    def check(self):
        pass

    def test_append_send_ops_pass(self):
        self.init("")
        self.debug_pass = 0
        self.ps_launch()

        self.init("append_send_ops_pass")
        self.debug_pass = 1
        self.ps_launch()

        self.check()

    def test_distributed_ops_pass(self):
        pass


if __name__ == '__main__':
    unittest.main()
