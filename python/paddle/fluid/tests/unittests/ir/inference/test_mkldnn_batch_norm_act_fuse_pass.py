# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Test for fusion of batch norm and activation."""
from __future__ import print_function

import unittest
import numpy as np

import paddle.fluid as fluid
from inference_pass_test import InferencePassTest
from paddle import enable_static
from paddle.fluid.core import PassVersionChecker

enable_static()


class BnReluOneDnnFusePassTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 100, 100], dtype="float32")
            bn_out = fluid.layers.batch_norm(
                input=data, is_test=True, use_global_stats=self.global_stats)
            relu_out = fluid.layers.relu(bn_out)

        self.feeds = {
            "data": np.random.random((1, 3, 100, 100)).astype("float32")
        }
        self.fetch_list = [relu_out]
        self.enable_mkldnn = True

    def set_params(self):
        self.global_stats = False
        self.pass_name = "batch_norm_act_fuse_pass"

    def test_check_output(self):
        self.check_output()
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class BnReluGlobalStatsOneDnnFusePassTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 100, 100], dtype="float32")
            bn_out = fluid.layers.batch_norm(
                input=data, is_test=True, use_global_stats=self.global_stats)
            relu_out = fluid.layers.relu(bn_out)

        self.feeds = {
            "data": np.random.random((1, 3, 100, 100)).astype("float32")
        }
        self.fetch_list = [relu_out]
        self.enable_mkldnn = True

    def set_params(self):
        self.global_stats = True
        self.pass_name = "batch_norm_act_fuse_pass"

    def test_check_output(self):
        self.check_output()
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


if __name__ == "__main__":
    unittest.main()
