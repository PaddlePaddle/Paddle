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
"""Test for fusion of fc and activation."""
from __future__ import print_function

import unittest
import numpy as np

import paddle.fluid as fluid
from inference_pass_test import InferencePassTest
from paddle import enable_static
from paddle.fluid.core import PassVersionChecker

enable_static()


class FCGeluTanhOneDnnFusePassTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 128, 768], dtype="float32")
            fc_out = fluid.layers.fc(input=data, size=3072, num_flatten_dims=2)
            gelu_out = fluid.layers.gelu(fc_out, approximate=False)

        self.feeds = {"data": np.random.random((1, 128, 768)).astype("float32")}

        self.fetch_list = [gelu_out]
        self.enable_mkldnn = True

    def set_params(self):
        self.pass_name = "fc_act_mkldnn_fuse_pass"

    def test_check_output(self):
        self.check_output()


class FCGeluErfOneDnnFusePassTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 128, 768], dtype="float32")
            fc_out = fluid.layers.fc(input=data, size=3072, num_flatten_dims=2)
            gelu_out = fluid.layers.gelu(fc_out, approximate=True)

        self.feeds = {"data": np.random.random((1, 128, 768)).astype("float32")}

        self.fetch_list = [gelu_out]
        self.enable_mkldnn = True

    def set_params(self):
        self.pass_name = "fc_act_mkldnn_fuse_pass"

    def test_check_output(self):
        self.check_output()
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class FCTanhOneDnnFusePassTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 128, 768], dtype="float32")
            fc_out = fluid.layers.fc(input=data, size=3072, num_flatten_dims=2)
            tanh_out = fluid.layers.tanh(fc_out)

        self.feeds = {"data": np.random.random((1, 128, 768)).astype("float32")}

        self.fetch_list = [tanh_out]
        self.enable_mkldnn = True

    def set_params(self):
        self.pass_name = "fc_act_mkldnn_fuse_pass"

    def test_check_output(self):
        self.check_output()
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class FCSigmoidOneDnnFusePassTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 128, 768], dtype="float32")
            fc_out = fluid.layers.fc(input=data, size=3072, num_flatten_dims=2)
            sigmoid_out = fluid.layers.sigmoid(fc_out)

        self.feeds = {"data": np.random.random((1, 128, 768)).astype("float32")}

        self.fetch_list = [sigmoid_out]
        self.enable_mkldnn = True

    def set_params(self):
        self.pass_name = "fc_act_mkldnn_fuse_pass"

    def test_check_output(self):
        self.check_output()
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class FCHardSwishOneDnnFusePassTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 128, 768], dtype="float32")
            fc_out = fluid.layers.fc(input=data, size=3072, num_flatten_dims=2)
            hardswish_out = fluid.layers.hard_swish(fc_out)

        self.feeds = {"data": np.random.random((1, 128, 768)).astype("float32")}

        self.fetch_list = [hardswish_out]
        self.enable_mkldnn = True

    def set_params(self):
        self.pass_name = "fc_act_mkldnn_fuse_pass"

    def test_check_output(self):
        self.check_output()
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class FCMishOneDnnFusePassTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 128, 768], dtype="float32")
            fc_out = fluid.layers.fc(input=data, size=3072, num_flatten_dims=2)
            mish_out = fluid.layers.mish(fc_out)

        self.feeds = {"data": np.random.random((1, 128, 768)).astype("float32")}

        self.fetch_list = [mish_out]
        self.enable_mkldnn = True

    def set_params(self):
        self.pass_name = "fc_act_mkldnn_fuse_pass"

    def test_check_output(self):
        self.check_output()
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


if __name__ == "__main__":
    unittest.main()
