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

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker


class ConvBnFusePassExplicitPaddingTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=6,
                filter_size=6,
                groups=3,
                padding=[1, 1, 1, 1],
                bias_attr=False,
                act=None)
            bn_out = fluid.layers.batch_norm(conv_out, is_test=True)

        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [bn_out]

    def test_check_output(self):
        self.check_output()
        self.assertTrue(PassVersionChecker.IsCompatible('conv_bn_fuse_pass'))


class ConvBnFusePassValidPaddingTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=6,
                filter_size=6,
                groups=3,
                padding='VALID',
                bias_attr=False,
                act=None)
            bn_out = fluid.layers.batch_norm(conv_out, is_test=True)

        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [bn_out]

    def test_check_output(self):
        self.check_output()
        self.assertTrue(PassVersionChecker.IsCompatible('conv_bn_fuse_pass'))


class ConvBnFusePassSamePaddingTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=6,
                filter_size=6,
                groups=3,
                padding='SAME',
                bias_attr=False,
                act=None)
            bn_out = fluid.layers.batch_norm(conv_out, is_test=True)

        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [bn_out]

    def test_check_output(self):
        self.check_output()
        self.assertTrue(PassVersionChecker.IsCompatible('conv_bn_fuse_pass'))


class ConvEltwiseAddBnFuseExplicitPaddingPass(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=6,
                filter_size=6,
                groups=3,
                padding=[1, 1, 1, 1],
                bias_attr=None,
                act=None)
            bn_out = fluid.layers.batch_norm(conv_out, is_test=True)

        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [bn_out]

    def test_check_output(self):
        self.check_output()
        self.assertTrue(
            PassVersionChecker.IsCompatible('conv_eltwiseadd_bn_fuse_pass'))


class ConvEltwiseAddBnFuseValidPaddingPass(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=6,
                filter_size=6,
                groups=3,
                padding='VALID',
                bias_attr=None,
                act=None)
            bn_out = fluid.layers.batch_norm(conv_out, is_test=True)

        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [bn_out]

    def test_check_output(self):
        self.check_output()
        self.assertTrue(
            PassVersionChecker.IsCompatible('conv_eltwiseadd_bn_fuse_pass'))


class ConvEltwiseAddBnFuseSamePaddingPass(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=6,
                filter_size=6,
                groups=3,
                padding='SAME',
                bias_attr=None,
                act=None)
            bn_out = fluid.layers.batch_norm(conv_out, is_test=True)

        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [bn_out]

    def test_check_output(self):
        self.check_output()
        self.assertTrue(
            PassVersionChecker.IsCompatible('conv_eltwiseadd_bn_fuse_pass'))


if __name__ == "__main__":
    unittest.main()
