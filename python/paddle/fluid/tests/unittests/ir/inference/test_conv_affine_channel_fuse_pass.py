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


class ConvAffineChannelFusePassExplicitPaddingTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                groups=3,
                padding=[1, 1, 1, 1],
                bias_attr=False,
                act=None)
            input_scale = fluid.layers.create_parameter(
                shape=[3], dtype="float32")
            input_bias = fluid.layers.create_parameter(
                shape=[3], dtype="float32")
            ac_out = fluid.layers.affine_channel(
                x=conv_out, scale=input_scale, bias=input_bias)

        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [ac_out]

    def test_check_output(self):
        self.check_output()

        self.assertTrue(
            PassVersionChecker.IsCompatible('conv_affine_channel_fuse_pass'))


class ConvAffineChannelFusePassValidPaddingTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                groups=3,
                padding='VALID',
                bias_attr=False,
                act=None)
            input_scale = fluid.layers.create_parameter(
                shape=[3], dtype="float32")
            input_bias = fluid.layers.create_parameter(
                shape=[3], dtype="float32")
            ac_out = fluid.layers.affine_channel(
                x=conv_out, scale=input_scale, bias=input_bias)

        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [ac_out]

    def test_check_output(self):
        self.check_output()

        self.assertTrue(
            PassVersionChecker.IsCompatible('conv_affine_channel_fuse_pass'))


class ConvAffineChannelFusePassSamePaddingTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                groups=3,
                padding='SAME',
                bias_attr=False,
                act=None)
            input_scale = fluid.layers.create_parameter(
                shape=[3], dtype="float32")
            input_bias = fluid.layers.create_parameter(
                shape=[3], dtype="float32")
            ac_out = fluid.layers.affine_channel(
                x=conv_out, scale=input_scale, bias=input_bias)

        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [ac_out]

    def test_check_output(self):
        self.check_output()

        self.assertTrue(
            PassVersionChecker.IsCompatible('conv_affine_channel_fuse_pass'))


class ConvEltwiseAddAffineChannelFusePassExplicitPaddingTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                groups=3,
                padding=[1, 1, 1, 1],
                bias_attr=param_attr,
                act=None)
            input_scale = fluid.layers.create_parameter(
                shape=[3], dtype="float32")
            input_bias = fluid.layers.create_parameter(
                shape=[3], dtype="float32")
            ac_out = fluid.layers.affine_channel(
                x=conv_out, scale=input_scale, bias=input_bias)

        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [ac_out]

    def test_check_output(self):
        self.check_output()

        self.assertTrue(
            PassVersionChecker.IsCompatible(
                'conv_eltwiseadd_affine_channel_fuse_pass'))


class ConvEltwiseAddAffineChannelFusePassValidPaddingTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                groups=3,
                padding='VALID',
                bias_attr=param_attr,
                act=None)
            input_scale = fluid.layers.create_parameter(
                shape=[3], dtype="float32")
            input_bias = fluid.layers.create_parameter(
                shape=[3], dtype="float32")
            ac_out = fluid.layers.affine_channel(
                x=conv_out, scale=input_scale, bias=input_bias)

        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [ac_out]

    def test_check_output(self):
        self.check_output()

        self.assertTrue(
            PassVersionChecker.IsCompatible(
                'conv_eltwiseadd_affine_channel_fuse_pass'))


class ConvEltwiseAddAffineChannelFusePassSamePaddingTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                groups=3,
                padding='Same',
                bias_attr=param_attr,
                act=None)
            input_scale = fluid.layers.create_parameter(
                shape=[3], dtype="float32")
            input_bias = fluid.layers.create_parameter(
                shape=[3], dtype="float32")
            ac_out = fluid.layers.affine_channel(
                x=conv_out, scale=input_scale, bias=input_bias)

        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [ac_out]

    def test_check_output(self):
        self.check_output()

        self.assertTrue(
            PassVersionChecker.IsCompatible(
                'conv_eltwiseadd_affine_channel_fuse_pass'))


if __name__ == "__main__":
    unittest.main()
