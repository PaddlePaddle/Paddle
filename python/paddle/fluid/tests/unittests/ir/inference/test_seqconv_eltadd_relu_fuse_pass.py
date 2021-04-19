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

from __future__ import print_function

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import PassVersionChecker


class SeqconvEltaddReluFusePassTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[100, 100], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.sequence_conv(
                input=data,
                num_filters=16,
                filter_size=4,
                padding_start=0,
                act="relu",
                bias_attr=param_attr)

        np_data = np.random.random((80, 100)).astype('float32')
        x_lod_tensor = fluid.create_lod_tensor(np_data, [[10, 20, 30, 20]],
                                               fluid.CPUPlace())
        self.feeds = {"data": x_lod_tensor}
        self.fetch_list = [conv_out]
        self.enable_mkldnn = True

    def test_check_output(self):
        self.check_output()
        self.assertTrue(
            PassVersionChecker.IsCompatible('seqconv_eltadd_relu_fuse_pass'))


class SeqconvEltaddReluFusePassTestPaddingStartPositive(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[-1, 4], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.sequence_conv(
                input=data,
                num_filters=16,
                filter_size=3,
                padding_start=2,
                act="relu",
                bias_attr=param_attr)

        np_data = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                            [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7]]).astype('float32')
        x_lod_tensor = fluid.create_lod_tensor(np_data, [[5, 2]],
                                               fluid.CPUPlace())
        self.feeds = {"data": x_lod_tensor}
        self.fetch_list = [conv_out]
        self.enable_mkldnn = True

    def test_check_output(self):
        self.check_output()
        self.assertTrue(
            PassVersionChecker.IsCompatible('seqconv_eltadd_relu_fuse_pass'))


class SeqconvEltaddReluFusePassTestPaddingStartNegative(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[100, 100], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.sequence_conv(
                input=data,
                num_filters=16,
                filter_size=4,
                padding_start=-1,
                act="relu",
                bias_attr=param_attr)

        np_data = np.random.random((80, 100)).astype('float32')
        x_lod_tensor = fluid.create_lod_tensor(np_data, [[10, 20, 30, 20]],
                                               fluid.CPUPlace())
        self.feeds = {"data": x_lod_tensor}
        self.fetch_list = [conv_out]
        self.enable_mkldnn = True

    def test_check_output(self):
        self.check_output()
        self.assertTrue(
            PassVersionChecker.IsCompatible('seqconv_eltadd_relu_fuse_pass'))


class SeqconvEltaddReluFusePassTestPaddingStartNone(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[100, 100], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.sequence_conv(
                input=data,
                num_filters=16,
                filter_size=4,
                act="relu",
                bias_attr=param_attr)

        np_data = np.random.random((80, 100)).astype('float32')
        x_lod_tensor = fluid.create_lod_tensor(np_data, [[10, 20, 30, 20]],
                                               fluid.CPUPlace())
        self.feeds = {"data": x_lod_tensor}
        self.fetch_list = [conv_out]
        self.enable_mkldnn = True

    def test_check_output(self):
        self.check_output()
        self.assertTrue(
            PassVersionChecker.IsCompatible('seqconv_eltadd_relu_fuse_pass'))


if __name__ == "__main__":
    unittest.main()
