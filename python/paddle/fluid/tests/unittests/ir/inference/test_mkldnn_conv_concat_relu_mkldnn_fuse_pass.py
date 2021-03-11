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


class ConvConcatReluMkldnnFusePassTest_0(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data_1 = fluid.data(
                name="data_1", shape=[-1, 3, 100, 100], dtype="float32")
            data_2 = fluid.data(
                name="data_2", shape=[-1, 3, 100, 100], dtype="float32")
            conv_1 = fluid.layers.conv2d(
                data_1,
                num_filters=self.conv1_num_filters,
                filter_size=self.conv1_filter_size,
                padding=self.conv1_padding,
                bias_attr=self.conv1_bias_attr)
            conv_2 = fluid.layers.conv2d(
                data_2,
                num_filters=self.conv2_num_filters,
                filter_size=self.conv2_filter_size,
                padding=self.conv2_padding,
                bias_attr=self.conv2_bias_attr)
            concat = fluid.layers.concat(
                [conv_1, conv_2], axis=self.concat_axis)
            out = fluid.layers.relu(concat)

        self.feeds = {
            "data_1": np.random.random((1, 3, 100, 100)).astype("float32"),
            "data_2": np.random.random((1, 3, 100, 100)).astype("float32")
        }
        self.fetch_list = [out]
        self.enable_mkldnn = True

    def set_params(self):
        self.conv1_num_filters = 3
        self.conv1_filter_size = 3
        self.conv1_padding = 0
        self.conv1_bias_attr = False
        self.conv2_num_filters = 3
        self.conv2_filter_size = 3
        self.conv2_padding = 0
        self.conv2_bias_attr = False
        self.concat_axis = 0
        self.pass_name = "conv_concat_relu_mkldnn_fuse_pass"

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)

    def test_pass_compatible(self):
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class ConvConcatReluMkldnnFusePassTest_1(ConvConcatReluMkldnnFusePassTest_0):
    def set_params(self):
        self.conv1_num_filters = 3
        self.conv1_filter_size = 3
        self.conv1_padding = 0
        self.conv1_bias_attr = False
        self.conv2_num_filters = 5
        self.conv2_filter_size = 5
        self.conv2_padding = 1
        self.conv2_bias_attr = True
        self.concat_axis = 1
        self.pass_name = "conv_concat_relu_mkldnn_fuse_pass"


if __name__ == "__main__":
    unittest.main()
