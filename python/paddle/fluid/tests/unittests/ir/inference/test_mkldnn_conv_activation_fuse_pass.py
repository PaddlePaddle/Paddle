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


class ConvActivationMkldnnFusePassTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 100, 100], dtype="float32")
            conv_out = fluid.layers.conv2d(
                data,
                num_filters=self.conv_num_filters,
                filter_size=self.conv_filter_size,
                bias_attr=self.conv_bias_attr,
                act=self.act)

        self.feeds = {
            "data": np.random.random((1, 3, 100, 100)).astype("float32")
        }
        self.fetch_list = [conv_out]
        self.enable_mkldnn = True

    def set_params(self):
        self.conv_num_filters = 3
        self.conv_filter_size = 3
        self.conv_bias_attr = False
        self.act = "relu"
        self.pass_name = 'conv_relu_mkldnn_fuse_pass'

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)

    def test_pass_compatible(self):
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class ConvActivationMkldnnFusePassTest_1(ConvActivationMkldnnFusePassTest):
    def set_params(self):
        self.conv_num_filters = 5
        self.conv_filter_size = 5
        self.conv_bias_attr = True
        self.act = "relu"
        self.pass_name = 'conv_relu_mkldnn_fuse_pass'


class ConvActivationMkldnnFusePassTest_2(ConvActivationMkldnnFusePassTest):
    def set_params(self):
        self.conv_num_filters = 3
        self.conv_filter_size = 3
        self.conv_bias_attr = False
        self.act = "leaky_relu"
        self.pass_name = 'conv_leaky_relu_mkldnn_fuse_pass'


class ConvActivationMkldnnFusePassTest_3(ConvActivationMkldnnFusePassTest):
    def set_params(self):
        self.conv_num_filters = 5
        self.conv_filter_size = 5
        self.conv_bias_attr = True
        self.act = "leaky_relu"
        self.pass_name = 'conv_leaky_relu_mkldnn_fuse_pass'


class ConvActivationMkldnnFusePassTest_4(ConvActivationMkldnnFusePassTest):
    def set_params(self):
        self.conv_num_filters = 3
        self.conv_filter_size = 3
        self.conv_bias_attr = False
        self.act = "relu6"
        self.pass_name = 'conv_relu6_mkldnn_fuse_pass'


class ConvActivationMkldnnFusePassTest_5(ConvActivationMkldnnFusePassTest):
    def set_params(self):
        self.conv_num_filters = 5
        self.conv_filter_size = 5
        self.conv_bias_attr = True
        self.act = "hard_swish"
        self.pass_name = 'conv_hard_swish_mkldnn_fuse_pass'


class ConvActivationMkldnnFusePassTest_6(ConvActivationMkldnnFusePassTest):
    def set_params(self):
        self.conv_num_filters = 5
        self.conv_filter_size = 5
        self.conv_bias_attr = True
        self.act = "mish"
        self.pass_name = 'conv_mish_mkldnn_fuse_pass'


class ConvHardSigmoidOneDNNFusePassTest(ConvActivationMkldnnFusePassTest):
    def set_params(self):
        self.conv_num_filters = 5
        self.conv_filter_size = 5
        self.conv_bias_attr = True
        self.act = "hard_sigmoid"
        self.pass_name = 'conv_hard_sigmoid_mkldnn_fuse_pass'


if __name__ == "__main__":
    unittest.main()
