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

import os
import shutil
import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TensorRTSubgraphPassConvTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 64, 64], dtype="float32")
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=self.conv_num_filters,
                filter_size=self.conv_filter_size,
                groups=self.conv_groups,
                padding=self.conv_padding,
                bias_attr=False,
                act=None)
        self.feeds = {
            "data": np.random.random([1, 6, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassConvTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [conv_out]

    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = [1, 1]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTSubgraphPassConvValidPaddingTest(TensorRTSubgraphPassConvTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = 'VALID'


class TensorRTSubgraphPassConvSamePaddingTest(InferencePassTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = 'SAME'


class TensorRTSubgraphPassDepthwiseConvTest(TensorRTSubgraphPassConvTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 6
        self.conv_padding = [1, 1]


class TensorRTSubgraphPassConvTransposeTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 64, 64], dtype="float32")
            conv_out = fluid.layers.conv2d_transpose(
                input=data,
                num_filters=self.conv_num_filters,
                filter_size=self.conv_filter_size,
                groups=self.conv_groups,
                padding=self.conv_padding,
                bias_attr=False,
                act=None)
        self.feeds = {
            "data": np.random.random([1, 6, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassConvTransposeTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [conv_out]

    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = [1, 1]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTSubgraphPassConvTransposeValidPaddingTest(
        TensorRTSubgraphPassConvTransposeTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = 'VALID'


class TensorRTSubgraphPassConvTransposeSamePaddingTest(
        TensorRTSubgraphPassConvTransposeTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = 'SAME'


class TensorRTSubgraphPassDepthwiseConvTransposeTest(
        TensorRTSubgraphPassConvTransposeTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = [1, 1]


if __name__ == "__main__":
    unittest.main()
