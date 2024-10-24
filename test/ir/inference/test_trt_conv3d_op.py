# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker


class TensorRTSubgraphPassConv3dTest(InferencePassTest):
    def setUp(self):
        self.init_params()
        self.set_params()
        with paddle.pir_utils.OldIrGuard():
            with base.program_guard(self.main_program, self.startup_program):
                data = paddle.static.data(
                    name="data", shape=[-1, 3, 6, 32, 32], dtype="float32"
                )
                conv_out = paddle.nn.Conv3D(
                    in_channels=3,
                    out_channels=self.conv_num_filters,
                    kernel_size=self.conv_filter_size,
                    groups=self.conv_groups,
                    stride=self.stride,
                    padding=self.conv_padding,
                    bias_attr=False,
                    data_format="NCDHW",
                )(data)
            self.feeds = {
                "data": np.random.random([1, 3, 6, 32, 32]).astype("float32"),
            }
            self.enable_trt = True
            self.trt_parameters = TensorRTSubgraphPassConv3dTest.TensorRTParam(
                1 << 30, 32, 1, self.precision, self.use_static, False
            )
        self.fetch_list = [conv_out]

    def init_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = [1, 1, 1]
        self.use_cudnn = True
        self.use_static = False
        self.precision = AnalysisConfig.Precision.Float32
        self.stride = 1

    def set_params(self):
        pass

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassConv3dValidPaddingTest(
    TensorRTSubgraphPassConv3dTest
):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = 'VALID'


class TensorRTSubgraphPassConv3dSamePaddingTest(TensorRTSubgraphPassConv3dTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = 'SAME'


class TensorRTSubgraphPassConv3dPaddingTest(TensorRTSubgraphPassConv3dTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = [2, 3, 3]


class TensorRTSubgraphPassConv3dStrideTest(TensorRTSubgraphPassConv3dTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = 'SAME'
        self.stride = [1, 2, 2]


class DynamicShapeTensorRTSubgraphPassConv3dTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with paddle.pir_utils.OldIrGuard():
            with base.program_guard(self.main_program, self.startup_program):
                data = paddle.static.data(
                    name="data", shape=[-1, 6, -1, -1, -1], dtype="float32"
                )
                conv_out = paddle.nn.Conv3D(
                    in_channels=6,
                    out_channels=self.conv_num_filters,
                    kernel_size=self.conv_filter_size,
                    groups=self.conv_groups,
                    stride=self.stride,
                    padding=self.conv_padding,
                    bias_attr=False,
                    data_format="NCDHW",
                )(data)
            self.feeds = {
                "data": np.random.random([1, 6, 32, 32, 8]).astype("float32"),
            }
            self.enable_trt = True
            self.trt_parameters = (
                DynamicShapeTensorRTSubgraphPassConv3dTest.TensorRTParam(
                    1 << 30,
                    32,
                    0,
                    AnalysisConfig.Precision.Float32,
                    False,
                    False,
                )
            )
            self.dynamic_shape_params = (
                DynamicShapeTensorRTSubgraphPassConv3dTest.DynamicShapeParam(
                    {
                        "data": [1, 6, 8, 8, 8],
                        "conv3d_0.tmp_0": [1, 6, 8, 8, 4],
                    },
                    {
                        "data": [32, 6, 32, 32, 8],
                        "conv3d_0.tmp_0": [32, 6, 32, 32, 8],
                    },
                    {
                        "data": [16, 6, 16, 16, 8],
                        "conv3d_0.tmp_0": [16, 6, 16, 16, 8],
                    },
                    False,
                )
            )
        self.fetch_list = [conv_out]

    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 6
        self.conv_padding = 'SAME'
        self.use_cudnn = True
        self.stride = [2, 2, 2]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


if __name__ == "__main__":
    unittest.main()
