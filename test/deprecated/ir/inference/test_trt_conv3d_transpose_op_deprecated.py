# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


class TensorRTSubgraphPassConv3dTransposeTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[-1, 4, 4, 32, 32], dtype="float32"
            )
            conv_out = paddle.nn.Conv3DTranspose(
                in_channels=4,
                out_channels=self.conv_num_filters,
                kernel_size=self.conv_filter_size,
                groups=self.conv_groups,
                padding=self.conv_padding,
                bias_attr=False,
                stride=1,
                data_format="NCDHW",
            )(data)
        self.feeds = {
            "data": np.random.random([1, 4, 4, 32, 32]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = (
            TensorRTSubgraphPassConv3dTransposeTest.TensorRTParam(
                1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False
            )
        )
        self.fetch_list = [conv_out]

    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = [1, 1, 1]
        self.use_cudnn = True

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassConv3dTransposeSamePaddingTest(
    TensorRTSubgraphPassConv3dTransposeTest
):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = 'VALID'
        self.use_cudnn = True


class TensorRTSubgraphPassConv3dTransposeMultigroupTest(
    TensorRTSubgraphPassConv3dTransposeTest
):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 2
        self.conv_padding = 'VALID'
        self.use_cudnn = True


class DynamicShapeTensorRTSubgraphPassConv3dTransposeTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[-1, 6, -1, -1, -1], dtype="float32"
            )
            conv_out = paddle.nn.Conv3DTranspose(
                in_channels=6,
                out_channels=self.conv_num_filters,
                kernel_size=self.conv_filter_size,
                groups=self.conv_groups,
                padding=self.conv_padding,
                bias_attr=False,
                stride=self.stride,
                data_format="NCDHW",
            )(data)
        self.feeds = {
            "data": np.random.random([1, 6, 32, 32, 8]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = (
            DynamicShapeTensorRTSubgraphPassConv3dTransposeTest.TensorRTParam(
                1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
            )
        )
        self.dynamic_shape_params = DynamicShapeTensorRTSubgraphPassConv3dTransposeTest.DynamicShapeParam(
            {
                "data": [1, 6, 8, 8, 8],
                "conv3d_transpose_0.tmp_0": [1, 6, 8, 8, 1],
            },
            {
                "data": [32, 6, 32, 32, 8],
                "conv3d_transpose_0.tmp_0": [32, 6, 64, 64, 16],
            },
            {
                "data": [16, 6, 16, 16, 8],
                "conv3d_transpose_0.tmp_0": [16, 6, 16, 16, 8],
            },
            False,
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
