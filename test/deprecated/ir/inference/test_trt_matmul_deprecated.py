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

import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.static import nn


class TensorRTMatMulDims2Test(InferencePassTest):
    def setUp(self):
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[24, 24], dtype="float32"
            )
            matmul_out = paddle.matmul(
                x=data,
                y=data,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
            )
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            out = nn.batch_norm(matmul_out, is_test=True)

        self.feeds = {
            "data": np.ones([24, 24]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulDims2Test.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = TensorRTMatMulDims2Test.DynamicShapeParam(
            {'data': [1, 24]},
            {'data': [32, 24]},
            {'data': [24, 24]},
            False,
        )
        self.fetch_list = [out]

    def set_params(self):
        self.transpose_x = True
        self.transpose_y = True
        self.alpha = 2.0

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTMatMulTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[-1, 6, 24, 24], dtype="float32"
            )
            matmul_out = paddle.matmul(
                x=data,
                y=data,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
            )
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            out = nn.batch_norm(matmul_out, is_test=True)

        self.feeds = {
            "data": np.ones([1, 6, 24, 24]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = TensorRTMatMulTest.DynamicShapeParam(
            {'data': [1, 6, 24, 24]},
            {'data': [32, 6, 24, 24]},
            {'data': [1, 6, 24, 24]},
            False,
        )
        self.fetch_list = [out]

    def set_params(self):
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 1.0

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTMatMulTransposeXTest(TensorRTMatMulTest):
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = False
        self.alpha = 1.0


class TensorRTMatMulTransposeYTest(TensorRTMatMulTest):
    def set_params(self):
        self.transpose_x = False
        self.transpose_y = True
        self.alpha = 1.0


class TensorRTMatMulScaleTest(TensorRTMatMulTest):
    def set_params(self):
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 2.0


class TensorRTMatMulBroadcastTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        place = base.CPUPlace()
        with base.program_guard(self.main_program, self.startup_program):
            data_x = paddle.static.data(
                name="data_x", shape=[-1, 6, 24], dtype="float32"
            )
            data_y = paddle.static.data(
                name="data_y", shape=[24, 16], dtype="float32"
            )
            matmul_out = paddle.matmul(
                x=data_x,
                y=data_y,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
            )
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            out = nn.batch_norm(matmul_out, is_test=True)

        self.feeds = {
            "data_x": np.ones([2, 6, 24]).astype("float32"),
            "data_y": np.ones([24, 16]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulBroadcastTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = (
            TensorRTMatMulBroadcastTest.DynamicShapeParam(
                {'data_x': [1, 6, 24], 'data_y': [24, 16]},
                {'data_x': [32, 6, 24], 'data_y': [24, 16]},
                {'data_x': [2, 6, 24], 'data_y': [24, 16]},
                False,
            )
        )
        self.fetch_list = [out]

    def set_params(self):
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 1.0

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


@unittest.skipIf(
    not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core does not support bfloat16",
)
class TensorRTMatMulBroadcastBF16Test(InferencePassTest):
    def setUp(self):
        self.set_params()
        place = base.CPUPlace()
        with base.program_guard(self.main_program, self.startup_program):
            data_x = paddle.static.data(
                name="data_x", shape=[-1, 6, 24], dtype="float32"
            )
            data_y = paddle.static.data(
                name="data_y", shape=[24, 16], dtype="float32"
            )
            matmul_out = paddle.matmul(
                x=data_x,
                y=data_y,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
            )
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            out = nn.batch_norm(matmul_out, is_test=True)

        self.feeds = {
            "data_x": np.ones([2, 6, 24]).astype("float32"),
            "data_y": np.ones([24, 16]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulBroadcastTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Bfloat16, False, False
        )
        self.dynamic_shape_params = (
            TensorRTMatMulBroadcastTest.DynamicShapeParam(
                {'data_x': [1, 6, 24], 'data_y': [24, 16]},
                {'data_x': [32, 6, 24], 'data_y': [24, 16]},
                {'data_x': [2, 6, 24], 'data_y': [24, 16]},
                False,
            )
        )
        self.fetch_list = [out]

    def set_params(self):
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 1.0

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


if __name__ == "__main__":
    unittest.main()
