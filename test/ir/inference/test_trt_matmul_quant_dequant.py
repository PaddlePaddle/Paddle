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
from quant_dequant_test import QuantDequantTest

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker


class TensorRTMatMulQuantDequantDims3Test(QuantDequantTest):
    def setUp(self):
        self.set_params()

        def network():
            self.data = paddle.static.data(
                name='data', shape=[1, 28, 28], dtype='float32'
            )
            self.label = paddle.static.data(
                name='label', shape=[1, 1], dtype='int64'
            )
            matmul_out = paddle.matmul(
                x=self.data,
                y=self.data,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
            )
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            fc_out = paddle.static.nn.fc(
                x=matmul_out,
                size=10,
                num_flatten_dims=1,
                bias_attr=False,
                activation=None,
            )
            result = F.relu(fc_out)
            loss = paddle.nn.functional.cross_entropy(
                input=result,
                label=self.label,
                reduction='none',
                use_softmax=False,
            )
            avg_loss = paddle.mean(loss)
            return avg_loss, result

        paddle.seed(2)
        with base.unique_name.guard():
            with base.program_guard(self.main_program, self.startup_program):
                self.loss, result = network()
                opt = paddle.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with base.unique_name.guard():
            with base.program_guard(
                self.test_main_program, self.startup_program
            ):
                network()
        self.feeds = {"data": np.random.random([1, 28, 28]).astype("float32")}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulQuantDequantDims3Test.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False
        )
        self.dynamic_shape_params = (
            TensorRTMatMulQuantDequantDims3Test.DynamicShapeParam(
                {'data': [1, 28, 28]},
                {'data': [4, 28, 28]},
                {'data': [3, 28, 28]},
                False,
            )
        )
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def set_params(self):
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 1.0

    def test_check_output(self):
        # self.quant_dequant()
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(
                use_gpu, atol=1, flatten=False, rtol=1e-1
            )
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTMatMulQuantDequantDims3TransposeXTest(
    TensorRTMatMulQuantDequantDims3Test
):
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = False
        self.alpha = 2.1


class TensorRTMatMulQuantDequantDims3TransposeYTest(
    TensorRTMatMulQuantDequantDims3Test
):
    def set_params(self):
        self.transpose_x = False
        self.transpose_y = True
        self.alpha = 3.9


class TensorRTMatMulQuantDequantDims3TransposeXYTest(
    TensorRTMatMulQuantDequantDims3Test
):
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = True
        self.alpha = 8.4


class TensorRTMatMulQuantDequantDims4Test(QuantDequantTest):
    def setUp(self):
        self.set_params()

        def network():
            self.data = paddle.static.data(
                name='data', shape=[1, 28, 28], dtype='float32'
            )
            self.label = paddle.static.data(
                name='label', shape=[1, 1], dtype='int64'
            )
            reshape_out = paddle.reshape(self.data, shape=[0, 4, 14, 14])
            matmul_out = paddle.matmul(
                x=reshape_out,
                y=reshape_out,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
            )
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            out = paddle.static.nn.batch_norm(matmul_out, is_test=True)
            fc_out = paddle.static.nn.fc(
                x=matmul_out,
                size=10,
                num_flatten_dims=1,
                bias_attr=False,
                activation=None,
            )
            result = F.relu(fc_out)
            loss = paddle.nn.functional.cross_entropy(
                input=result,
                label=self.label,
                reduction='none',
                use_softmax=False,
            )
            avg_loss = paddle.mean(loss)
            return avg_loss, result

        paddle.seed(2)
        with base.unique_name.guard():
            with base.program_guard(self.main_program, self.startup_program):
                self.loss, result = network()
                opt = paddle.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with base.unique_name.guard():
            with base.program_guard(
                self.test_main_program, self.startup_program
            ):
                network()
        self.feeds = {"data": np.random.random([1, 28, 28]).astype("float32")}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulQuantDequantDims4Test.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False
        )
        self.dynamic_shape_params = (
            TensorRTMatMulQuantDequantDims4Test.DynamicShapeParam(
                {'data': [1, 28, 28]},
                {'data': [4, 28, 28]},
                {'data': [3, 28, 28]},
                False,
            )
        )
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def set_params(self):
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 1.0

    def test_check_output(self):
        # self.quant_dequant()
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(
                use_gpu, atol=1, flatten=False, rtol=1e-1
            )
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTMatMulQuantDequantDims4TransposeXTest(
    TensorRTMatMulQuantDequantDims4Test
):
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = False
        self.alpha = 3.2


class TensorRTMatMulQuantDequantDims4TransposeYTest(
    TensorRTMatMulQuantDequantDims4Test
):
    def set_params(self):
        self.transpose_x = False
        self.transpose_y = True
        self.alpha = 7.5


class TensorRTMatMulQuantDequantDims4TransposeXYTest(
    TensorRTMatMulQuantDequantDims4Test
):
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = True
        self.alpha = 11.2


class TensorRTMatMulQuantDequantDims3DynamicTest(QuantDequantTest):
    def setUp(self):
        self.set_params()

        def network():
            self.data = paddle.static.data(
                name='data', shape=[-1, 28, 28], dtype='float32'
            )
            self.label = paddle.static.data(
                name='label', shape=[1, 1], dtype='int64'
            )
            matmul_out = paddle.matmul(
                x=self.data,
                y=self.data,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
            )
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            out = paddle.static.nn.batch_norm(matmul_out, is_test=True)
            fc_out = paddle.static.nn.fc(
                x=matmul_out,
                size=10,
                num_flatten_dims=1,
                bias_attr=False,
                activation=None,
            )
            result = F.relu(fc_out)
            loss = paddle.nn.functional.cross_entropy(
                input=result,
                label=self.label,
                reduction='none',
                use_softmax=False,
            )
            avg_loss = paddle.mean(loss)
            return avg_loss, result

        paddle.seed(2)
        with base.unique_name.guard():
            with base.program_guard(self.main_program, self.startup_program):
                self.loss, result = network()
                opt = paddle.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with base.unique_name.guard():
            with base.program_guard(
                self.test_main_program, self.startup_program
            ):
                network()
        self.feeds = {"data": np.random.random([3, 28, 28]).astype("float32")}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = (
            TensorRTMatMulQuantDequantDims3DynamicTest.TensorRTParam(
                1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False
            )
        )
        self.dynamic_shape_params = (
            TensorRTMatMulQuantDequantDims3DynamicTest.DynamicShapeParam(
                {'data': [1, 28, 28]},
                {'data': [4, 28, 28]},
                {'data': [3, 28, 28]},
                False,
            )
        )
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def set_params(self):
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 1.0

    def test_check_output(self):
        # self.quant_dequant()
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(
                use_gpu, atol=1, flatten=False, rtol=1e-1
            )
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTMatMulQuantDequantDims4TransposeXDynamicTest(
    TensorRTMatMulQuantDequantDims3DynamicTest
):
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = False
        self.alpha = 2.0


class TensorRTMatMulQuantDequantDims4TransposeYDynamicTest(
    TensorRTMatMulQuantDequantDims3DynamicTest
):
    def set_params(self):
        self.transpose_x = False
        self.transpose_y = True
        self.alpha = 2.2


class TensorRTMatMulQuantDequantDims4TransposeXYDynamicTest(
    TensorRTMatMulQuantDequantDims3DynamicTest
):
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = True
        self.alpha = 7.8


if __name__ == "__main__":
    unittest.main()
