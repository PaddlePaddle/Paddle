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
<<<<<<< HEAD

import numpy as np
from quant_dequant_test import QuantDequantTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.nn.functional as F
from paddle.fluid.core import AnalysisConfig, PassVersionChecker


class TensorRTMatMulQuantDequantDims3Test(QuantDequantTest):
=======
import numpy as np
from inference_pass_test import InferencePassTest
from quant_dequant_test import QuantDequantTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TensorRTMatMulQuantDequantDims3Test(QuantDequantTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_params()

        def network():
<<<<<<< HEAD
            self.data = fluid.data(
                name='data', shape=[1, 28, 28], dtype='float32'
            )
            self.label = fluid.data(name='label', shape=[1, 1], dtype='int64')
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
=======
            self.data = fluid.data(name='data',
                                   shape=[1, 28, 28],
                                   dtype='float32')
            self.label = fluid.data(name='label', shape=[1, 1], dtype='int64')
            matmul_out = fluid.layers.matmul(x=self.data,
                                             y=self.data,
                                             transpose_x=self.transpose_x,
                                             transpose_y=self.transpose_y,
                                             alpha=self.alpha)
            fc_out = fluid.layers.fc(input=matmul_out,
                                     size=10,
                                     num_flatten_dims=1,
                                     bias_attr=False,
                                     act=None)
            result = fluid.layers.relu(fc_out)
            loss = fluid.layers.cross_entropy(input=result, label=self.label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            avg_loss = paddle.mean(loss)
            return avg_loss, result

        self.main_program.random_seed = 2
        self.startup_program.random_seed = 2
        self.test_main_program.random_seed = 2
<<<<<<< HEAD
        # self.test_startup_program.random_seed = 2
=======
        #self.test_startup_program.random_seed = 2
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        with fluid.unique_name.guard():
            with fluid.program_guard(self.main_program, self.startup_program):
                self.loss, result = network()
                opt = fluid.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with fluid.unique_name.guard():
<<<<<<< HEAD
            with fluid.program_guard(
                self.test_main_program, self.startup_program
            ):
=======
            with fluid.program_guard(self.test_main_program,
                                     self.startup_program):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                network()
        self.feeds = {"data": np.random.random([1, 28, 28]).astype("float32")}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulQuantDequantDims3Test.TensorRTParam(
<<<<<<< HEAD
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False
        )
=======
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def set_params(self):
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 1.0

    def test_check_output(self):
<<<<<<< HEAD
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
=======
        #self.quant_dequant()
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu,
                                          atol=1,
                                          flatten=False,
                                          rtol=1e-1)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTMatMulQuantDequantDims3TransposeXTest(
        TensorRTMatMulQuantDequantDims3Test):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = False
        self.alpha = 2.1


class TensorRTMatMulQuantDequantDims3TransposeYTest(
<<<<<<< HEAD
    TensorRTMatMulQuantDequantDims3Test
):
=======
        TensorRTMatMulQuantDequantDims3Test):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_params(self):
        self.transpose_x = False
        self.transpose_y = True
        self.alpha = 3.9


class TensorRTMatMulQuantDequantDims3TransposeXYTest(
<<<<<<< HEAD
    TensorRTMatMulQuantDequantDims3Test
):
=======
        TensorRTMatMulQuantDequantDims3Test):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = True
        self.alpha = 8.4


class TensorRTMatMulQuantDequantDims4Test(QuantDequantTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_params()

        def network():
<<<<<<< HEAD
            self.data = fluid.data(
                name='data', shape=[1, 28, 28], dtype='float32'
            )
            self.label = fluid.data(name='label', shape=[1, 1], dtype='int64')
            reshape_out = paddle.reshape(self.data, shape=[1, 4, 14, 14])
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
=======
            self.data = fluid.data(name='data',
                                   shape=[1, 28, 28],
                                   dtype='float32')
            self.label = fluid.data(name='label', shape=[1, 1], dtype='int64')
            reshape_out = fluid.layers.reshape(self.data, shape=[1, 4, 14, 14])
            matmul_out = fluid.layers.matmul(x=reshape_out,
                                             y=reshape_out,
                                             transpose_x=self.transpose_x,
                                             transpose_y=self.transpose_y,
                                             alpha=self.alpha)
            out = fluid.layers.batch_norm(matmul_out, is_test=True)
            fc_out = fluid.layers.fc(input=matmul_out,
                                     size=10,
                                     num_flatten_dims=1,
                                     bias_attr=False,
                                     act=None)
            result = fluid.layers.relu(fc_out)
            loss = fluid.layers.cross_entropy(input=result, label=self.label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            avg_loss = paddle.mean(loss)
            return avg_loss, result

        self.main_program.random_seed = 2
        self.startup_program.random_seed = 2
        self.test_main_program.random_seed = 2
<<<<<<< HEAD
        # self.test_startup_program.random_seed = 2
=======
        #self.test_startup_program.random_seed = 2
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        with fluid.unique_name.guard():
            with fluid.program_guard(self.main_program, self.startup_program):
                self.loss, result = network()
                opt = fluid.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with fluid.unique_name.guard():
<<<<<<< HEAD
            with fluid.program_guard(
                self.test_main_program, self.startup_program
            ):
=======
            with fluid.program_guard(self.test_main_program,
                                     self.startup_program):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                network()
        self.feeds = {"data": np.random.random([1, 28, 28]).astype("float32")}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulQuantDequantDims4Test.TensorRTParam(
<<<<<<< HEAD
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False
        )
=======
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def set_params(self):
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 1.0

    def test_check_output(self):
<<<<<<< HEAD
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
=======
        #self.quant_dequant()
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu,
                                          atol=1,
                                          flatten=False,
                                          rtol=1e-1)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTMatMulQuantDequantDims4TransposeXTest(
        TensorRTMatMulQuantDequantDims4Test):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = False
        self.alpha = 3.2


class TensorRTMatMulQuantDequantDims4TransposeYTest(
<<<<<<< HEAD
    TensorRTMatMulQuantDequantDims4Test
):
=======
        TensorRTMatMulQuantDequantDims4Test):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_params(self):
        self.transpose_x = False
        self.transpose_y = True
        self.alpha = 7.5


class TensorRTMatMulQuantDequantDims4TransposeXYTest(
<<<<<<< HEAD
    TensorRTMatMulQuantDequantDims4Test
):
=======
        TensorRTMatMulQuantDequantDims4Test):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = True
        self.alpha = 11.2


class TensorRTMatMulQuantDequantDims3DynamicTest(QuantDequantTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_params()

        def network():
<<<<<<< HEAD
            self.data = fluid.data(
                name='data', shape=[-1, 28, 28], dtype='float32'
            )
            self.label = fluid.data(name='label', shape=[1, 1], dtype='int64')
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
=======
            self.data = fluid.data(name='data',
                                   shape=[-1, 28, 28],
                                   dtype='float32')
            self.label = fluid.data(name='label', shape=[1, 1], dtype='int64')
            matmul_out = fluid.layers.matmul(x=self.data,
                                             y=self.data,
                                             transpose_x=self.transpose_x,
                                             transpose_y=self.transpose_y,
                                             alpha=self.alpha)
            out = fluid.layers.batch_norm(matmul_out, is_test=True)
            fc_out = fluid.layers.fc(input=matmul_out,
                                     size=10,
                                     num_flatten_dims=1,
                                     bias_attr=False,
                                     act=None)
            result = fluid.layers.relu(fc_out)
            loss = fluid.layers.cross_entropy(input=result, label=self.label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            avg_loss = paddle.mean(loss)
            return avg_loss, result

        self.main_program.random_seed = 2
        self.startup_program.random_seed = 2
        self.test_main_program.random_seed = 2
<<<<<<< HEAD
        # self.test_startup_program.random_seed = 2
=======
        #self.test_startup_program.random_seed = 2
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        with fluid.unique_name.guard():
            with fluid.program_guard(self.main_program, self.startup_program):
                self.loss, result = network()
                opt = fluid.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with fluid.unique_name.guard():
<<<<<<< HEAD
            with fluid.program_guard(
                self.test_main_program, self.startup_program
            ):
=======
            with fluid.program_guard(self.test_main_program,
                                     self.startup_program):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                network()
        self.feeds = {"data": np.random.random([3, 28, 28]).astype("float32")}
        self.fetch_list = [result]
        self.enable_trt = True
<<<<<<< HEAD
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
=======
        self.trt_parameters = TensorRTMatMulQuantDequantDims3DynamicTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False)
        self.dynamic_shape_params = TensorRTMatMulQuantDequantDims3DynamicTest.DynamicShapeParam(
            {'data': [1, 28, 28]}, {'data': [4, 28, 28]}, {'data': [3, 28, 28]},
            False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def set_params(self):
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 1.0

    def test_check_output(self):
<<<<<<< HEAD
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
=======
        #self.quant_dequant()
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu,
                                          atol=1,
                                          flatten=False,
                                          rtol=1e-1)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTMatMulQuantDequantDims4TransposeXDynamicTest(
        TensorRTMatMulQuantDequantDims3DynamicTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = False
        self.alpha = 2.0


class TensorRTMatMulQuantDequantDims4TransposeYDynamicTest(
<<<<<<< HEAD
    TensorRTMatMulQuantDequantDims3DynamicTest
):
=======
        TensorRTMatMulQuantDequantDims3DynamicTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_params(self):
        self.transpose_x = False
        self.transpose_y = True
        self.alpha = 2.2


class TensorRTMatMulQuantDequantDims4TransposeXYDynamicTest(
<<<<<<< HEAD
    TensorRTMatMulQuantDequantDims3DynamicTest
):
=======
        TensorRTMatMulQuantDequantDims3DynamicTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = True
        self.alpha = 7.8


if __name__ == "__main__":
    unittest.main()
