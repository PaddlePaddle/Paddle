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
from quant_dequant_test import QuantDequantTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import PassVersionChecker


class FCQuantDequantFusePassTRTDims3Cols1Test(QuantDequantTest):

    def setUp(self):

        def network():
            self.data = fluid.data(name='data',
                                   shape=[1, 28, 28],
                                   dtype='float32')
            self.label = fluid.data(name='label', shape=[1, 1], dtype='int64')
            fc_out = fluid.layers.fc(input=self.data,
                                     size=10,
                                     num_flatten_dims=1,
                                     bias_attr=False,
                                     act="relu")
            result = fluid.layers.relu(fc_out)
            loss = fluid.layers.cross_entropy(input=result, label=self.label)
            avg_loss = paddle.mean(loss)
            return avg_loss, result

        self.main_program.random_seed = 2
        self.startup_program.random_seed = 2
        self.test_main_program.random_seed = 2
        #self.test_startup_program.random_seed = 2
        with fluid.unique_name.guard():
            with fluid.program_guard(self.main_program, self.startup_program):
                self.loss, result = network()
                opt = fluid.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with fluid.unique_name.guard():
            with fluid.program_guard(self.test_main_program,
                                     self.startup_program):
                network()
        self.feeds = {"data": np.random.random((1, 28, 28)).astype("float32")}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = FCQuantDequantFusePassTRTDims3Cols1Test.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False)
        self.dynamic_shape_params = FCQuantDequantFusePassTRTDims3Cols1Test.DynamicShapeParam(
            {
                'data': [1, 28, 28],
                'reshape2_1.tmp_0': [1, 1, 10]
            }, {
                'data': [2, 28, 28],
                'reshape2_1.tmp_0': [2, 1, 10]
            }, {
                'data': [1, 28, 28],
                'reshape2_1.tmp_0': [1, 1, 10]
            }, False)
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def test_check_output(self):
        #self.quant_dequant()
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu,
                                          atol=1e-2,
                                          flatten=False,
                                          rtol=1e-2)
            self.assertTrue(
                PassVersionChecker.IsCompatible(
                    'quant_conv2d_dequant_fuse_pass'))


class FCQuantDequantFusePassTRTDims3Cols2Test(QuantDequantTest):

    def setUp(self):

        def network():
            self.data = fluid.data(name='data',
                                   shape=[1, 28, 28],
                                   dtype='float32')
            self.label = fluid.data(name='label', shape=[1, 1], dtype='int64')
            fc_out = fluid.layers.fc(input=self.data,
                                     size=28,
                                     num_flatten_dims=2,
                                     bias_attr=False,
                                     act=None)
            c_out = fluid.layers.reshape(fc_out, shape=[0, 784])
            result = fluid.layers.relu(c_out)
            loss = fluid.layers.cross_entropy(input=result, label=self.label)
            avg_loss = paddle.mean(loss)
            return avg_loss, result

        self.main_program.random_seed = 2
        self.startup_program.random_seed = 2
        self.test_main_program.random_seed = 2
        #self.test_startup_program.random_seed = 2
        with fluid.unique_name.guard():
            with fluid.program_guard(self.main_program, self.startup_program):
                self.loss, result = network()
                opt = fluid.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with fluid.unique_name.guard():
            with fluid.program_guard(self.test_main_program,
                                     self.startup_program):
                network()
        self.feeds = {"data": np.random.random((1, 28, 28)).astype("float32")}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = FCQuantDequantFusePassTRTDims3Cols2Test.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False)
        self.dynamic_shape_params = FCQuantDequantFusePassTRTDims3Cols2Test.DynamicShapeParam(
            {
                'data': [1, 28, 28],
                'reshape2_0.tmp_0': [1, 784]
            }, {
                'data': [4, 28, 28],
                'reshape2_0.tmp_0': [4, 784]
            }, {
                'data': [1, 28, 28],
                'reshape2_0.tmp_0': [1, 784]
            }, False)
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def test_check_output(self):
        #self.quant_dequant()
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu,
                                          atol=1e-1,
                                          flatten=False,
                                          rtol=1e-1)
            self.assertTrue(
                PassVersionChecker.IsCompatible(
                    'quant_conv2d_dequant_fuse_pass'))


class FCQuantDequantFusePassTRTDims3Cols3Test(QuantDequantTest):

    def setUp(self):

        def network():
            self.data = fluid.data(name='data',
                                   shape=[1, 28, 28],
                                   dtype='float32')
            self.label = fluid.data(name='label', shape=[1, 1], dtype='int64')
            label_shape = fluid.layers.reshape(self.label, shape=[1, 1, 1])
            reshape_out = fluid.layers.reshape(self.data, shape=[1, 14, 14, 4])
            fc_out = fluid.layers.fc(input=reshape_out,
                                     size=14,
                                     num_flatten_dims=3,
                                     bias_attr=False,
                                     act=None)
            c_out = fluid.layers.reshape(fc_out, shape=[1, 1, 2744])
            result = fluid.layers.relu(c_out)
            loss = fluid.layers.cross_entropy(input=result, label=label_shape)
            avg_loss = paddle.mean(loss)
            return avg_loss, result

        self.main_program.random_seed = 2
        self.startup_program.random_seed = 2
        self.test_main_program.random_seed = 2
        #self.test_startup_program.random_seed = 2
        with fluid.unique_name.guard():
            with fluid.program_guard(self.main_program, self.startup_program):
                self.loss, result = network()
                opt = fluid.optimizer.Adam(learning_rate=0.0001)
                opt.minimize(self.loss)
        with fluid.unique_name.guard():
            with fluid.program_guard(self.test_main_program,
                                     self.startup_program):
                network()
        self.feeds = {"data": np.random.random((1, 28, 28)).astype("float32")}
        self.fetch_list = [result]
        self.enable_trt = True
        self.trt_parameters = FCQuantDequantFusePassTRTDims3Cols3Test.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False)
        self.dynamic_shape_params = FCQuantDequantFusePassTRTDims3Cols3Test.DynamicShapeParam(
            {
                'data': [1, 28, 28],
                "reshape2_1.tmp_0": [1, 14, 14, 4],
                "reshape2_2.tmp_0": [1, 1, 2744]
            }, {
                'data': [4, 28, 28],
                "reshape2_1.tmp_0": [4, 14, 14, 4],
                "reshape2_2.tmp_0": [4, 1, 2744]
            }, {
                'data': [1, 28, 28],
                "reshape2_1.tmp_0": [1, 14, 14, 4],
                "reshape2_2.tmp_0": [1, 1, 2744]
            }, False)
        self.activation_quantize_type = 'moving_average_abs_max'
        self.weight_quantize_type = 'channel_wise_abs_max'

    def test_check_output(self):
        #self.quant_dequant()
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu,
                                          atol=1e0,
                                          flatten=False,
                                          rtol=1e0)
            self.assertTrue(
                PassVersionChecker.IsCompatible(
                    'quant_conv2d_dequant_fuse_pass'))


if __name__ == "__main__":
    unittest.main()
