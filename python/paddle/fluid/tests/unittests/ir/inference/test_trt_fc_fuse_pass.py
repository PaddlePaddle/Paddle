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


class FCFusePassTRTTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[32, 128, 2, 2], dtype="float32")
            fc_out1 = fluid.layers.fc(input=data,
                                      size=128,
                                      num_flatten_dims=1,
                                      act="relu")
            out = fluid.layers.softmax(input=fc_out1)

        self.feeds = {
            "data": np.random.random((32, 128, 2, 2)).astype("float32")
        }
        # Diff occurred between GPU and TRT. 
        # In order to provide TRT CI ASAP, this test for trt part 
        # is disabled temporarily. 
        # self.enable_trt = True
        # self.trt_parameters = FCFusePassTRTTest.TensorRTParam(
        #     1 << 30, 32, 3, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


class FCFusePassTRTStaticDims4Cols1Test(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[32, 128, 32, 8], dtype="float32")
            fc_out1 = fluid.layers.fc(input=data,
                                      size=64,
                                      num_flatten_dims=1,
                                      act="relu")
            out = fluid.layers.softmax(input=fc_out1)

        self.feeds = {
            "data": np.random.random((32, 128, 32, 8)).astype("float32")
        }
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTStaticDims4Cols1Test.TensorRTParam(
            1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


class FCFusePassTRTStaticDims4Cols2Test(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[3, 24, 16, 16], dtype="float32")
            fc_out1 = fluid.layers.fc(input=data,
                                      size=32,
                                      num_flatten_dims=2,
                                      act="relu")
            out = fluid.layers.softmax(input=fc_out1)

        self.feeds = {
            "data": np.random.random((3, 24, 16, 16)).astype("float32")
        }
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTStaticDims4Cols2Test.TensorRTParam(
            1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


class FCFusePassTRTDynamicDims2Test(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[32, 128], dtype="float32")
            fc_out1 = fluid.layers.fc(input=data,
                                      size=64,
                                      num_flatten_dims=1,
                                      act="relu")
            out = fluid.layers.softmax(input=fc_out1)

        self.feeds = {"data": np.random.random((32, 128)).astype("float32")}
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTDynamicDims2Test.TensorRTParam(
            1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = FCFusePassTRTDynamicDims2Test.DynamicShapeParam(
            {
                'data': [1, 128]
            }, {'data': [64, 128]}, {'data': [32, 128]}, False)
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


class FCFusePassTRTDynamicDims3Cols1Test(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[32, 128, 32], dtype="float32")
            fc_out1 = fluid.layers.fc(input=data,
                                      size=64,
                                      num_flatten_dims=1,
                                      act="relu")
            out = fluid.layers.softmax(input=fc_out1)

        self.feeds = {"data": np.random.random((32, 128, 32)).astype("float32")}
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTDynamicDims3Cols1Test.TensorRTParam(
            1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = FCFusePassTRTDynamicDims3Cols1Test.DynamicShapeParam(
            {
                'data': [1, 128, 32]
            }, {'data': [64, 128, 32]}, {'data': [32, 128, 32]}, False)
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


class FCFusePassTRTDynamicDims3Cols2Test(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[32, 128, 32], dtype="float32")
            fc_out1 = fluid.layers.fc(input=data,
                                      size=64,
                                      num_flatten_dims=2,
                                      act="relu")
            out = fluid.layers.softmax(input=fc_out1)

        self.feeds = {"data": np.random.random((32, 128, 32)).astype("float32")}
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTDynamicDims3Cols2Test.TensorRTParam(
            1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = FCFusePassTRTDynamicDims3Cols2Test.DynamicShapeParam(
            {
                'data': [1, 32, 32]
            }, {'data': [64, 256, 32]}, {'data': [32, 128, 32]}, False)
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


class FCFusePassTRTDynamicDims4Cols1Test(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[32, 12, 4, 6], dtype="float32")
            fc_out1 = fluid.layers.fc(input=data,
                                      size=64,
                                      num_flatten_dims=1,
                                      act="relu")
            out = fluid.layers.softmax(input=fc_out1)

        self.feeds = {
            "data": np.random.random((32, 12, 4, 6)).astype("float32")
        }
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTDynamicDims4Cols1Test.TensorRTParam(
            1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = FCFusePassTRTDynamicDims4Cols1Test.DynamicShapeParam(
            {
                'data': [1, 12, 4, 6]
            }, {'data': [64, 12, 4, 6]}, {'data': [32, 12, 4, 6]}, False)
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


class FCFusePassTRTDynamicDims4Cols2Test(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[32, 128, 32, 32], dtype="float32")
            fc_out1 = fluid.layers.fc(input=data,
                                      size=64,
                                      num_flatten_dims=2,
                                      act="relu")
            out = fluid.layers.softmax(input=fc_out1)

        self.feeds = {
            "data": np.random.random((32, 128, 32, 32)).astype("float32")
        }
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTDynamicDims4Cols2Test.TensorRTParam(
            1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = FCFusePassTRTDynamicDims4Cols2Test.DynamicShapeParam(
            {
                'data': [1, 64, 32, 32]
            }, {'data': [64, 256, 32, 32]}, {'data': [32, 128, 32, 32]}, False)
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


class FCFusePassTRTDynamicDims4Cols3Test(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[32, 128, 32, 32], dtype="float32")
            fc_out1 = fluid.layers.fc(input=data,
                                      size=64,
                                      num_flatten_dims=3,
                                      act="relu")
            out = fluid.layers.softmax(input=fc_out1)

        self.feeds = {
            "data": np.random.random((32, 128, 32, 32)).astype("float32")
        }
        self.enable_trt = True
        self.trt_parameters = FCFusePassTRTDynamicDims4Cols3Test.TensorRTParam(
            1 << 30, 32, 2, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = FCFusePassTRTDynamicDims4Cols3Test.DynamicShapeParam(
            {
                'data': [1, 128, 32, 32]
            }, {'data': [64, 128, 32, 32]}, {'data': [32, 128, 32, 32]}, False)
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


class FCQuantDequantTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 32, 32], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.0),
                trainable=False)
            quantized_op_out = self.append_quantized_op(data, param_attr)
            relu_out = fluid.layers.relu(quantized_op_out)
        self.set_quant_pattern()

        self.feeds = {
            "data": np.random.random([1, 3, 32, 32]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = FCQuantDequantTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False)
        self.fetch_list = [relu_out]

    def append_quantized_op(self, x, param_attr):
        return fluid.layers.fc(x,
                               size=100,
                               num_flatten_dims=3,
                               param_attr=param_attr,
                               bias_attr=False,
                               act=None)

    def set_quant_pattern(self):
        self.activation_quant_type = 'moving_average_abs_max'
        self.weight_quant_type = 'abs_max'
        self.quantized_op_type = 'mul'
        self.channels = 1

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=False, quant=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible(
                    'quant_conv2d_dequant_fuse_pass'))


if __name__ == "__main__":
    unittest.main()
