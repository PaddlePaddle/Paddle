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
import itertools
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TensorRTPool3dTest(InferencePassTest):
    def setUp(self):
        self.bs = 1
        self.channel = 3
        self.depth = 8
        self.height = 8
        self.width = 8
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = False
        self.enable_trt = True
        self.serialize = False
        self.precision = AnalysisConfig.Precision.Float32
        self.feeds = {
            'data': np.random.random(
                [self.bs, self.channel, self.depth, self.height,
                 self.width]).astype('float32'),
        }

    def set_extra_config(self):
        pass

    def build_network(self):
        self.set_extra_config()
        self.trt_parameters = TensorRTPool3dTest.TensorRTParam(
            1 << 30, self.bs, 0, self.precision, self.serialize, False)

        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data',
                shape=[-1, self.channel, self.depth, self.height, self.width],
                dtype='float32')
            pool_out = fluid.layers.pool3d(
                input=data,
                pool_size=self.pool_size,
                pool_type=self.pool_type,
                pool_stride=self.pool_stride,
                pool_padding=self.pool_padding,
                global_pooling=self.global_pooling,
                ceil_mode=self.ceil_mode,
                exclusive=self.exclusive)
            #out = fluid.layers.batch_norm(pool_out, is_test=True)
            self.fetch_list = [pool_out]

    def check_output(self):
        if os.path.exists(self.path + "_opt_cache"):
            shutil.rmtree(self.path + "_opt_cache")
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

    def run_test(self):
        self.build_network()
        self.check_output()

    def test(self):
        precision_options = [
            AnalysisConfig.Precision.Float32, AnalysisConfig.Precision.Half
        ]
        serialize_options = [False, True]
        dynamic_shape_profile = InferencePassTest.DynamicShapeParam({
            'data': [
                self.bs, self.channel, self.depth // 2, self.height // 2,
                self.width // 2
            ]
        }, {
            'data':
            [self.bs, self.channel, self.depth, self.height, self.width]
        }, {
            'data':
            [self.bs, self.channel, self.depth, self.height, self.width]
        }, False)
        dynamic_shape_options = [None, dynamic_shape_profile]

        for precision, serialize, dynamic_shape in itertools.product(
                precision_options, serialize_options, dynamic_shape_options):
            is_dynamic = True if dynamic_shape_options is not None else False
            with self.subTest('Precision: {}, Serialize: {}, Dynamic: {}'.
                              format(precision, serialize, is_dynamic)):
                self.precision = precision
                self.serialize = serialize
                self.dynamic_shape_params = dynamic_shape
                self.run_test()


class TensorRTAvgPool3dTest(TensorRTPool3dTest):
    def set_extra_config(self):
        self.pool_size = 2
        self.pool_type = 'avg'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = False


class TensorRTGlobalPool3dTest(TensorRTPool3dTest):
    def set_extra_config(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = True
        self.ceil_mode = False
        self.exclusive = False


class TensorRTCeilPool3dTest(TensorRTPool3dTest):
    def set_extra_config(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = False
        self.ceil_mode = True
        self.exclusive = False


class TensorRTExclusivePool3dTest(TensorRTPool3dTest):
    def set_extra_config(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = True


class TensorRTSamePaddingPool3dTest(InferencePassTest):
    def set_extra_config(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 'SAME'
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = False


class TensorRTValidPaddingPool3dTest(InferencePassTest):
    def set_extra_config(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 'VALID'
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = False


class TensorRTAdaptiveAvgPool3DTest(InferencePassTest):
    def setUp(self):
        self.bs = 1
        self.channel = 3
        self.depth = 8
        self.height = 8
        self.width = 8
        self.enable_trt = True
        self.serialize = False
        self.precision = AnalysisConfig.Precision.Float32
        self.feeds = {
            'data': np.random.random(
                [self.bs, self.channel, self.depth, self.height,
                 self.width]).astype('float32'),
        }

    def build_network(self):
        self.trt_parameters = TensorRTPool3dTest.TensorRTParam(
            1 << 30, self.bs, 0, self.precision, self.serialize, False)

        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data',
                shape=[-1, self.channel, self.depth, self.height, self.width],
                dtype='float32')
            pool_out = paddle.nn.functional.adaptive_avg_pool3d(
                x=data, output_size=[3, 3, 3])
            #out = fluid.layers.batch_norm(pool_out, is_test=True)
            self.fetch_list = [pool_out]

    def check_output(self):
        if os.path.exists(self.path + "_opt_cache"):
            shutil.rmtree(self.path + "_opt_cache")
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

    def run_test(self):
        self.build_network()
        self.check_output()

    def test(self):
        precision_options = [
            AnalysisConfig.Precision.Float32, AnalysisConfig.Precision.Half
        ]
        serialize_options = [False, True]
        dynamic_shape_profile = InferencePassTest.DynamicShapeParam({
            'data': [
                self.bs, self.channel, self.depth // 2, self.height // 2,
                self.width // 2
            ]
        }, {
            'data':
            [self.bs, self.channel, self.depth, self.height, self.width]
        }, {
            'data':
            [self.bs, self.channel, self.depth, self.height, self.width]
        }, False)
        dynamic_shape_options = [None, dynamic_shape_profile]

        for precision, serialize, dynamic_shape in itertools.product(
                precision_options, serialize_options, dynamic_shape_options):
            is_dynamic = True if dynamic_shape_options is not None else False
            with self.subTest('Precision: {}, Serialize: {}, Dynamic: {}'.
                              format(precision, serialize, is_dynamic)):
                self.precision = precision
                self.serialize = serialize
                self.dynamic_shape_params = dynamic_shape
                self.run_test()


class TensorRTAdaptiveMaxPool3DTest(InferencePassTest):
    def setUp(self):
        self.bs = 1
        self.channel = 3
        self.depth = 8
        self.height = 8
        self.width = 8
        self.enable_trt = True
        self.serialize = False
        self.precision = AnalysisConfig.Precision.Float32
        self.feeds = {
            'data': np.random.random(
                [self.bs, self.channel, self.depth, self.height,
                 self.width]).astype('float32'),
        }

    def build_network(self):
        self.trt_parameters = TensorRTPool3dTest.TensorRTParam(
            1 << 30, self.bs, 0, self.precision, self.serialize, False)

        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data',
                shape=[-1, self.channel, self.depth, self.height, self.width],
                dtype='float32')
            pool_out = paddle.nn.functional.adaptive_max_pool3d(
                x=data, output_size=[3, 3, 3])
            #out = fluid.layers.batch_norm(pool_out, is_test=True)
            self.fetch_list = [pool_out]

    def check_output(self):
        if os.path.exists(self.path + "_opt_cache"):
            shutil.rmtree(self.path + "_opt_cache")
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

    def run_test(self):
        self.build_network()
        self.check_output()

    def test(self):
        precision_options = [
            AnalysisConfig.Precision.Float32, AnalysisConfig.Precision.Half
        ]
        serialize_options = [False, True]
        dynamic_shape_profile = InferencePassTest.DynamicShapeParam({
            'data': [
                self.bs, self.channel, self.depth // 2, self.height // 2,
                self.width // 2
            ]
        }, {
            'data':
            [self.bs, self.channel, self.depth, self.height, self.width]
        }, {
            'data':
            [self.bs, self.channel, self.depth, self.height, self.width]
        }, False)
        dynamic_shape_options = [None, dynamic_shape_profile]

        for precision, serialize, dynamic_shape in itertools.product(
                precision_options, serialize_options, dynamic_shape_options):
            is_dynamic = True if dynamic_shape_options is not None else False
            with self.subTest('Precision: {}, Serialize: {}, Dynamic: {}'.
                              format(precision, serialize, is_dynamic)):
                self.precision = precision
                self.serialize = serialize
                self.dynamic_shape_params = dynamic_shape
                self.run_test()


if __name__ == "__main__":
    unittest.main()
