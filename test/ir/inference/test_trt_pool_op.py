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

import itertools
import os
import shutil
import unittest

import numpy as np
from inference_pass_test import InferencePassTest

import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.static import nn


class TensorRTPoolTest(InferencePassTest):
    def setUp(self):
        self.bs = 1
        self.channel = 2
        self.height = 2
        self.width = 2
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
                [self.bs, self.channel, self.height, self.width]
            ).astype('float32'),
        }

    def set_extra_config(self):
        pass

    def build_network(self):
        self.set_extra_config()
        self.trt_parameters = TensorRTPoolTest.TensorRTParam(
            1 << 30, self.bs, 0, self.precision, self.serialize, False
        )

        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name='data',
                shape=[-1, self.channel, self.height, self.width],
                dtype='float32',
            )
            if self.pool_type == 'max':
                pool_out = paddle.nn.functional.max_pool2d(
                    x=data,
                    kernel_size=self.pool_size,
                    stride=self.pool_stride,
                    padding=self.pool_padding,
                    ceil_mode=self.ceil_mode,
                )
            else:
                pool_out = paddle.nn.functional.avg_pool2d(
                    x=data,
                    kernel_size=self.pool_size,
                    stride=self.pool_stride,
                    padding=self.pool_padding,
                    ceil_mode=self.ceil_mode,
                    exclusive=self.exclusive,
                )
            out = nn.batch_norm(pool_out, is_test=True)

            self.fetch_list = [out]

    def check_output(self):
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if core.is_compiled_with_cuda():
            use_gpu = True
            if self.precision == AnalysisConfig.Precision.Float32:
                atol, rtol = (1e-5, 1e-5)
            elif self.precision == AnalysisConfig.Precision.Half:
                atol, rtol = (1e-3, 1e-3)
            else:
                raise ValueError(f"Unsupported precision {self.precision}")
            self.check_output_with_option(use_gpu, atol=atol, rtol=rtol)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )

    def run_test(self):
        self.build_network()
        self.check_output()

    def test(self):
        precision_options = [
            AnalysisConfig.Precision.Float32,
            AnalysisConfig.Precision.Half,
        ]
        serialize_options = [False, True]
        dynamic_shape_profile = InferencePassTest.DynamicShapeParam(
            {
                'data': [
                    self.bs,
                    self.channel,
                    self.height // 2,
                    self.width // 2,
                ]
            },
            {'data': [self.bs, self.channel, self.height, self.width]},
            {'data': [self.bs, self.channel, self.height, self.width]},
            False,
        )
        dynamic_shape_options = [None, dynamic_shape_profile]

        for precision, serialize, dynamic_shape in itertools.product(
            precision_options, serialize_options, dynamic_shape_options
        ):
            is_dynamic = True if dynamic_shape_options is not None else False
            with self.subTest(
                f'Precision: {precision}, Serialize: {serialize}, Dynamic: {is_dynamic}'
            ):
                self.precision = precision
                self.serialize = serialize
                self.dynamic_shape = dynamic_shape
                self.run_test()


class TensorRTAvgPoolTest(TensorRTPoolTest):
    def set_extra_config(self):
        self.pool_size = 2
        self.pool_type = 'avg'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = False


class TensorRTAvgCeilPoolTest(TensorRTPoolTest):
    def set_extra_config(self):
        self.pool_size = 2
        self.pool_type = 'avg'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = False
        self.ceil_mode = True
        self.exclusive = False


class TensorRTGlobalPoolTest(TensorRTPoolTest):
    def set_extra_config(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = True
        self.ceil_mode = False
        self.exclusive = False


class TensorRTCeilPoolTest(TensorRTPoolTest):
    def set_extra_config(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = False
        self.ceil_mode = True
        self.exclusive = False


class TensorRTExclusivePoolTest(TensorRTPoolTest):
    def set_extra_config(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = True


class TensorRTSamePaddingPoolTest(InferencePassTest):
    def set_extra_config(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 'SAME'
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = False


class TensorRTValidPaddingPoolTest(InferencePassTest):
    def set_extra_config(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 'VALID'
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = False


if __name__ == "__main__":
    unittest.main()
