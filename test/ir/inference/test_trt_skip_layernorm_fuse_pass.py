# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker


class SkipLayernormFusePassTest(InferencePassTest):
    def setUp(self):
        self.set_args()
        input_shape_with_batch = [self.batch_size, *self.input_shape]
        min_input_shape_with_batch = [1, *self.min_input_shape]
        with base.program_guard(self.main_program, self.startup_program):
            data1 = paddle.static.data(
                name='data1', shape=[-1, *self.input_shape], dtype='float32'
            )
            data2 = paddle.static.data(
                name='data2', shape=[-1, *self.input_shape], dtype='float32'
            )
            eltwise_out = paddle.add(data1, data2)
            out = paddle.nn.LayerNorm(eltwise_out.shape[-1:])(eltwise_out)
        self.feeds = {
            'data1': np.random.random(input_shape_with_batch).astype('float32'),
            'data2': np.random.random(input_shape_with_batch).astype('float32'),
        }
        self.enable_trt = True
        self.trt_parameters = SkipLayernormFusePassTest.TensorRTParam(
            1 << 30, 32, 0, self.trt_precision, True, False
        )
        self.dynamic_shape_params = SkipLayernormFusePassTest.DynamicShapeParam(
            {
                'data1': min_input_shape_with_batch,
                'data2': min_input_shape_with_batch,
            },
            {'data1': input_shape_with_batch, 'data2': input_shape_with_batch},
            {'data1': input_shape_with_batch, 'data2': input_shape_with_batch},
            False,
        )
        self.fetch_list = [out]

    def set_args(self):
        self.input_shape = [3, 128, 256]
        self.batch_size = 1
        self.trt_precision = AnalysisConfig.Precision.Float32
        self.min_input_shape = [1, 1, 256]
        self.atol = 1e-2
        self.rtol = 1e-5

    def test_check_output(self):
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(
                use_gpu, atol=self.atol, rtol=self.rtol
            )
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class SkipLayernormFusePassTest1(SkipLayernormFusePassTest):
    def set_args(self):
        self.input_shape = [256, 1536]
        self.batch_size = 1
        self.trt_precision = AnalysisConfig.Precision.Float32
        self.min_input_shape = [1, 1]
        self.atol = 1e-2
        self.rtol = 1e-5


class SkipLayernormFusePassTest2(SkipLayernormFusePassTest):
    def set_args(self):
        self.input_shape = [128, 64, 768]
        self.batch_size = 1
        self.trt_precision = AnalysisConfig.Precision.Half
        self.min_input_shape = [1, 1, 1]
        self.atol = 1e-1
        self.rtol = 1e-5


class SkipLayernormFusePassTest3(SkipLayernormFusePassTest):
    def set_args(self):
        self.input_shape = [128, 256]
        self.batch_size = 1
        self.trt_precision = AnalysisConfig.Precision.Half
        self.min_input_shape = [1, 1]
        self.atol = 1e-1
        self.rtol = 1e-5


if __name__ == "__main__":
    unittest.main()
