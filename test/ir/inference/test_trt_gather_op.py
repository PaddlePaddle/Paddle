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


class TRTGatherTest1(InferencePassTest):
    def setUp(self):
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name='data', shape=[-1, 128], dtype='float32'
            )
            index = paddle.static.data(
                name='index', shape=[-1, 1], dtype='int32'
            )
            scale_out = paddle.gather(data, index=index)
            out = paddle.nn.functional.softmax(scale_out)

        self.feeds = {
            "data": np.random.random([self.bs, 128]).astype("float32"),
            "index": self.index,
        }

        self.enable_trt = True
        self.trt_parameters = TRTGatherTest1.TensorRTParam(
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = TRTGatherTest1.DynamicShapeParam(
            {'data': [1, 1], 'index': [1, 1]},
            {'data': [32, 128], 'index': [3, 1]},
            {'data': [32, 128], 'index': [3, 1]},
            False,
        )
        self.fetch_list = [out]

    def set_params(self):
        self.index = np.array([[1], [2], [3]], dtype='int32')
        self.bs = 4

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=False)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TRTGatherTest2(InferencePassTest):
    def setUp(self):
        self.set_params()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name='data', shape=[16, 64], dtype='float32'
            )
            index = paddle.static.data(name='index', shape=[2], dtype='int32')
            scale_out = paddle.gather(data, index=index)
            out = paddle.nn.functional.softmax(scale_out)

        self.feeds = {
            "data": np.random.random([self.bs, 64]).astype("float32"),
            "index": self.index,
        }

        self.enable_trt = True
        self.trt_parameters = TRTGatherTest2.TensorRTParam(
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = TRTGatherTest2.DynamicShapeParam(
            {'data': [2, 4], 'index': [1]},
            {'data': [256, 256], 'index': [4]},
            {'data': [64, 32], 'index': [2]},
            False,
        )
        self.fetch_list = [out]

    def set_params(self):
        self.index = np.array([1, 4], dtype='int32')
        self.bs = 16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=False)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


if __name__ == "__main__":
    unittest.main()
