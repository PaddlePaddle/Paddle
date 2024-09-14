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


class TRTTileTest(InferencePassTest):
    def setUp(self):
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[4, 3, 224, 256], dtype="float32"
            )
            tile_out = paddle.tile(x=data, repeat_times=[1, 1, 1, 1])
            out = paddle.static.nn.batch_norm(tile_out, is_test=True)

        self.feeds = {
            "data": np.random.random([4, 3, 224, 256]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTTileTest.TensorRTParam(
            1 << 30, 16, 1, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = TRTTileTest.DynamicShapeParam(
            {'data': [4, 3, 224, 256]},
            {'data': [4, 3, 224, 256]},
            {'data': [4, 3, 224, 256]},
            False,
        )
        self.fetch_list = [out]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TRTTileExpandTest(InferencePassTest):
    def setUp(self):
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[1, 1, 1, 1], dtype="float32"
            )
            tile_out = paddle.tile(x=data, repeat_times=[1, 4, 1080, 1920])
            out = paddle.static.nn.batch_norm(tile_out, is_test=True)

        self.feeds = {
            "data": np.random.random([1, 1, 1, 1]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTTileExpandTest.TensorRTParam(
            1 << 30, 1, 1, AnalysisConfig.Precision.Float32, False, False
        )
        self.dynamic_shape_params = TRTTileTest.DynamicShapeParam(
            {'data': [1, 1, 1, 1]},
            {'data': [1, 1, 1, 1]},
            {'data': [1, 1, 1, 1]},
            False,
        )
        self.fetch_list = [out]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TRTTileExpandStaticTest(InferencePassTest):
    def setUp(self):
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[1, 1, 1, 1], dtype="float32"
            )
            tile_out = paddle.tile(x=data, repeat_times=[1, 4, 1080, 1920])
            out = paddle.static.nn.batch_norm(tile_out, is_test=True)

        self.feeds = {
            "data": np.random.random([1, 1, 1, 1]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTTileExpandStaticTest.TensorRTParam(
            1 << 30, 1, 1, AnalysisConfig.Precision.Float32, True, False
        )
        self.dynamic_shape_params = TRTTileExpandStaticTest.DynamicShapeParam(
            {'data': [1, 1, 1, 1]},
            {'data': [1, 1, 1, 1]},
            {'data': [1, 1, 1, 1]},
            False,
        )
        self.fetch_list = [out]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TRTTileExpandHalfTest(InferencePassTest):
    def setUp(self):
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[1, 1, 1, 1], dtype="float32"
            )
            tile_out = paddle.tile(x=data, repeat_times=[1, 4, 1080, 1920])
            out = paddle.static.nn.batch_norm(tile_out, is_test=True)

        self.feeds = {
            "data": np.random.random([1, 1, 1, 1]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTTileExpandHalfTest.TensorRTParam(
            1 << 30, 1, 1, AnalysisConfig.Precision.Half, False, False
        )
        self.dynamic_shape_params = TRTTileTest.DynamicShapeParam(
            {'data': [1, 1, 1, 1]},
            {'data': [1, 1, 1, 1]},
            {'data': [1, 1, 1, 1]},
            False,
        )
        self.fetch_list = [out]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, 1e-4, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


if __name__ == "__main__":
    unittest.main()
