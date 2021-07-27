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

from __future__ import print_function

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TRTReduceMeanTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, -1, -1], dtype="float32")
            reduce_mean = fluid.layers.reduce_mean(
                data, dim=[2, -1], keep_dim=True)
            out = fluid.layers.batch_norm(reduce_mean, is_test=True)

        self.feeds = {
            "data": np.random.random([3, 3, 224, 224]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTReduceMeanTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]
        self.dynamic_shape_params = TRTReduceMeanTest.DynamicShapeParam({
            'data': [1, 3, 64, 64]
        }, {'data': [3, 3, 224, 224]}, {'data': [3, 3, 224, 224]}, False)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTReduceMeanAllNoBatchTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, -1, -1], dtype="float32")
            reduce_mean = fluid.layers.reduce_mean(data, keep_dim=True)
            out = fluid.layers.batch_norm(reduce_mean, is_test=True)

        self.feeds = {
            "data": np.random.random([3, 3, 224, 224]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTReduceMeanAllNoBatchTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]
        self.dynamic_shape_params = TRTReduceMeanAllNoBatchTest.DynamicShapeParam(
            {
                'data': [1, 3, 64, 64]
            }, {'data': [3, 3, 224, 224]}, {'data': [3, 3, 224, 224]}, False)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTReduceMeanTestFP16(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, -1, -1], dtype="float32")
            reduce_mean = fluid.layers.reduce_mean(
                data, dim=[2, -1], keep_dim=True)
            out = fluid.layers.batch_norm(reduce_mean, is_test=True)

        self.feeds = {
            "data": np.random.random([3, 3, 224, 224]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTReduceMeanTestFP16.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Half, False, False)
        self.fetch_list = [out]
        self.dynamic_shape_params = TRTReduceMeanTestFP16.DynamicShapeParam({
            'data': [1, 3, 64, 64]
        }, {'data': [3, 3, 224, 224]}, {'data': [3, 3, 224, 224]}, False)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTReduceMeanAllTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 224, 224], dtype="float32")
            reduce_mean = fluid.layers.reduce_mean(data, keep_dim=True)
            out = fluid.layers.batch_norm(reduce_mean, is_test=True)

        self.feeds = {
            "data": np.random.random([3, 3, 224, 224]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTReduceMeanAllTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]
        self.dynamic_shape_params = TRTReduceMeanAllTest.DynamicShapeParam({
            'data': [1, 3, 224, 224]
        }, {'data': [3, 3, 224, 224]}, {'data': [3, 3, 224, 224]}, False)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTReduceMeanTestStatic(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[3, 3, 224, 224], dtype="float32")
            reduce_mean = fluid.layers.reduce_mean(
                data, dim=[2, -1], keep_dim=True)
            out = fluid.layers.batch_norm(reduce_mean, is_test=True)

        self.feeds = {
            "data": np.random.random([3, 3, 224, 224]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTReduceMeanTestStatic.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTReduceMeanStaticAllTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[4, 3, 224, 224], dtype="float32")
            reduce_mean = fluid.layers.reduce_mean(data, keep_dim=True)
            out = fluid.layers.batch_norm(reduce_mean, is_test=True)

        self.feeds = {
            "data": np.random.random([4, 3, 224, 224]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTReduceMeanStaticAllTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTReduceMeanStaticFP16(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[4, 3, 224, 224], dtype="float32")
            reduce_mean = fluid.layers.reduce_mean(data, keep_dim=True)
            out = fluid.layers.batch_norm(reduce_mean, is_test=True)

        self.feeds = {
            "data": np.random.random([4, 3, 224, 224]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTReduceMeanStaticFP16.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Half, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTReduceMeanFP16Static(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[4, 3, 224, 224], dtype="float32")
            reduce_mean = fluid.layers.reduce_mean(data, keep_dim=True)
            out = fluid.layers.batch_norm(reduce_mean, is_test=True)

        self.feeds = {
            "data": np.random.random([4, 3, 224, 224]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTReduceMeanFP16Static.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Half, True, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


if __name__ == "__main__":
    unittest.main()
