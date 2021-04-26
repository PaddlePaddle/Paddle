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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TensorRTMatMulDims2Test(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[24, 24], dtype="float32")
            matmul_out = fluid.layers.matmul(
                x=data,
                y=data,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
                alpha=self.alpha)
            out = fluid.layers.batch_norm(matmul_out, is_test=True)

        self.feeds = {"data": np.ones([24, 24]).astype("float32"), }
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulDims2Test.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        self.transpose_x = True
        self.transpose_y = True
        self.alpha = 2.0

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTMatMulTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 24, 24], dtype="float32")
            matmul_out = fluid.layers.matmul(
                x=data,
                y=data,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
                alpha=self.alpha)
            out = fluid.layers.batch_norm(matmul_out, is_test=True)

        self.feeds = {"data": np.ones([1, 6, 24, 24]).astype("float32"), }
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 1.0

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTMatMulTransposeXTest(TensorRTMatMulTest):
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = False
        self.alpha = 1.0


class TensorRTMatMulTransposeYTest(TensorRTMatMulTest):
    def set_params(self):
        self.transpose_x = False
        self.transpose_y = True
        self.alpha = 1.0


class TensorRTMatMulScaleTest(TensorRTMatMulTest):
    def set_params(self):
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 2.0


if __name__ == "__main__":
    unittest.main()
