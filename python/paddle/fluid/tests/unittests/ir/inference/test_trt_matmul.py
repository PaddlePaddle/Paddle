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
<<<<<<< HEAD

import numpy as np
from inference_pass_test import InferencePassTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.static.nn as nn
from paddle.fluid.core import AnalysisConfig, PassVersionChecker


class TensorRTMatMulDims2Test(InferencePassTest):
=======
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TensorRTMatMulDims2Test(InferencePassTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[24, 24], dtype="float32")
<<<<<<< HEAD
            matmul_out = paddle.matmul(
                x=data,
                y=data,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
            )
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            out = nn.batch_norm(matmul_out, is_test=True)
=======
            matmul_out = fluid.layers.matmul(x=data,
                                             y=data,
                                             transpose_x=self.transpose_x,
                                             transpose_y=self.transpose_y,
                                             alpha=self.alpha)
            out = fluid.layers.batch_norm(matmul_out, is_test=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.feeds = {
            "data": np.ones([24, 24]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulDims2Test.TensorRTParam(
<<<<<<< HEAD
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
=======
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTMatMulTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 24, 24], dtype="float32"
            )
            matmul_out = paddle.matmul(
                x=data,
                y=data,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
            )
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            out = nn.batch_norm(matmul_out, is_test=True)
=======
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTMatMulTest(InferencePassTest):

    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data",
                              shape=[-1, 6, 24, 24],
                              dtype="float32")
            matmul_out = fluid.layers.matmul(x=data,
                                             y=data,
                                             transpose_x=self.transpose_x,
                                             transpose_y=self.transpose_y,
                                             alpha=self.alpha)
            out = fluid.layers.batch_norm(matmul_out, is_test=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.feeds = {
            "data": np.ones([1, 6, 24, 24]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulTest.TensorRTParam(
<<<<<<< HEAD
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
=======
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTMatMulTransposeXTest(TensorRTMatMulTest):
=======
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTMatMulTransposeXTest(TensorRTMatMulTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_params(self):
        self.transpose_x = True
        self.transpose_y = False
        self.alpha = 1.0


class TensorRTMatMulTransposeYTest(TensorRTMatMulTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_params(self):
        self.transpose_x = False
        self.transpose_y = True
        self.alpha = 1.0


class TensorRTMatMulScaleTest(TensorRTMatMulTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_params(self):
        self.transpose_x = False
        self.transpose_y = False
        self.alpha = 2.0


class TensorRTMatMulBroadcastTest(InferencePassTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_params()
        place = fluid.CPUPlace()
        with fluid.program_guard(self.main_program, self.startup_program):
<<<<<<< HEAD
            data_x = fluid.data(
                name="data_x", shape=[-1, 6, 24], dtype="float32"
            )
            data_y = fluid.data(name="data_y", shape=[24, 16], dtype="float32")
            matmul_out = paddle.matmul(
                x=data_x,
                y=data_y,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
            )
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            out = nn.batch_norm(matmul_out, is_test=True)

        self.feeds = {
            "data_x": np.ones([2, 6, 24]).astype("float32"),
            "data_y": np.ones([24, 16]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulBroadcastTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False
        )
=======
            data_x = fluid.data(name="data_x",
                                shape=[-1, 6, 24],
                                dtype="float32")
            data_y = fluid.data(name="data_y", shape=[24, 16], dtype="float32")
            matmul_out = fluid.layers.matmul(x=data_x,
                                             y=data_y,
                                             transpose_x=self.transpose_x,
                                             transpose_y=self.transpose_y,
                                             alpha=self.alpha)
            out = fluid.layers.batch_norm(matmul_out, is_test=True)

        self.feeds = {
            "data_x": np.ones([2, 6, 24]).astype("float32"),
            "data_y": np.ones([24, 16]).astype("float32")
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTMatMulBroadcastTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )
=======
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == "__main__":
    unittest.main()
