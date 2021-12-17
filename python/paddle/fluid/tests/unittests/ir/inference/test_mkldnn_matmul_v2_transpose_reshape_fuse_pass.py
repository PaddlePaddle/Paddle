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
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import PassVersionChecker


class TestMatmulV2OneDNNTransposeReshapeFusePass(InferencePassTest):
    def setUp(self):
        self.set_params()
        self.tranpose_perm = [0, 2, 1, 3]
        self.pass_name = 'matmul_v2_transpose_reshape_fuse_pass'

        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=self.data_shape, dtype="float32")
            weight = fluid.layers.create_parameter(
                shape=self.weight_shape, dtype="float32")
            matmul = paddle.matmul(
                data,
                weight,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y)
            transpose = fluid.layers.transpose(matmul, self.tranpose_perm)
            reshape = fluid.layers.reshape(transpose, shape=self.reshape_shape)

        self.fetch_list = [reshape]
        self.enable_mkldnn = True

    def set_params(self):
        self.data_shape = [-1, 3, 100, 110]
        self.weight_shape = [1, 3, 110, 100]
        self.feeds = {
            "data": np.random.random((1, 3, 100, 110)).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False
        self.reshape_shape = [3, 100, 100]

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)

    def test_pass_compatible(self):
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class TestMatmulV2OneDNNTransposeReshapeFusePassDifferentDims(
        TestMatmulV2OneDNNTransposeReshapeFusePass):
    def set_params(self):
        self.data_shape = [-1, 4, 100, 80]
        self.weight_shape = [1, 4, 80, 100]
        self.feeds = {
            "data": np.random.random((1, 4, 100, 80)).astype("float32")
        }
        self.transpose_x = True
        self.transpose_y = True
        self.reshape_shape = [8, 40, 80]


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
