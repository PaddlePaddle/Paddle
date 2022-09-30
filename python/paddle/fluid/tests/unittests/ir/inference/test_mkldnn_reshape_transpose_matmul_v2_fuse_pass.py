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

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import PassVersionChecker


class TestReshapeTransposeMatmulV2OneDNNFusePass(InferencePassTest):

    def setUp(self):
        self.set_params()
        self.tranpose_perm = [0, 2, 1, 3]
        self.pass_name = 'reshape_transpose_matmul_mkldnn_fuse_pass'

        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data",
                              shape=self.data_shape,
                              dtype="float32")
            weight = fluid.layers.create_parameter(shape=self.weight_shape,
                                                   dtype="float32")
            reshape = fluid.layers.reshape(data, shape=self.reshape_shape)
            transpose = fluid.layers.transpose(reshape, self.tranpose_perm)
            matmul = paddle.matmul(transpose,
                                   weight,
                                   transpose_x=self.transpose_x,
                                   transpose_y=self.transpose_y)

        self.fetch_list = [matmul]
        self.enable_mkldnn = True

    def set_params(self):
        self.data_shape = [-1, 128, 768]
        self.weight_shape = [1, 12, 64, 128]
        self.feeds = {"data": np.random.random((1, 128, 768)).astype("float32")}
        self.transpose_x = False
        self.transpose_y = False
        self.reshape_shape = [0, 0, 12, 64]

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)

    def test_pass_compatible(self):
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class TestReshapeTransposeMatmulV2OneDNNFusePassBroadcast(
        TestReshapeTransposeMatmulV2OneDNNFusePass):

    def set_params(self):
        self.data_shape = [2, 64, 16]
        self.weight_shape = [1, 2, 8, 64]
        self.feeds = {"data": np.random.random((2, 64, 16)).astype("float32")}
        self.transpose_x = True
        self.transpose_y = True
        self.reshape_shape = [0, 0, 2, 8]


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
