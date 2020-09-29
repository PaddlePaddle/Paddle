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

from __future__ import print_function

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import PassVersionChecker


class ScaleMatmulMkldnnFusePassTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[1, 3, 100, 100], dtype="float32")
            weight = fluid.layers.create_parameter(
                shape=[1, 3, 100, 100], dtype="float32")
            scale = fluid.layers.scale(data, scale=self.scale_scale)
            matmul = fluid.layers.matmul(
                scale,
                weight,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y)

        self.fetch_list = [matmul]
        self.enable_mkldnn = True

    def set_params(self):
        self.feeds = {
            "data": np.random.random((1, 3, 100, 100)).astype("float32")
        }
        self.scale_scale = 2.0
        self.transpose_x = False
        self.transpose_y = False
        self.pass_name = "scale_matmul_fuse_pass"

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)

    def test_pass_compatible(self):
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class ScaleMatmulMkldnnFusePassTest_1(ScaleMatmulMkldnnFusePassTest):
    def set_params(self):
        self.feeds = {
            "data": np.random.random((1, 3, 100, 100)).astype("float32")
        }
        self.scale_scale = 5.0
        self.transpose_x = True
        self.transpose_y = True
        self.pass_name = "scale_matmul_fuse_pass"


if __name__ == "__main__":
    unittest.main()
