#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

from cinn.common import *
from cinn.frontend import *
from op_test import OpTest, OpTestTool

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestGaussianRandomOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.shape = [2, 3]
        self.mean = 0.0
        self.std = 1.0
        self.seed = 10
        self.dtype = "float32"

    def build_paddle_program(self, target):
        out = paddle.tensor.random.gaussian(
            shape=self.shape, mean=self.mean, std=self.std, dtype=self.dtype
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("gaussian_random")
        out = builder.gaussian_random(
            self.shape, self.mean, self.std, self.seed, self.dtype
        )
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [out], passes=[])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        # Due to the different random number generation numbers implemented
        # in the specific implementation, the random number results generated
        # by CINN and Paddle are not the same, but they all conform to the
        # Uniform distribution.
        self.check_outputs_and_grads(max_relative_error=10000)


class TestGaussianRandomCase1(TestGaussianRandomOp):
    def init_case(self):
        self.shape = [2, 3, 4]
        self.mean = 1.0
        self.std = 2.0
        self.seed = 10
        self.dtype = "float32"


class TestGaussianRandomCase2(TestGaussianRandomOp):
    def init_case(self):
        self.shape = [2, 3, 4]
        self.mean = 2.0
        self.std = 3.0
        self.seed = 10
        self.dtype = "float64"


if __name__ == "__main__":
    unittest.main()
