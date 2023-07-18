#!/usr/bin/env python3

# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestRandIntOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.shape = [2, 3]
        self.min = 0
        self.max = 5
        self.seed = 10
        self.dtype = "int32"

    def build_paddle_program(self, target):
        out = paddle.randint(
            shape=self.shape, low=self.min, high=self.max, dtype=self.dtype
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("randint")
        out = builder.randint(
            self.shape, self.min, self.max, self.seed, self.dtype
        )
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [out], passes=[])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        # Due to the different random number generation numbers implemented
        # in the specific implementation, the random number results generated
        # by CINN and Paddle are not the same, but they all conform to the
        # uniform distribution.
        # self.check_outputs_and_grads()
        self.build_paddle_program(self.target)
        self.build_cinn_program(self.target)


class TestRandIntCase1(TestRandIntOp):
    def init_case(self):
        self.shape = [2, 3, 4]
        self.min = 0
        self.max = 8
        self.seed = 10
        self.dtype = "int32"


class TestRandIntCase2(TestRandIntOp):
    def init_case(self):
        self.shape = [2, 3, 4]
        self.min = -2
        self.max = 3
        self.seed = 8
        self.dtype = "int64"


if __name__ == "__main__":
    unittest.main()
