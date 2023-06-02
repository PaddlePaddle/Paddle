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
class TestConstantOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.value = 1.0
        self.name = 'x'
        self.dtype = "float32"

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.value, dtype=self.dtype)

        self.paddle_outputs = [x]

    def build_cinn_program(self, target):
        builder = NetBuilder("constant")
        x = builder.constant(self.value, self.name, self.dtype)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [x])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestConstantCase1(TestConstantOp):
    def init_case(self):
        self.value = [1.0]
        self.name = 'x'
        self.dtype = "float32"


class TestConstantCase2(TestConstantOp):
    def init_case(self):
        self.value = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.name = 'x'
        self.dtype = "float32"


class TestConstantCase3(TestConstantOp):
    def init_case(self):
        self.value = [[1.0, 2.0], [3.0, 4.0]]
        self.name = 'x'
        self.dtype = "float32"


class TestConstantCase4(TestConstantOp):
    def init_case(self):
        self.value = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        self.name = 'x'
        self.dtype = "float32"


class TestConstantCase5(TestConstantOp):
    def init_case(self):
        self.value = [[[1.0], [3.0]], [[5.0], [7.0]]]
        self.name = 'x'
        self.dtype = "float32"


class TestConstantCase6(TestConstantOp):
    def init_case(self):
        self.value = [[[1.0]]]
        self.name = 'x'
        self.dtype = "float32"


class TestConstantCase7(TestConstantOp):
    def init_case(self):
        self.value = self.random([200], "int32", 1, 1000).tolist()
        self.name = 'x'
        self.dtype = "int32"


class TestConstantCase8(TestConstantOp):
    def init_case(self):
        self.value = self.random([10], "int64", 1, 1000).tolist()
        self.name = 'x'
        self.dtype = "int64"


if __name__ == "__main__":
    unittest.main()
