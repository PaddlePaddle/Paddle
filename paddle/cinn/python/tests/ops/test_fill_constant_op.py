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
class TestFillConstantOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.shape = [32]
        self.value = 1.0
        self.dtype = "float32"

    def build_paddle_program(self, target):
        x = paddle.full(self.shape, self.value, dtype=self.dtype)

        self.paddle_outputs = [x]

    def build_cinn_program(self, target):
        builder = NetBuilder("fill_constant")
        x = builder.fill_constant(self.shape, self.value, "out", self.dtype)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [x])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestFillConstantCase1(TestFillConstantOp):
    def init_case(self):
        self.shape = [10, 32, 4]
        self.value = 1.0
        self.dtype = "float32"


class TestFillConstantCase2(TestFillConstantOp):
    def init_case(self):
        self.shape = [32]
        self.value = 1
        self.dtype = "int32"


class TestFillConstantCase3(TestFillConstantOp):
    def init_case(self):
        self.shape = [32]
        self.value = True
        self.dtype = "bool"


class TestFillConstantCase4(TestFillConstantOp):
    def init_case(self):
        self.shape = [32]
        self.value = int(1)
        self.dtype = "uint8"


class TestFillConstantCase5(TestFillConstantOp):
    def init_case(self):
        self.shape = [32]
        self.value = int(1)
        self.dtype = "int16"


class TestFillConstantStringValue(TestFillConstantOp):
    def init_case(self):
        self.shape = [32]
        self.value = "0.12345678987654321"
        self.dtype = "float64"


class TestFillConstantStringValueCase1(TestFillConstantStringValue):
    def init_case(self):
        self.shape = [32]
        self.value = "0.12345678987654321"
        self.dtype = "float16"


class TestFillConstantStringValueCase2(TestFillConstantStringValue):
    def init_case(self):
        self.shape = [32]
        self.value = "123456789"
        self.dtype = "int64"


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestFillConstantByValueOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.shape = [32]
        self.value = float(1.0)
        self.dtype = "float32"

    def build_paddle_program(self, target):
        x = paddle.full(self.shape, self.value, dtype=self.dtype)

        self.paddle_outputs = [x]

    def build_cinn_program(self, target):
        builder = NetBuilder("fill_constant")
        x = builder.fill_constant(self.shape, self.value, "out")

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [x])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestFillConstantByValueCase1(TestFillConstantByValueOp):
    def init_case(self):
        self.shape = [32]
        self.value = int(1)
        # only for paddle.full
        self.dtype = "int32"


class TestFillConstantByValueCase2(TestFillConstantByValueOp):
    def init_case(self):
        self.shape = [32]
        self.value = bool(True)
        # only for paddle.full
        self.dtype = "bool"


if __name__ == "__main__":
    unittest.main()
