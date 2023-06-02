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

import numpy as np
from cinn.common import *
from cinn.frontend import *
from op_test import OpTest, OpTestTool

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestConcatOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x1": np.random.random([10201, 50]).astype("float32"),
            "x2": np.random.random((10201, 50)).astype("float32"),
        }
        self.axis = 0

    def paddle_inputs(self, inputs):
        return [
            paddle.to_tensor(data, stop_gradient=True)
            for _, data in inputs.items()
        ]

    def cinn_inputs(self, builder, inputs):
        return [
            builder.create_input(Float(32), data.shape, name)
            for name, data in inputs.items()
        ]

    def build_paddle_program(self, target):
        out = paddle.concat(x=self.paddle_inputs(self.inputs), axis=self.axis)

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("concat")
        input_list = self.cinn_inputs(builder, self.inputs)
        out = builder.concat(input_list, axis=self.axis)

        prog = builder.build()

        input_datas = [data for _, data in self.inputs.items()]

        res = self.get_cinn_output(prog, target, input_list, input_datas, [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestConcatCase1(TestConcatOp):
    def init_case(self):
        self.inputs = {
            "x1": np.random.random([4, 3]).astype("float32"),
            "x2": np.random.random((8, 3)).astype("float32"),
        }
        self.axis = 0


class TestConcatCase2(TestConcatOp):
    def init_case(self):
        self.inputs = {
            "x1": np.random.random([2, 4, 8]).astype("float32"),
            "x2": np.random.random((2, 4, 4)).astype("float32"),
        }
        self.axis = -1


class TestConcatCase3(TestConcatOp):
    def init_case(self):
        self.inputs = {
            "x1": np.random.random([2, 8, 4]).astype("float32"),
            "x2": np.random.random((2, 4, 4)).astype("float32"),
        }
        self.axis = 1


class TestConcatCase5(TestConcatOp):
    def init_case(self):
        self.inputs = {
            "x1": np.random.random([1, 16]).astype("float32"),
            "x2": np.random.random([2, 16]).astype("float32"),
            "x3": np.random.random([3, 16]).astype("float32"),
            "x4": np.random.random([4, 16]).astype("float32"),
            "x5": np.random.random([5, 16]).astype("float32"),
        }
        self.axis = 0


if __name__ == "__main__":
    unittest.main()
