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

import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle

INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)
INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)


def count_leading_zeros(integer, dtype):
    if dtype == "int32":
        bits = 32
    elif dtype == "int64":
        bits = 64
    else:
        raise NotImplementedError
    if integer < 0:
        return 0
    mask = 1 << (bits - 1)
    integer &= mask - 1
    clz = 0
    while mask > 0 and integer & mask == 0:
        clz += 1
        mask >>= 1
    return clz


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestClzOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        dtype = self.case["dtype"]
        low = INT32_MIN if dtype == "int32" else INT64_MIN
        high = INT32_MAX if dtype == "int32" else INT64_MAX
        x = self.random(self.case["shape"], dtype, low=low, high=high)
        y = [count_leading_zeros(num, dtype) for num in x.reshape(-1).tolist()]
        self.inputs = {"x": x}
        self.outputs = {"y": np.array(y).reshape(x.shape).astype(dtype)}

    def build_paddle_program(self, target):
        y = paddle.to_tensor(self.outputs["y"], stop_gradient=False)
        self.paddle_outputs = [y]

    def build_cinn_program(self, target):
        builder = NetBuilder("clz")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.clz(x)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestClzOpShapeDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestClzOpShapeDtype"
        self.cls = TestClzOp
        self.inputs = [
            {
                "shape": [10],
            },
            {
                "shape": [8, 5],
            },
            {
                "shape": [10, 3, 5],
            },
            {
                "shape": [80, 40, 5, 7],
            },
            {
                "shape": [80, 1, 5, 7],
            },
            {
                "shape": [80, 3, 1024, 7],
            },
            {
                "shape": [10, 5, 2048, 2],
            },
            {
                "shape": [1],
            },
            {
                "shape": [512],
            },
            {
                "shape": [1024],
            },
            {
                "shape": [2048],
            },
            {
                "shape": [1, 1, 1, 1],
            },
        ]
        self.dtypes = [
            {"dtype": "int32"},
            {"dtype": "int64"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestClzOpShapeDtype().run()
