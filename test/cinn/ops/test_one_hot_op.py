#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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


from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle
import paddle.nn.functional as F


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestOneHotOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"], dtype=self.case["x_dtype"]
        )
        self.dtype = "float32"

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = F.one_hot(x, num_classes=self.case["depth"])

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("one_hot")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]),
            self.case["x_shape"],
            "x",
        )
        on_value = builder.fill_constant(
            [1], 1, 'on_value', dtype=self.case["x_dtype"]
        )
        off_value = builder.fill_constant(
            [1], 0, 'off_value', dtype=self.case["x_dtype"]
        )
        out = builder.one_hot(
            x,
            on_value,
            off_value,
            depth=self.case["depth"],
            axis=self.case["axis"],
            dtype=self.dtype,
        )

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        max_relative_error = (
            self.case["max_relative_error"]
            if "max_relative_error" in self.case
            else 1e-5
        )
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestOneHotOpTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestOneHotOpTest"
        self.cls = TestOneHotOp
        self.inputs = [
            {
                "x_shape": [1],
                "depth": 10,
                "axis": -1,
            },
            {
                "x_shape": [1024],
                "depth": 10,
                "axis": -1,
            },
            {
                "x_shape": [32, 64],
                "depth": 10,
                "axis": -1,
            },
            {
                "x_shape": [16, 8, 4],
                "depth": 10,
                "axis": -1,
            },
            {
                "x_shape": [16, 8, 4, 2],
                "depth": 10,
                "axis": -1,
            },
            {
                "x_shape": [16, 8, 4, 2, 1],
                "depth": 10,
                "axis": -1,
            },
        ]
        self.dtypes = [
            {
                "x_dtype": "int32",
            },
            {
                "x_dtype": "int64",
            },
        ]
        self.attrs = []


if __name__ == "__main__":
    TestOneHotOpTest().run()
