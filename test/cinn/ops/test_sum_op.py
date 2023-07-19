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

from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestSumOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        shapes = self.case["shapes"]
        dtype = self.case["dtype"]
        self.inputs = []
        for shape in shapes:
            self.inputs.append(self.random(shape, dtype))

    def build_paddle_program(self, target):
        inputs = []
        for input in self.inputs:
            inputs.append(paddle.to_tensor(input, stop_gradient=True))
        out = paddle.add_n(inputs)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("sum")
        cinn_inputs = []
        for id, input in enumerate(self.inputs):
            cinn_input = builder.create_input(
                self.nptype2cinntype(input.dtype),
                input.shape,
                "input_" + str(id),
            )
            cinn_inputs.append(cinn_input)
        out = builder.sum(cinn_inputs)
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, cinn_inputs, self.inputs, [out]
        )
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSumOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSumOpShapeTest"
        self.cls = TestSumOp
        self.inputs = [
            {
                "shapes": [[64]] * 2,
            },
            {
                "shapes": [[64, 32]] * 3,
            },
            {
                "shapes": [[64, 1]] * 4,
            },
            {
                "shapes": [[64, 32, 128]] * 2,
            },
            {
                "shapes": [[1, 32, 128]] * 2,
            },
            {
                "shapes": [[64, 32, 16, 32]] * 2,
            },
            {
                "shapes": [[64, 32, 1, 32]] * 2,
            },
            {
                "shapes": [[64, 32, 16, 1, 128]] * 2,
            },
            {
                "shapes": [[1]] * 2,
            },
            {
                "shapes": [[1, 1]] * 2,
            },
            {
                "shapes": [[1, 1, 1]] * 3,
            },
            {
                "shapes": [[1, 1, 1, 1]] * 3,
            },
            {
                "shapes": [[1, 1, 1, 1, 1]] * 4,
            },
            {
                "shapes": [[1, 1, 1024, 1, 1]] * 4,
            },
            {
                "shapes": [[65536]] * 1,
            },
            {
                "shapes": [[131072]] * 2,
            },
            {
                "shapes": [[1048576]] * 3,
            },
            {
                "shapes": [[64, 32, 16, 8, 4]] * 4,
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = []


class TestSumOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSumOpDtypeTest"
        self.cls = TestSumOp
        self.inputs = [
            {
                "shapes": [[64, 1, 128]] * 2,
            },
            {
                "shapes": [[64, 32, 1]] * 2,
            },
        ]
        self.dtypes = [
            {"dtype": "float16"},
            {"dtype": "float32"},
            {"dtype": "float64"},
            {"dtype": "int32"},
            {"dtype": "int64"},
        ]
        self.attrs = [{"axes": []}]
        self.attrs = []


if __name__ == "__main__":
    TestSumOpShapeTest().run()
    TestSumOpDtypeTest().run()
