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
class TestConcatOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {}
        self.axis = self.case["axis"]
        dtype = self.case["dtype"]
        shapes = self.case["shapes"]
        for i, shape in enumerate(shapes):
            name = "x" + str(i)
            self.inputs[name] = self.random(shape, dtype)

    def paddle_inputs(self, inputs):
        return [
            paddle.to_tensor(data, stop_gradient=True)
            for _, data in inputs.items()
        ]

    def cinn_inputs(self, builder, inputs):
        return [
            builder.create_input(
                self.nptype2cinntype(data.dtype), data.shape, name
            )
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


class TestConcatOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestConcatOpShape"
        self.cls = TestConcatOp
        self.inputs = [
            {
                "shapes": [[10], [6]],
            },
            {
                "shapes": [[8, 5], [8, 5]],
            },
            {
                "shapes": [[10, 3, 5], [4, 3, 5]],
            },
            {
                "shapes": [[80, 40, 5, 7], [20, 40, 5, 7]],
            },
            {
                "shapes": [[80, 1, 5, 7], [8, 1, 5, 7]],
            },
            {
                "shapes": [[80, 3, 1024, 7], [100, 3, 1024, 7]],
            },
            {
                "shapes": [[1, 5, 1024, 2048], [2, 5, 1024, 2048]],
            },
            {
                "shapes": [[1], [1]],
            },
            {
                "shapes": [[512], [512]],
            },
            {
                "shapes": [[1024], [512]],
            },
            {
                "shapes": [[2048], [4096]],
            },
            {
                "shapes": [[1, 1, 1, 1], [1, 1, 1, 1]],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {"axis": 0},
        ]


class TestConcatOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestConcatOpDtype"
        self.cls = TestConcatOp
        self.inputs = [
            {
                "shapes": [[10], [6]],
            },
            {
                "shapes": [[8, 5], [8, 5]],
            },
            {
                "shapes": [[10, 3, 5], [4, 3, 5]],
            },
            {
                "shapes": [[80, 40, 5, 7], [20, 40, 5, 7]],
            },
        ]
        self.dtypes = [
            {"dtype": "float16"},
            {"dtype": "float32"},
            {"dtype": "float64"},
            {"dtype": "bool"},
            {"dtype": "uint8"},
            {"dtype": "int8"},
            {"dtype": "int32"},
            {"dtype": "int64"},
        ]
        self.attrs = [
            {"axis": 0},
        ]


class TestConcatOpMultipleInputs(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestConcatOpMultipleInputs"
        self.cls = TestConcatOp
        self.inputs = [
            # 1D tensor with 1~4 inputs
            {"shapes": [[10]], "axis": 0},
            {"shapes": [[10], [6]], "axis": 0},
            {"shapes": [[10], [6], [8]], "axis": 0},
            {"shapes": [[10], [6], [10], [6]], "axis": 0},
            # 2D tensor with 1~4 inputs
            {"shapes": [[8, 5]], "axis": 1},
            {"shapes": [[8, 5], [8, 8]], "axis": 1},
            {"shapes": [[8, 5], [8, 5], [16, 5]], "axis": 0},
            {"shapes": [[8, 5], [8, 5], [8, 5], [8, 5]], "axis": 0},
            # 3D tensor with 1~4 inputs
            {"shapes": [[10, 3, 5]], "axis": 0},
            {"shapes": [[10, 3, 5], [10, 7, 5]], "axis": 1},
            {"shapes": [[10, 3, 5], [10, 3, 6], [10, 3, 7]], "axis": 2},
            {"shapes": [[10, 3, 5], [4, 3, 5], [2, 3, 5]], "axis": 0},
            # 4D tensor with 1~4 inputs
            {"shapes": [[80, 1, 5, 7]], "axis": 0},
            {"shapes": [[80, 1, 5, 7], [80, 79, 5, 7]], "axis": 1},
            {
                "shapes": [[80, 1, 50, 7], [80, 1, 5, 7], [80, 1, 10, 7]],
                "axis": 2,
            },
            {
                "shapes": [
                    [80, 1, 5, 17],
                    [80, 1, 5, 27],
                    [80, 1, 5, 37],
                    [80, 1, 5, 47],
                ],
                "axis": 3,
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = []


class TestConcatOpAttrs(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestConcatOpAttrs"
        self.cls = TestConcatOp
        self.inputs = [
            # 1D tensor
            {"shapes": [[10], [8]], "axis": 0},
            {"shapes": [[10], [6]], "axis": -1},
            # 2D tensor
            {"shapes": [[8, 5], [10, 5]], "axis": 0},
            {"shapes": [[8, 5], [8, 8]], "axis": 1},
            # 3D tensor
            {"shapes": [[10, 3, 5], [10, 3, 5]], "axis": 0},
            {"shapes": [[10, 3, 5], [10, 7, 5]], "axis": 1},
            {"shapes": [[10, 3, 15], [10, 3, 5]], "axis": 2},
            {"shapes": [[10, 3, 7], [10, 3, 5]], "axis": -1},
            {"shapes": [[10, 3, 5], [10, 7, 5]], "axis": -2},
            {"shapes": [[10, 7, 5], [20, 7, 5]], "axis": -3},
            # 4D tensor
            {"shapes": [[80, 1, 5, 7], [80, 1, 5, 7]], "axis": 0},
            {"shapes": [[80, 1, 5, 7], [80, 79, 5, 7]], "axis": 1},
            {"shapes": [[80, 1, 5, 7], [80, 1, 10, 7]], "axis": 2},
            {"shapes": [[80, 1, 5, 7], [80, 1, 5, 7]], "axis": 3},
            {"shapes": [[80, 1, 5, 7], [80, 1, 5, 13]], "axis": -1},
            {"shapes": [[80, 1, 5, 7], [80, 1, 5, 7]], "axis": -2},
            {"shapes": [[80, 15, 5, 7], [80, 5, 5, 7]], "axis": -3},
            {"shapes": [[80, 1, 5, 7], [20, 1, 5, 7]], "axis": -4},
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestConcatOpShape().run()
    TestConcatOpDtype().run()
    TestConcatOpMultipleInputs().run()
    TestConcatOpAttrs().run()
