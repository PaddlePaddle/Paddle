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


def paddle_slice_assign(data, update, axes, starts, ends, strides):
    assert len(axes) == len(starts) == len(ends) == len(strides)

    # prepare
    for i in range(len(ends)):
        input_len = data.shape[axes[i]]
        if ends[i] < 0:
            ends[i] += input_len
        elif ends[i] > input_len:
            ends[i] = input_len
        if starts[i] < 0:
            starts[i] += input_len
        elif starts[i] > input_len:
            starts[i] = input_len - 1

    # slice & assign
    dims = len(data.shape)
    slices = ['::'] * dims
    for i, axis in enumerate(axes):
        slices[axis] = (
            str(starts[i]) + ':' + str(ends[i]) + ':' + str(strides[i])
        )
    res = data.clone()
    exec(f"res[{','.join(slices)}] = update")
    return res


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestSliceAssignOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "inputs": self.random(
                self.case["inputs_shape"], self.case["dtype"]
            ),
            "assign": self.random(
                self.case["assign_shape"], self.case["dtype"]
            ),
        }
        if self.case["assign_zeros"]:
            self.inputs["assign"] = np.zeros(self.case["assign_shape"]).astype(
                self.case["dtype"]
            )
        self.axes = self.case["axes"]
        self.starts = self.case["starts"]
        self.ends = self.case["ends"]
        self.strides = self.case["strides"]

    def build_paddle_program(self, target):
        inputs = paddle.to_tensor(self.inputs["inputs"], stop_gradient=True)
        assign = paddle.to_tensor(self.inputs["assign"], stop_gradient=True)
        res = paddle_slice_assign(
            inputs, assign, self.axes, self.starts, self.ends, self.strides
        )
        self.paddle_outputs = [res]

    def build_cinn_program(self, target):
        builder = NetBuilder("slice_assign")
        inputs = builder.create_input(
            self.nptype2cinntype(self.inputs["inputs"].dtype),
            self.inputs["inputs"].shape,
            "inputs",
        )
        assign = builder.create_input(
            self.nptype2cinntype(self.inputs["assign"].dtype),
            self.inputs["assign"].shape,
            "assign",
        )
        out = builder.slice_assign(
            inputs,
            assign,
            starts=self.starts,
            ends=self.ends,
            axes=self.axes,
            strides=self.strides,
        )

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [inputs, assign],
            [self.inputs["inputs"], self.inputs["assign"]],
            [out],
        )
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestSliceAssignOpLegacyTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSliceAssignOpLegacyTest"
        self.cls = TestSliceAssignOp
        self.inputs = [
            {
                "inputs_shape": [10, 12],
                "assign_shape": [3, 3],
                "axes": [0, 1],
                "starts": [2, 2],
                "ends": [5, 5],
                "strides": [1, 1],
            },
            {
                "inputs_shape": [10, 12],
                "assign_shape": [5, 5],
                "axes": [0, 1],
                "starts": [1, 2],
                "ends": [6, 1000],
                "strides": [1, 2],
            },
            {
                "inputs_shape": [10, 12],
                "assign_shape": [3, 3],
                "axes": [0, 1],
                "starts": [2, 1],
                "ends": [-1, 7],
                "strides": [3, 2],
            },
            {
                "inputs_shape": [10, 12],
                "assign_shape": [6, 5],
                "axes": [0, 1],
                "starts": [2, 1000],
                "ends": [8, 1],
                "strides": [1, -2],
            },
            {
                "inputs_shape": [10, 12],
                "assign_shape": [4, 3],
                "axes": [0, 1],
                "starts": [-1, -2],
                "ends": [-5, -8],
                "strides": [-1, -2],
            },
            {
                "inputs_shape": [121, 2],
                "assign_shape": [121, 1],
                "axes": [1],
                "starts": [0],
                "ends": [1],
                "strides": [1],
            },
            {
                "inputs_shape": [121, 2],
                "assign_shape": [121, 1],
                "axes": [1],
                "starts": [1],
                "ends": [2],
                "strides": [1],
            },
            {
                "inputs_shape": [121, 2],
                "assign_shape": [121, 1],
                "axes": [1],
                "starts": [1],
                "ends": [0],
                "strides": [-1],
            },
            {
                "inputs_shape": [121, 2, 2],
                "assign_shape": [121, 2, 1],
                "axes": [2],
                "starts": [0],
                "ends": [1],
                "strides": [1],
            },
            {
                "inputs_shape": [121, 2, 2],
                "assign_shape": [121, 2, 1],
                "axes": [2],
                "starts": [1],
                "ends": [2],
                "strides": [1],
            },
            {
                "inputs_shape": [121, 2, 2],
                "assign_shape": [121, 2, 1],
                "axes": [2],
                "starts": [1],
                "ends": [0],
                "strides": [-1],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [
            {
                "assign_zeros": True,
            },
            {
                "assign_zeros": False,
            },
        ]


class TestSliceAssignOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSliceAssignOpShapeTest"
        self.cls = TestSliceAssignOp
        self.inputs = [
            {
                "inputs_shape": [64],
                "assign_shape": [3],
                "axes": [0],
                "starts": [2],
                "ends": [5],
                "strides": [1],
            },
            {
                "inputs_shape": [128, 32],
                "assign_shape": [32, 16],
                "axes": [0, 1],
                "starts": [24, 10],
                "ends": [56, 26],
                "strides": [1, 1],
            },
            {
                "inputs_shape": [32, 10, 64],
                "assign_shape": [8, 4, 16],
                "axes": [0, 1, 2],
                "starts": [24, 4, 0],
                "ends": [32, 8, 64],
                "strides": [1, 1, 4],
            },
            {
                "inputs_shape": [10, 12, 9, 5],
                "assign_shape": [3, 5, 4, 5],
                "axes": [0, 1, 2],
                "starts": [2, 4, 0],
                "ends": [5, 9, 7],
                "strides": [1, 1, 2],
            },
            {
                "inputs_shape": [1],
                "assign_shape": [1],
                "axes": [0],
                "starts": [0],
                "ends": [1],
                "strides": [1],
            },
            {
                "inputs_shape": [1, 1, 1, 1, 1],
                "assign_shape": [1, 1, 1, 1, 1],
                "axes": [0],
                "starts": [0],
                "ends": [1],
                "strides": [1],
            },
            {
                "inputs_shape": [1024, 1, 2],
                "assign_shape": [512, 1, 2],
                "axes": [0],
                "starts": [128],
                "ends": [640],
                "strides": [1],
            },
            {
                "inputs_shape": [2, 4096, 8],
                "assign_shape": [2, 2048, 8],
                "axes": [1],
                "starts": [1024],
                "ends": [3072],
                "strides": [1],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {
                "assign_zeros": False,
            },
        ]


class TestSliceAssignOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSliceAssignOpDtypeTest"
        self.cls = TestSliceAssignOp
        self.inputs = [
            {
                "inputs_shape": [10, 12, 9, 5],
                "assign_shape": [3, 5, 4, 5],
                "axes": [0, 1, 2],
                "starts": [2, 4, 0],
                "ends": [5, 9, 7],
                "strides": [1, 1, 2],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
            {"dtype": "float64"},
            {"dtype": "int32"},
            {"dtype": "int64"},
            {"dtype": "bool"},
        ]
        self.attrs = [
            {
                "assign_zeros": False,
            },
        ]


class TestSliceAssignOpAxesTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSliceAssignOpAxesTest"
        self.cls = TestSliceAssignOp
        self.inputs = [
            {
                "inputs_shape": [128, 32],
                "assign_shape": [128, 16],
                "axes": [1],
                "starts": [10],
                "ends": [26],
                "strides": [1],
            },
            {
                "inputs_shape": [32, 10, 64],
                "assign_shape": [8, 10, 16],
                "axes": [0, 2],
                "starts": [24, 0],
                "ends": [32, 64],
                "strides": [1, 4],
            },
            {
                "inputs_shape": [10, 12, 9, 5],
                "assign_shape": [3, 12, 9, 3],
                "axes": [0, 3],
                "starts": [2, 0],
                "ends": [5, 3],
                "strides": [1, 1],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {
                "assign_zeros": False,
            },
        ]


class TestSliceAssignOpStridesTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSliceAssignOpStridesTest"
        self.cls = TestSliceAssignOp
        self.inputs = [
            {
                "inputs_shape": [128, 32],
                "assign_shape": [8, 16],
                "axes": [0, 1],
                "starts": [0, 0],
                "ends": [128, 32],
                "strides": [16, 2],
            },
            {
                "inputs_shape": [32, 10, 64],
                "assign_shape": [8, 10, 16],
                "axes": [0, 2],
                "starts": [16, 0],
                "ends": [32, 64],
                "strides": [2, 4],
            },
            {
                "inputs_shape": [8, 16, 32, 64, 128],
                "assign_shape": [8, 8, 8, 8, 8],
                "axes": [0, 1, 2, 3, 4],
                "starts": [0, 0, 0, 0, 0],
                "ends": [8, 16, 32, 64, 128],
                "strides": [1, 2, 4, 8, 16],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {
                "assign_zeros": False,
            },
        ]


if __name__ == "__main__":
    TestSliceAssignOpLegacyTest().run()
    TestSliceAssignOpShapeTest().run()
    TestSliceAssignOpDtypeTest().run()
    TestSliceAssignOpAxesTest().run()
    TestSliceAssignOpStridesTest().run()
