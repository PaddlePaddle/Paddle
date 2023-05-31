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
class TestScatterAddOp(OpTest):
    def setUp(self):
        self.init_case()
        self.target = DefaultNVGPUTarget()

    def init_case(self):
        self.axis = 0
        self.inputs = {
            "x": np.random.random([10, 5]).astype("float32"),
            "y": np.random.random([5, 5]).astype("float32"),
            "index": np.array([0, 5, 0, 9, 0]).astype("int32"),
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=True)

        pos_axis = self.axis
        if pos_axis < 0:
            pos_axis += len(x.shape)

        if pos_axis == 0:
            index_nd = []
            for i in range(len(self.inputs["index"])):
                index_nd.append([self.inputs["index"][i]])
        elif pos_axis == 1:
            index_nd = []
            for i in range(self.inputs['x'].shape[0]):
                index_nd.append([])
                for j in range(len(self.inputs["index"])):
                    index_nd[i].append([i, self.inputs["index"][j]])
        elif pos_axis == 2:
            index_nd = []
            for i in range(self.inputs['x'].shape[0]):
                index_nd.append([])
                for j in range(self.inputs['x'].shape[1]):
                    index_nd[i].append([])
                    for k in range(len(self.inputs["index"])):
                        index_nd[i][j].append([i, j, self.inputs["index"][k]])
        else:
            self.assertTrue(False, f"Axis {pos_axis} No Implement")

        index = paddle.to_tensor(index_nd, stop_gradient=True)
        res = paddle.scatter_nd_add(x, index, y)
        self.paddle_outputs = [res]

    def build_cinn_program(self, target):
        builder = NetBuilder("scatter_add")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(32), self.inputs["y"].shape, "y")
        index = builder.create_input(
            Int(32), self.inputs["index"].shape, "index"
        )
        out = builder.scatter_add(x, y, index, self.axis)

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [x, y, index],
            [self.inputs["x"], self.inputs["y"], self.inputs["index"]],
            [out],
        )

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestScatterAddCase1(TestScatterAddOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10, 5]).astype("float32"),
            "y": np.random.random([10, 3]).astype("float32"),
            "index": np.random.randint(0, 5, size=3).astype("int32"),
        }
        self.axis = 1


class TestScatterAddCase2(TestScatterAddOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10, 5, 5]).astype("float32"),
            "y": np.random.random([10, 5, 3]).astype("float32"),
            "index": np.random.randint(0, 5, size=3).astype("int32"),
        }
        self.axis = -1


class TestScatterAddCase3(TestScatterAddOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10, 5, 5]).astype("float32"),
            "y": np.random.random([10, 3, 5]).astype("float32"),
            "index": np.random.randint(0, 5, size=3).astype("int32"),
        }
        self.axis = 1


class TestScatterAddCase4(TestScatterAddOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10]).astype("float32"),
            "y": np.random.random([1]).astype("float32"),
            "index": np.random.randint(0, 10, size=1).astype("int32"),
        }
        self.axis = -1


class TestScatterAddCase5(TestScatterAddOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10, 5]).astype("float32"),
            "y": np.random.random([3, 5]).astype("float32"),
            "index": np.random.randint(0, 10, size=3).astype("int32"),
        }
        self.axis = 0


class TestScatterAddCase6(TestScatterAddOp):
    def build_cinn_program(self, target):
        builder = NetBuilder("scatter_add")
        x = builder.create_input(Float(64), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(32), self.inputs["y"].shape, "y")
        x1 = builder.cast(x, dtype="float32")  # newly added
        index = builder.create_input(
            Int(32), self.inputs["index"].shape, "index"
        )
        out = builder.scatter_add(x1, y, index, self.axis)

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [x, y, index],
            [
                self.inputs["x"].astype("float64"),
                self.inputs["y"],
                self.inputs["index"],
            ],
            [out],
        )

        self.cinn_outputs = [res[0]]


class TestScatterAddCase7(TestScatterAddOp):
    def build_cinn_program(self, target):
        builder = NetBuilder("scatter_add")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(64), self.inputs["y"].shape, "y")
        y1 = builder.cast(y, dtype="float32")  # newly added
        index = builder.create_input(
            Int(32), self.inputs["index"].shape, "index"
        )
        out = builder.scatter_add(x, y1, index, self.axis)

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [x, y, index],
            [
                self.inputs["x"],
                self.inputs["y"].astype("float64"),
                self.inputs["index"],
            ],
            [out],
        )

        self.cinn_outputs = [res[0]]


class TestScatterAddCase8(TestScatterAddCase7):
    def init_case(self):
        self.axis = 0
        self.inputs = {
            "x": np.random.random([10, 5]).astype("float32"),
            "y": np.random.random([10, 5]).astype("float32"),
            "index": np.array([0, 5, 0, 9, 0, 1, 2, 3, 4, 5]).astype("int32"),
        }


class TestScatterAddOp9(TestScatterAddOp):
    def setUp(self):
        self.init_case()
        self.target = DefaultNVGPUTarget()

    def init_case(self):
        self.axis = 0
        self.inputs = {
            "x": np.random.random([10, 5]).astype("float64"),
            "y": np.random.random([5, 5]).astype("float64"),
            "index": np.array([0, 5, 0, 9, 0]).astype("int32"),
        }

    def build_cinn_program(self, target):
        builder = NetBuilder("scatter_add")
        x = builder.create_input(Float(64), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(64), self.inputs["y"].shape, "y")
        index = builder.create_input(
            Int(32), self.inputs["index"].shape, "index"
        )
        out = builder.scatter_add(x, y, index, self.axis)

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [x, y, index],
            [self.inputs["x"], self.inputs["y"], self.inputs["index"]],
            [out],
        )

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
