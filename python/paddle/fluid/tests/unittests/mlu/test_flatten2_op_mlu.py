#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid as fluid
import paddle
import sys
sys.path.append("..")
from op_test import OpTest
paddle.enable_static()


class TestFlattenOp(OpTest):
    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.op_type = "flatten2"
        self.init_test_case()
        self.inputs = {"X": np.random.random(self.in_shape).astype("float64")}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": np.random.random(self.in_shape).astype("float32")
        }

    def test_check_output(self):
        self.check_output_with_place(self.place, no_check_set=["XShape"])

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")

    def init_test_case(self):
        self.in_shape = (3, 2, 4, 5)
        self.axis = 1
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axis": self.axis}


class TestFlattenOp1(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.axis = 0
        self.new_shape = (1, 120)


class TestFlattenOpWithDefaultAxis(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (10, 2, 2, 3)
        self.new_shape = (10, 12)

    def init_attrs(self):
        self.attrs = {}


class TestFlattenOpSixDims(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 3, 2, 4, 4)
        self.axis = 4
        self.new_shape = (36, 16)


class TestStaticFlattenInferShapePythonAPI(unittest.TestCase):
    def execute_api(self, x, axis=1):
        return fluid.layers.flatten(x, axis=axis)

    def test_static_api(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            x = paddle.static.data(
                name="x", shape=[-1, 3, -1, -1], dtype='float32')
            out = self.execute_api(x, axis=2)
        self.assertTrue((-1, -1) == out.shape)


class TestFlatten2OpError(unittest.TestCase):
    def test_errors(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input_data = np.random.random((3, 2, 4, 5)).astype("float64")

        def test_Variable():
            # the input type must be Variable
            fluid.layers.flatten(input_data, axis=1)

        self.assertRaises(TypeError, test_Variable)

        def test_type():
            # dtype must be float32, float64, int8, int32, int64, uint8.
            x2 = fluid.layers.data(
                name='x2', shape=[3, 2, 4, 5], dtype='float16')
            fluid.layers.flatten(x2, axis=1)

        self.assertRaises(TypeError, test_type)


if __name__ == "__main__":
    unittest.main()
