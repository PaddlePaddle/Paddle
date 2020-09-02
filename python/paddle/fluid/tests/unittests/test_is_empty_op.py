#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle.fluid as fluid


class TestEmpty(OpTest):
    def setUp(self):
        self.op_type = "is_empty"
        self.inputs = {'X': np.array([1, 2, 3])}
        self.outputs = {'Out': np.array([False])}

    def test_check_output(self):
        self.check_output()


class TestNotEmpty(TestEmpty):
    def setUp(self):
        self.op_type = "is_empty"
        self.inputs = {'X': np.array([])}
        self.outputs = {'Out': np.array([True])}


class TestIsEmptyOpError(unittest.TestCase):
    def test_errors(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input_data = np.random.random((3, 2)).astype("float64")

            def test_Variable():
                # the input type must be Variable
                fluid.layers.is_empty(x=input_data)

            self.assertRaises(TypeError, test_Variable)

            def test_cond_Variable():
                # cond type must be Variable or None
                x2 = fluid.layers.data(name="x2", shape=[3, 2], dtype="float32")
                cond_data = np.random.random((3, 2)).astype("float32")
                fluid.layers.is_empty(x=x2, cond=cond_data)

            self.assertRaises(TypeError, test_cond_Variable)

            def test_type():
                # dtype must be float32, float64, int32, int64
                x3 = fluid.layers.data(
                    name="x3", shape=[4, 32, 32], dtype="bool")
                res = fluid.layers.is_empty(x=x3)

            self.assertRaises(TypeError, test_type)

            def test_cond_type():
                # cond dtype must be bool.
                x4 = fluid.layers.data(name="x4", shape=[3, 2], dtype="float32")
                cond = fluid.layers.data(
                    name="cond", shape=[1], dtype="float32")
                fluid.layers.is_empty(x=x4, cond=cond)

            self.assertRaises(TypeError, test_cond_type)


if __name__ == "__main__":
    unittest.main()
