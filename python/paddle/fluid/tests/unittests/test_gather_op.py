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
import paddle
import paddle.fluid as fluid


class TestGatherOp(OpTest):
    def setUp(self):
        self.op_type = "gather"
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.inputs = {
            'X': xnp,
            'Index': np.array(self.index).astype(self.index_type)
        }
        self.outputs = {'Out': self.inputs["X"][self.inputs["Index"]]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 20)
        self.x_type = "float64"
        self.index = [1, 3, 5]
        self.index_type = "int32"


class TestCase1(TestGatherOp):
    def config(self):
        """
        For one dimension input
        """
        self.x_shape = (100)
        self.x_type = "float64"
        self.index = [1, 3, 5]
        self.index_type = "int32"


class TestCase2(TestGatherOp):
    def config(self):
        """
        For int64_t index type
        """
        self.x_shape = (100)
        self.x_type = "float64"
        self.index = [1, 3, 5]
        self.index_type = "int64"


class TestCase3(TestGatherOp):
    def config(self):
        """
        For other input type
        """
        self.x_shape = (10, 20)
        self.x_type = "float64"
        self.index = [1, 3, 5]
        self.index_type = "int64"


class TestCase4(TestGatherOp):
    def config(self):
        self.x_shape = (10, 20)
        self.attrs = {'overwrite': False}
        self.x_type = "double"
        self.index = [1, 1]
        self.index_type = "int32"


class TestCase5(TestGatherOp):
    def config(self):
        self.x_shape = (10, 20)
        self.attrs = {'overwrite': False}
        self.x_type = "float64"
        self.index = [1, 1, 3]
        self.index_type = "int32"


class TestCase6(TestGatherOp):
    def config(self):
        self.x_shape = (10, 20)
        self.attrs = {'overwrite': True}
        self.x_type = "float64"
        self.index = [1, 3]
        self.index_type = "int32"


class API_TestGather(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data1 = fluid.layers.data('data1', shape=[-1, 2], dtype='float64')
            index = fluid.layers.data('index', shape=[-1, 1], dtype='float64')
            out = paddle.gather(data1, index)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            input = np.array([[1, 2], [3, 4], [5, 6]])
            index_1 = np.array([1, 2])
            result, = exe.run(feed={"data1": input,
                                    "index": index_1},
                              fetch_list=[out])
            expected_output = np.array([[3, 4], [5, 6]])
        self.assertTrue(np.allclose(result, expected_output))


class API_TestDygraphGather(unittest.TestCase):
    def test_out(self):
        with fluid.dygraph.guard():
            input_1 = np.array([[1, 2], [3, 4], [5, 6]])
            index_1 = np.array([1, 2])
            input = fluid.dygraph.to_variable(input_1)
            index = fluid.dygraph.to_variable(index_1)
            output = paddle.fluid.layers.gather(input, index)
            output_np = output.numpy()
            expected_output = np.array([[3, 4], [5, 6]])
        self.assertTrue(np.allclose(output_np, expected_output))


if __name__ == "__main__":
    unittest.main()
