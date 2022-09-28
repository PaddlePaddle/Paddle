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

import unittest
import numpy as np
import sys

sys.path.append('..')
from op_test import OpTest, convert_float_to_uint16
import paddle
import paddle.fluid as fluid
from paddle.framework import core
from paddle.fluid.dygraph.base import switch_to_static_graph

paddle.enable_static()


def gather_numpy(x, index, axis):
    x_transpose = np.swapaxes(x, 0, axis)
    tmp_gather = x_transpose[index, ...]
    gather = np.swapaxes(tmp_gather, 0, axis)
    return gather


class TestGatherOp(OpTest):

    def setUp(self):
        self.op_type = "gather"
        self.place = paddle.MLUPlace(0)
        self.__class__.use_mlu = True
        self.python_api = paddle.gather
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.inputs = {
            'X': xnp,
            'Index': np.array(self.index).astype(self.index_type)
        }
        self.outputs = {'Out': self.inputs["X"][self.inputs["Index"]]}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 20)
        self.x_type = "float32"
        self.index = [1, 3, 5]
        self.index_type = "int32"


class TestCase1(TestGatherOp):

    def config(self):
        """
        For one dimension input
        """
        self.x_shape = (100)
        self.x_type = "float32"
        self.index = [1, 3, 5]
        self.index_type = "int32"


class TestCase2(TestGatherOp):

    def config(self):
        """
        For int64_t index type
        """
        self.x_shape = (100)
        self.x_type = "float32"
        self.index = [1, 3, 5]
        self.index_type = "int64"


class API_TestDygraphGather(unittest.TestCase):

    def test_out1(self):
        paddle.disable_static()
        input_1 = np.array([[1, 2], [3, 4], [5, 6]]).astype('int32')
        index_1 = np.array([1, 2])
        input = paddle.to_tensor(input_1)
        index = paddle.to_tensor(index_1)
        output = paddle.fluid.layers.gather(input, index)
        output_np = output.numpy()
        expected_output = np.array([[3, 4], [5, 6]]).astype('int32')
        np.testing.assert_allclose(output_np, expected_output)
        paddle.enable_static()

    def test_out12(self):
        paddle.disable_static()
        input_1 = np.array([[1, 2], [3, 4], [5, 6]]).astype('int32')
        index_1 = np.array([1, 2])
        x = paddle.to_tensor(input_1)
        index = paddle.to_tensor(index_1)
        output = paddle.gather(x, index, axis=0)
        output_np = output.numpy()
        expected_output = gather_numpy(input_1, index_1, axis=0)
        np.testing.assert_allclose(output_np, expected_output)
        paddle.enable_static()

    def test_zero_index(self):
        paddle.disable_static()
        x = paddle.to_tensor([[1, 2], [3, 4]]).astype('int32')
        index = paddle.to_tensor(np.array([]).astype('int64'))
        for axis in range(len(x.shape)):
            out = paddle.gather(x, index, axis)
            expected_shape = list(x.shape)
            expected_shape[axis] = 0
            self.assertEqual(list(out.shape), expected_shape)
        paddle.enable_static()


class TestGathertError(unittest.TestCase):

    def test_error1(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):

            shape = [8, 9, 6]
            x = paddle.fluid.data(shape=shape, dtype='int8', name='x')
            axis = paddle.fluid.data(shape=[1], dtype='float32', name='axis')
            index = paddle.fluid.data(shape=shape, dtype='int32', name='index')
            index_float = paddle.fluid.data(shape=shape,
                                            dtype='float32',
                                            name='index_float')

            def test_x_type():
                paddle.gather(x, index)

            self.assertRaises(TypeError, test_x_type)

            def test_index_type():
                paddle.gather(x, index_float)

            self.assertRaises(TypeError, test_index_type)

            def test_axis_dtype():
                paddle.gather(x, index, axis=1.11)

            self.assertRaises(TypeError, test_axis_dtype)

            def test_axis_dtype1():
                paddle.gather(x, index, axis=axis)

            self.assertRaises(TypeError, test_axis_dtype1)

    def test_error2(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):

            shape = [8, 9, 6]
            x = fluid.data(shape=shape, dtype='int8', name='x')
            index = fluid.data(shape=shape, dtype='int32', name='mask')
            index_float = fluid.data(shape=shape,
                                     dtype='float32',
                                     name='index_float')

            def test_x_type():
                paddle.fluid.layers.gather(x, index)

            self.assertRaises(TypeError, test_x_type)

            def test_index_type():
                paddle.fluid.layers.gather(x, index_float)

            self.assertRaises(TypeError, test_index_type)


if __name__ == "__main__":
    unittest.main()
