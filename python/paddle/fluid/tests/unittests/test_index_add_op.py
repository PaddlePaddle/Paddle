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
from op_test import OpTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

np.random.seed(102)


def index_add_np(data, axis, index, added_value):
    if isinstance(index, np.ndarray):
        index = list(index.flatten())
    outer_loop = int(np.prod(data.shape[:axis]))
    x_reshape = [outer_loop] + list(data.shape[axis:])
    x_np_reshape = data.reshape(tuple(x_reshape))
    for i in range(outer_loop):
        for j in index:
            x_np_reshape[i, j] += added_value
    return x_np_reshape.reshape(data.shape)


# def index_fill_grad_np(data, axis, index):
#     outer_loop = int(np.prod(data.shape[:axis]))
#     x_reshape = [outer_loop] + list(data.shape[axis:])
#     x_np_reshape = data.reshape(tuple(x_reshape))
#     dim = data.shape[axis]
#     for i in range(outer_loop):
#         for j in range(dim):
#             x_np_reshape[i, j] = 0 if j in index else 1
#     return x_np_reshape.reshape(data.shape)


class TestIndexAddOp(OpTest):
    def setUp(self):
        self.python_api = paddle.index_add
        self.op_type = "index_add"
        self.init_data()
        self.inputs = {'X': self.x_np, 'Index': self.index_np}
        self.attrs = {'axis': self.axis, 'added_value': self.added_value}
        self.outputs = {'Out': self.out_np}

    def init_data(self):
        self.axis = 1
        self.x_type = np.float64
        self.index_type = np.int64
        self.x_shape = (4, 5, 6)
        self.index_size = 4
        self.added_value = 9.0
        self.x_np = np.random.random(self.x_shape).astype(self.x_type)
        self.index_np = np.random.randint(
            low=0, high=self.x_shape[self.axis], size=self.index_size)
        self.out_np = index_add_np(self.x_np, self.axis, self.index_np,
                                    self.added_value)

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', check_eager=True)


class API_IndexAdd(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()
        self.executed_api()
        self.init_data()

    def init_data(self):
        self.data_np = np.random.random((3, 4, 5)).astype(np.float32)
        self.axis = 0
        self.added_value = 7.0
        self.index = np.asarray([1, 2]).astype(np.int64)

    def executed_api(self):
        self.op_run = paddle.index_add

    def test_out(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', np.asarray(self.data_np).shape)
            idx = paddle.fluid.data(
                'Index', np.asarray(self.index).shape, dtype="int64")
            result_pd = self.op_run(
                x, idx, axis=self.axis, added_value=self.added_value)
            exe = paddle.static.Executor(self.place)
            result, = exe.run(feed={"X": self.data_np,
                                    "Index": self.index},
                              fetch_list=[result_pd])
            result_np = index_add_np(self.data_np, self.axis, self.index,
                                      self.added_value)
            self.assertTrue(np.allclose(result_np, result))


class API_TestStaticIndexAdd_(API_IndexAdd):
    def executed_api(self):
        self.op_run = paddle.index_add_

    def init_data(self):
        self.data_np = np.random.random((2, 3, 4, 5)).astype(np.float32)
        self.axis = 3
        self.added_value = 8.0
        self.index = np.asarray([0, 2, 3]).astype(np.int64)
        self.place = paddle.CPUPlace()


class API_TestDygraphIndexAdd(unittest.TestCase):
    def setUp(self):
        self.executed_api()
        self.init_data()
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def init_data(self):
        self.data_np = np.random.random((3, 4, 5)).astype(np.float32)
        self.axis = 0
        self.added_value = 11.0
        self.index = np.asarray([1, 2]).astype(np.int64)

    def executed_api(self):
        self.op_run = paddle.index_add

    def test_out(self):
        paddle.disable_static(self.place)
        input_1 = paddle.to_tensor(self.data_np)
        input = paddle.to_tensor(input_1)
        index = paddle.to_tensor(self.index)
        output = self.op_run(
            input, index, axis=self.axis, added_value=self.added_value)
        out_np = output.numpy()
        expected_out = index_add_np(self.data_np, self.axis, self.index,
                                     self.added_value)
        self.assertTrue(np.allclose(expected_out, out_np))


class API_TestDygraphIndexAddInplace(API_TestDygraphIndexAdd):
    def executed_api(self):
        self.op_run = paddle.index_add_

    def init_data(self):
        self.data_np = np.random.random((3, 4, 5)).astype(np.float32)
        self.axis = 2
        self.added_value = 12.0
        self.index = np.asarray([0, 2]).astype(np.int32)


class API_TestDygraphIndexAdd2(API_TestDygraphIndexAdd):
    def init_data(self):
        self.data_np = np.random.random((10)).astype(np.float32)
        self.axis = 0
        self.added_value = 13.0
        self.index = np.asarray([0]).astype(np.int32)


class API_TestDygraphIndexAdd3(API_TestDygraphIndexAdd):
    def init_data(self):
        self.data_np = np.random.random((3, 4, 5, 6)).astype(np.float64)
        self.axis = 2
        self.added_value = paddle.to_tensor([14.0])
        self.index = np.asarray([0, 1, 2, 3]).astype(np.int64)


class API_IndexAdd2(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_errors(self):
        def error_axis_dtype():
            with paddle.fluid.dygraph.guard():
                x = paddle.rand((2, 3))
                index = paddle.to_tensor([0])
                paddle.index_add(x, index, axis=0.5, added_value=0.1)

        def error_axis_range():
            with paddle.fluid.dygraph.guard():
                x = paddle.rand((2, 3))
                index = paddle.to_tensor([0])
                paddle.index_add(x, index, axis=10, added_value=0.1)

        def error_index_dtype():
            with paddle.fluid.dygraph.guard():
                x = paddle.rand((2, 3))
                paddle.index_add(x, [0, 1], axis=0, added_value=0.1)

        def error_index_dtype2():
            with paddle.fluid.dygraph.guard():
                x = paddle.rand((2, 3))
                index = paddle.to_tensor([0.2, 0.3])
                paddle.index_add(x, index, axis=1, added_value=0.1)

        def error_index_dtype3():
            with paddle.fluid.dygraph.guard():
                x = paddle.rand((2, 3))
                index = paddle.to_tensor([])
                paddle.index_add(x, index, axis=1, added_value=0.1)

        def error_index_value():
            with paddle.fluid.dygraph.guard():
                x = paddle.rand((2, 3))
                index = paddle.to_tensor([[1], [2]])
                added_value = paddle.to_tensor([[9.5], [9.8]])
                paddle.index_add(x, index, axis=1, added_value=added_value)

        def error_added_value():
            with paddle.fluid.dygraph.guard():
                x = paddle.rand((2, 3))
                index = paddle.to_tensor([1])
                added_value = paddle.to_tensor([[9.5], [9.8]])
                paddle.index_add(x, index, axis=1, added_value=added_value)

        self.assertRaises(ValueError, error_axis_dtype)
        self.assertRaises(ValueError, error_axis_range)
        self.assertRaises(TypeError, error_index_dtype)
        self.assertRaises(TypeError, error_index_dtype2)
        self.assertRaises(TypeError, error_index_dtype3)
        self.assertRaises(ValueError, error_index_value)
        self.assertRaises(ValueError, error_added_value)

    # def test_check_grad(self):
    #     paddle.disable_static(place=self.place)
    #     axis = 1
    #     x_np = np.random.uniform(-1, 1, (2, 3, 4)).astype(np.float64)
    #     index_np = [0, 2]
    #     x_tensor = paddle.to_tensor(x_np, stop_gradient=False)
    #     idx_tensor = paddle.to_tensor(index_np)

    #     y = paddle.index_add(x_tensor, idx_tensor, axis=axis, added_value=0.5)
    #     dx = paddle.grad(y, x_tensor)[0].numpy()
    #     np_grad = index_fill_grad_np(x_np, axis, index_np)
    #     self.assertTrue(np.allclose(np_grad, dx, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
