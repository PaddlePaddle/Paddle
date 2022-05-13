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


def index_fill_np(data, axis, index, fill_value):
    if isinstance(index, np.ndarray):
        index = list(index.flatten())
    outer_loop = int(np.prod(data.shape[:axis]))
    x_reshape = [outer_loop] + list(data.shape[axis:])
    x_np_reshape = data.reshape(tuple(x_reshape))
    for i in range(outer_loop):
        for j in index:
            x_np_reshape[i, j] = fill_value
    return x_np_reshape.reshape(data.shape)


def index_fill_x_grad_np(data, axis, index):
    outer_loop = int(np.prod(data.shape[:axis]))
    x_reshape = [outer_loop] + list(data.shape[axis:])
    x_np_reshape = data.reshape(tuple(x_reshape))
    dim = data.shape[axis]
    for i in range(outer_loop):
        for j in range(dim):
            x_np_reshape[i, j] = 0 if j in index else 1
    return x_np_reshape.reshape(data.shape)


def index_fill_fill_value_grad_np(data, axis, index):
    outer_loop = int(np.prod(data.shape[:axis]))
    stride = np.prod(data.shape) / outer_loop / data.shape[axis]
    dim = data.shape[axis]
    fill_value_grad = 0
    for i in range(outer_loop):
        for j in range(dim):
            fill_value_grad += 1 if j in index else 0
    return np.asarray(fill_value_grad * stride)


class TestIndexFillOp(OpTest):
    def setUp(self):
        self.init_data()
        self.bind_op_values()
        self.place = self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def init_data(self):
        self.x_type = np.float64
        self.index_type = np.int64
        self.x_shape = (4, 5, 6)
        self.index_size = 4
        self.fill_value = 9.0
        self.x_np = np.random.random(self.x_shape).astype(self.x_type)
        self.axis = 1
        self.index_np = np.random.randint(
            low=0, high=self.x_shape[self.axis], size=self.index_size)
        self.index_np = list(self.index_np)
        self.out_np = index_fill_np(self.x_np, self.axis, self.index_np,
                                    self.fill_value)
        self.outputs = {'Out': self.out_np}

    def bind_op_values(self):
        self.op_type = "index_fill"
        self.python_api = paddle.index_fill
        self.inputs = {'X': self.x_np}
        self.attrs = {
            'index': self.index_np,
            'axis': self.axis,
            'fill_value': self.fill_value
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestIndexFillTensorOp(TestIndexFillOp):
    def bind_op_values(self):
        self.op_type = "index_fill_tensor"
        self.fill_value = np.asarray([9.0])
        self.inputs = {'X': self.x_np, 'FillValue': self.fill_value}
        self.attrs = {'index': self.index_np, 'axis': self.axis}

    def test_check_grad_normal(self):
        self.check_grad(['X', 'FillValue'], 'Out', max_relative_error=0.5)

    # def test_check_grad_ingore_x(self):
    #     self.check_grad_with_place(
    #         self.place, ['X'], 'Out', no_grad_set=set("FillValue"))
    #
    # def test_check_grad_ingore_fill_value(self):
    #     self.check_grad_with_place(
    #         self.place, ['FillValue'], 'Out', no_grad_set=set('X'), max_relative_error=0.5)


class API_IndexFill(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()
        self.executed_api()
        self.init_data()

    def init_data(self):
        self.data_np = np.random.random((3, 4, 5)).astype(np.float32)
        self.axis = 0
        self.fill_value = 7.0
        self.index = [1, 2]

    def executed_api(self):
        self.op_run = paddle.index_fill

    def test_out(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', np.asarray(self.data_np).shape)
            result_np = index_fill_np(self.data_np, self.axis, self.index,
                                      self.fill_value)

            result_pd = self.op_run(
                x, index=self.index, axis=self.axis, fill_value=self.fill_value)
            exe = paddle.static.Executor(self.place)
            result_value, = exe.run(feed={"X": self.data_np},
                                    fetch_list=[result_pd])
            self.assertTrue(np.allclose(result_np, result_value))

            result_pd = self.op_run(
                x, index=self.index, axis=self.axis, fill_value=self.fill_value)
            exe = paddle.static.Executor(self.place)
            rresult_tensor, = exe.run(feed={"X": self.data_np},
                                      fetch_list=[result_pd])
            self.assertTrue(np.allclose(result_np, rresult_tensor))


class API_TestStaticIndexFill_(API_IndexFill):
    def executed_api(self):
        self.op_run = paddle.index_fill_

    def init_data(self):
        self.data_np = np.random.random((2, 3, 4, 5)).astype(np.float32)
        self.axis = 3
        self.fill_value = 8.0
        self.index = [0, 2, 3]
        self.place = paddle.CPUPlace()


class API_TestDygraphIndexFill(unittest.TestCase):
    def setUp(self):
        self.executed_api()
        self.init_data()
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def init_data(self):
        self.data_np = np.random.random((3, 4, 5)).astype(np.float32)
        self.axis = 0
        self.fill_value = 11.0
        self.index = [1, 2]

    def executed_api(self):
        self.op_run = paddle.index_fill

    def test_out(self):
        paddle.disable_static(self.place)
        input_1 = paddle.to_tensor(self.data_np)
        input = paddle.to_tensor(input_1)
        output = self.op_run(
            input, index=self.index, axis=self.axis, fill_value=self.fill_value)
        out_np = output.numpy()
        expected_out = index_fill_np(self.data_np, self.axis, self.index,
                                     self.fill_value)
        self.assertTrue(np.allclose(expected_out, out_np))


class API_TestDygraphIndexFillInplace(API_TestDygraphIndexFill):
    def executed_api(self):
        self.op_run = paddle.index_fill_

    def init_data(self):
        self.data_np = np.random.random((3, 4, 5)).astype(np.float32)
        self.axis = paddle.to_tensor(2)
        self.fill_value = 12.0
        self.index = paddle.to_tensor([0, 2])


class API_TestDygraphIndexFill2(API_TestDygraphIndexFill):
    def init_data(self):
        self.data_np = np.random.random((10)).astype(np.int32)
        self.axis = 0
        self.fill_value = 13
        self.index = paddle.to_tensor([0])


class API_TestDygraphIndexFill3(API_TestDygraphIndexFill):
    def init_data(self):
        self.data_np = np.random.random((3, 4, 5, 6)).astype(np.float64)
        self.axis = 2
        self.fill_value = 14.0
        self.index = paddle.to_tensor([0, 1, 2, 3])


#todo bool, fill_value_tensor
class API_TestDygraphIndexFill3(API_TestDygraphIndexFill):
    def init_data(self):
        self.data_np = np.random.random((3, 4, 5, 6)) > 0.5
        self.axis = 2
        self.fill_value = True
        self.index = paddle.to_tensor([0, 3])


class API_IndexFill2(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

        self.axis = 1
        self.x_np = np.random.uniform(-1, 1, (2, 3, 4)).astype(np.float64)
        self.index_np = [0, 2]
        self.x_grad = index_fill_x_grad_np(self.x_np, self.axis, self.index_np)

    def test_errors(self):
        def error_axis_dtype():
            with paddle.fluid.dygraph.guard():
                x = paddle.rand((2, 3))
                index = paddle.to_tensor([0])
                paddle.index_fill(x, index, axis=0.5, fill_value=0.1)

        def error_axis_range():
            with paddle.fluid.dygraph.guard():
                x = paddle.rand((2, 3))
                index = paddle.to_tensor([0])
                paddle.index_fill(x, index, axis=10, fill_value=0.1)

        def error_index_dtype2():
            with paddle.fluid.dygraph.guard():
                x = paddle.rand((2, 3))
                index = paddle.to_tensor([0.2, 0.3])
                paddle.index_fill(x, index, axis=1, fill_value=0.1)

        def error_index_dtype3():
            with paddle.fluid.dygraph.guard():
                x = paddle.rand((2, 3))
                index = paddle.to_tensor([])
                paddle.index_fill(x, index, axis=1, fill_value=0.1)

        def error_index_value():
            with paddle.fluid.dygraph.guard():
                x = paddle.rand((2, 3))
                index = paddle.to_tensor([[1], [2]])
                fill_value = paddle.to_tensor([[9.5], [9.8]])
                paddle.index_fill(x, index, axis=1, fill_value=fill_value)

        def error_fill_value():
            with paddle.fluid.dygraph.guard():
                x = paddle.rand((2, 3))
                index = paddle.to_tensor([1])
                fill_value = paddle.to_tensor([[9.5], [9.8]])
                paddle.index_fill(x, index, axis=1, fill_value=fill_value)

        self.assertRaises(TypeError, error_axis_dtype)
        self.assertRaises(ValueError, error_axis_range)
        self.assertRaises(TypeError, error_index_dtype2)
        self.assertRaises(TypeError, error_index_dtype3)
        self.assertRaises(ValueError, error_index_value)
        self.assertRaises(ValueError, error_fill_value)

    def test_check_grad(self):
        paddle.disable_static(place=self.place)
        x_tensor = paddle.to_tensor(self.x_np, stop_gradient=False)
        idx_tensor = paddle.to_tensor(self.index_np)
        y = paddle.index_fill(
            x_tensor, index=idx_tensor, axis=self.axis, fill_value=0.5)
        dx = paddle.grad(y, x_tensor)[0].numpy()
        self.assertTrue(np.allclose(self.x_grad, dx, equal_nan=True))

    def test_check_grad2(self):
        paddle.disable_static(place=self.place)
        x_tensor = paddle.to_tensor(self.x_np, stop_gradient=False)
        fill_tensor = paddle.to_tensor(
            0.9999, stop_gradient=False, dtype=x_tensor.dtype)
        y = paddle.index_fill(
            x_tensor,
            index=self.index_np,
            axis=self.axis,
            fill_value=fill_tensor)
        dx, df = paddle.grad(y, [x_tensor, fill_tensor])
        self.assertTrue(np.allclose(self.x_grad, dx.numpy(), equal_nan=True))
        fill_value_grad = index_fill_fill_value_grad_np(self.x_np, self.axis,
                                                        self.index_np)
        self.assertTrue(
            np.allclose(
                fill_value_grad, df.numpy(), equal_nan=True))


if __name__ == "__main__":
    unittest.main()
