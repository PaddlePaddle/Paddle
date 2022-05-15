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
# import paddle.fluid as fluid
import paddle.fluid.core as core
from typing import List

# np.random.seed(102)


def index_add_np(data: np.array, axis: int, index: List[int], add_value: float):
    # as the calculation will modify the elements of the input data inplace,
    # so copying the input data is important
    data = data.copy()
    # if isinstance(index, np.ndarray):
    #     index = list(index.flatten())
    # added_value = float(added_value)
    if axis == 0:
        data[index] += add_value
        return data
    outer_loop = int(np.prod(data.shape[:axis]))
    x_reshape = [outer_loop] + list(data.shape[axis:])
    x_np_reshape = data.reshape(x_reshape)
    for i in range(outer_loop):
        for j in index:
            x_np_reshape[i, j] += add_value
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
        self.inputs = {'X': self.x_np}
        self.attrs = {'axis': self.axis, 
                      'index': self.index,
                      'add_value': self.add_value}
        self.outputs = {'Out': self.out_np}

    def init_data(self):
        self.axis = 1
        self.x_type = np.float64
        # self.index_type = np.int64
        self.x_shape = (4, 5, 6)
        # self.index_size = 4
        self.add_value = 9.0
        self.x_np = np.random.random(self.x_shape).astype(self.x_type)
        # self.index_np = np.random.randint(
        #     low=0, high=self.x_shape[self.axis], size=self.index_size)
        # self.index_np = np.array([0, 3], dtype=self.index_type)
        self.index = [0, 4, 1]
        self.out_np = index_add_np(self.x_np, self.axis, self.index,
                                    self.add_value)

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', check_eager=True)


# class TestStaticIndexAddAPI(unittest.TestCase):
#     def setUp(self):
#         # self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
#         #     else paddle.CPUPlace()
#         self.places = [paddle.CPUPlace()]
#         if paddle.device.is_compiled_with_cuda():
#             self.places += [paddle.CUDAPlace(0)]
#         self.executed_api()
#         self.init_data()

#     def init_data(self):
#         self.data_np = np.random.random((3, 4, 5)).astype(np.float32)
#         self.axis = 0
#         self.add_value = 7.0
#         self.index = [1, 2]

#     def executed_api(self):
#         self.op_run = paddle.index_add

#     def test_out(self):
#         paddle.enable_static()
#         with paddle.static.program_guard(paddle.static.Program()):
#             x = paddle.static.data('X', self.data_np.shape)
#             # idx = paddle.static.data(
#             #     'index', np.asarray(self.index).shape, dtype="int64")
#             result_pd = self.op_run(
#                 x, axis=self.axis, index=self.index, add_value=self.add_value)
#             result_np = index_add_np(self.data_np, self.axis, self.index,
#                         self.add_value)
#             for place in self.places:
#                 exe = paddle.static.Executor(place)
#                 result, = exe.run(feed={"X": self.data_np},
#                                 fetch_list=[result_pd])
#                 self.assertTrue(np.allclose(result_np, result))


# class TestStaticIndexAddInplaceAPI(TestStaticIndexAddAPI):
#     def executed_api(self):
#         self.op_run = paddle.index_add_

#     def init_data(self):
#         self.data_np = np.random.random((2, 3, 4, 5)).astype(np.float32)
#         self.axis = 3
#         self.add_value = 8.0
#         self.index = [0, 2, 3]
#         # self.place = paddle.CPUPlace()


# class TestDygraphIndexAddAPI(unittest.TestCase):
#     def setUp(self):
#         self.executed_api()
#         self.init_data()
#         # self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
#         #     else paddle.CPUPlace()
#         self.places = [paddle.CPUPlace()]
#         if paddle.device.is_compiled_with_cuda():
#             self.places += [paddle.CUDAPlace(0)]

#     def init_data(self):
#         self.data_np = np.random.random((3, 4, 5)).astype(np.float32)
#         self.axis = 0
#         self.add_value = -11.0
#         self.index = [1, 2]

#     def executed_api(self):
#         self.op_run = paddle.index_add

#     def test_out(self):
#         # index is 1-D tensor
#         for place in self.places:
#             paddle.disable_static(place)
#             input = paddle.to_tensor(self.data_np)
#             index_tensor = paddle.to_tensor(self.index)
#             output = self.op_run(
#                 input, axis=self.axis, index=index_tensor, add_value=self.add_value)
#             out_np = output.numpy()
#             expected_out = index_add_np(self.data_np, self.axis, self.index,
#                                         self.add_value)
#             self.assertTrue(np.allclose(expected_out, out_np))

#         # index is a list of ints
#         for place in self.places:
#             paddle.disable_static(place)
#             input = paddle.to_tensor(self.data_np)
#             output = self.op_run(
#                 input, axis=self.axis, index=self.index, add_value=self.add_value)
#             out_np = output.numpy()
#             expected_out = index_add_np(self.data_np, self.axis, self.index,
#                                         self.add_value)
#             self.assertTrue(np.allclose(expected_out, out_np))


# class TestDygraphIndexAddInplaceAPI(TestDygraphIndexAddAPI):
#     def executed_api(self):
#         self.op_run = paddle.index_add_

#     def init_data(self):
#         self.data_np = np.random.random((3, 4, 5)).astype(np.float32)
#         self.axis = 2
#         self.add_value = 12.0
#         self.index = [0, 2]


# class TestDygraphIndexAddAPI_2(TestDygraphIndexAddAPI):
#     def init_data(self):
#         self.data_np = np.random.random((10)).astype(np.float32)
#         self.axis = 0
#         self.add_value = -13.0
#         self.index = [0, 2, 9]


# class TestDygraphIndexAddAPI_3(TestDygraphIndexAddAPI):
#     def init_data(self):
#         self.data_np = np.random.random((3, 4, 5, 6)).astype(np.float64)
#         self.axis = 2
#         self.add_value = paddle.to_tensor([-14.0])
#         self.index = [0, 1, 2, 3]


# class TestIndexAddError(unittest.TestCase):
#     def setUp(self):
#         self.places = [paddle.CPUPlace()]
#         if paddle.device.is_compiled_with_cuda():
#             self.places += [paddle.CUDAPlace(0)]

#     def test_errors(self):
#         paddle.disable_static()
#         def error_axis_dtype():
#             # with paddle.fluid.dygraph.guard():
#             x = paddle.rand((2, 3))
#             index = paddle.to_tensor([0])
#             paddle.index_add(x, axis=0.5, index=index, add_value=0.1)

#         def error_axis_range():
#             # with paddle.fluid.dygraph.guard():
#             x = paddle.rand((2, 3))
#             index = paddle.to_tensor([0, 1])
#             paddle.index_add(x, axis=10, index=index, add_value=0.1)

#         def error_index_dtype():
#             # with paddle.fluid.dygraph.guard():
#             x = paddle.rand((2, 3))
#             index = [[0, 1]]
#             paddle.index_add(x, axis=0, index=index, add_value=0.1)

#         def error_index_dtype2():
#             # with paddle.fluid.dygraph.guard():
#             x = paddle.rand((2, 3))
#             index = [0.2, 0.3]
#             paddle.index_add(x, axis=1, index=index, add_value=0.1)

#         def error_index_dtype3():
#             # with paddle.fluid.dygraph.guard():
#             x = paddle.rand((2, 3))
#             index = paddle.to_tensor([])
#             paddle.index_add(x, axis=1, index=index, add_value=0.1)

#         def error_index_value():
#             # with paddle.fluid.dygraph.guard():
#             x = paddle.rand((2, 3, 4))
#             index = paddle.to_tensor([[1], [2]])
#             add_value = paddle.to_tensor(-23)
#             paddle.index_add(x, axis=1, index=index, add_value=add_value)

#         def error_add_value():
#             # with paddle.fluid.dygraph.guard():
#             x = paddle.rand((2, 5, 7))
#             index = paddle.to_tensor([1])
#             add_value = paddle.to_tensor([[9.5], [9.8]])
#             paddle.index_add(x, axis=1, index=index, add_value=add_value)

#         self.assertRaises(TypeError, error_axis_dtype)
#         self.assertRaises(ValueError, error_axis_range)
#         self.assertRaises(TypeError, error_index_dtype)
#         self.assertRaises(TypeError, error_index_dtype2)
#         self.assertRaises(TypeError, error_index_dtype3)
#         self.assertRaises(TypeError, error_index_value)
#         self.assertRaises(TypeError, error_add_value)

#     # def test_check_grad(self):
#     #     paddle.disable_static(place=self.place)
#     #     axis = 1
#     #     x_np = np.random.uniform(-1, 1, (2, 3, 4)).astype(np.float64)
#     #     index_np = [0, 2]
#     #     x_tensor = paddle.to_tensor(x_np, stop_gradient=False)
#     #     idx_tensor = paddle.to_tensor(index_np)

#     #     y = paddle.index_add(x_tensor, idx_tensor, axis=axis, added_value=0.5)
#     #     dx = paddle.grad(y, x_tensor)[0].numpy()
#     #     np_grad = index_fill_grad_np(x_np, axis, index_np)
#     #     self.assertTrue(np.allclose(np_grad, dx, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
