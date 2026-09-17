#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import unittest

import numpy as np
from op_test import (
    OpTest,
    convert_float_to_uint16,
    get_device_place,
    get_places,
    is_custom_device,
)
from utils import dygraph_guard, static_guard

import paddle
from paddle.framework import core
from paddle.static import InputSpec

paddle.enable_static()


def put_along_axis_net(arr, axis=-1):
    indices = paddle.to_tensor([[[[2]]]], dtype='int32', stop_gradient=False)
    return paddle.put_along_axis(
        arr, indices=indices, values=-4.0, axis=axis, reduce='add'
    )


class TestPutAlongAxisOp(OpTest):
    def setUp(self):
        self.init_data()
        self.reduce_op = "assign"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.xnp_result = copy.deepcopy(self.xnp)
        np.put_along_axis(self.xnp_result, self.index, self.value, self.axis)
        self.target = self.xnp_result
        broadcast_shape_list = list(self.x_shape)
        broadcast_shape_list[self.axis] = 1
        self.broadcast_shape = tuple(broadcast_shape_list)
        self.index_broadcast = np.broadcast_to(self.index, self.broadcast_shape)
        self.value_broadcast = np.broadcast_to(self.value, self.broadcast_shape)
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index_broadcast,
            'Value': self.value_broadcast,
        }
        self.attrs = {'Axis': self.axis, 'Reduce': self.reduce_op}
        self.outputs = {'Result': self.target}

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(
            ["Input", "Value"], "Result", check_pir=True, check_prim_pir=True
        )

    def init_data(self):
        self.dtype = 'float64'
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.value_type = "float64"
        self.value = np.array([99]).astype(self.value_type)
        self.index_type = "int32"
        self.index = np.array([[[0]]]).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisInt16OpBase(TestPutAlongAxisOp):
    no_need_check_grad = True

    def init_data(self):
        self.set_type()
        self.x_shape = (10, 10, 10)
        self.index_type = "int64"
        self.axis = 1
        self.axis_type = "int64"
        self.set_reduce_op()
        self.set_value_and_index()

    def set_type(self):
        self.dtype = np.int16
        self.x_type = "int16"
        self.value_type = "int16"

    def set_value_and_index(self):
        self.value = np.array([99]).astype(self.value_type)
        self.index = np.array([[[0]]]).astype(self.index_type)

    def set_reduce_op(self):
        self.reduce_op = "assign"

    def test_check_grad(self):
        """int16 can not pass check_grad data type check for op multiply"""
        pass


class TestPutAlongAxisUInt8OpBase(TestPutAlongAxisInt16OpBase):
    no_need_check_grad = True

    def set_type(self):
        self.dtype = np.uint8
        self.x_type = "uint8"
        self.value_type = "uint8"

    def set_reduce_op(self):
        self.reduce_op = "assign"
        self.value = np.array([127]).astype(self.value_type)
        self.index = np.array([[[0]]]).astype(self.index_type)

    def test_check_grad(self):
        """uint8 can not pass check_grad data type check for op multiply"""
        pass


class TestPutAlongAxisInt16OpAdd(TestPutAlongAxisInt16OpBase):
    def set_reduce_op(self):
        self.reduce_op = "add"


class TestPutAlongAxisInt16OpMul(TestPutAlongAxisInt16OpBase):
    def set_reduce_op(self):
        self.reduce_op = "mul"


class TestPutAlongAxisInt16OpAMin(TestPutAlongAxisInt16OpBase):
    def set_reduce_op(self):
        self.reduce_op = "amin"


class TestPutAlongAxisInt16OpAMax(TestPutAlongAxisInt16OpBase):
    def set_reduce_op(self):
        self.reduce_op = "amax"


class TestPutAlongAxisUInt8OpAdd(TestPutAlongAxisUInt8OpBase):
    def set_reduce_op(self):
        self.reduce_op = "add"


class TestPutAlongAxisUInt8OpMul(TestPutAlongAxisUInt8OpBase):
    def set_reduce_op(self):
        self.reduce_op = "mul"


class TestPutAlongAxisUInt8OpAMin(TestPutAlongAxisUInt8OpBase):
    def set_reduce_op(self):
        self.reduce_op = "amin"


class TestPutAlongAxisUInt8OpAMax(TestPutAlongAxisUInt8OpBase):
    def set_reduce_op(self):
        self.reduce_op = "amax"


class TestPutAlongAxisFP16Op(TestPutAlongAxisOp):
    def init_data(self):
        self.dtype = np.float16
        self.x_type = "float16"
        self.x_shape = (10, 10, 10)
        self.value_type = "float16"
        self.value = np.array([99]).astype(self.value_type)
        self.index_type = "int64"
        self.index = np.array([[[0]]]).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisOpCase2(TestPutAlongAxisOp):
    def setUp(self):
        self.init_data()
        self.reduce_op = "assign"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.target[i, self.index[i, j, k], k] = self.value[i, j, k]
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
            'Value': self.value,
        }
        self.attrs = {
            'Axis': self.axis,
            'Reduce': self.reduce_op,
            'include_self': True,
            'broadcast': False,
        }
        self.outputs = {'Result': self.target}

    def init_data(self):
        self.dtype = 'float32'
        self.x_type = "float32"
        self.x_shape = (10, 10, 10)
        self.value_type = "float32"
        self.value = (
            np.arange(1, 126).reshape((5, 5, 5)).astype(self.value_type)
        )
        self.index_type = "int64"
        self.index = np.zeros((5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisOpMul(TestPutAlongAxisOp):
    def setUp(self):
        self.init_data()
        self.reduce_op = "mul"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.target[i, self.index[i, j, k], k] *= self.value[
                        i, j, k
                    ]
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
            'Value': self.value,
        }
        self.attrs = {
            'Axis': self.axis,
            'Reduce': self.reduce_op,
            'Include_self': True,
            'broadcast': False,
        }
        self.outputs = {'Result': self.target}

    def init_data(self):
        self.dtype = 'float64'
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.value_type = "float64"
        self.value = np.random.randint(1, 5, (5, 5, 5)).astype(self.value_type)
        self.index_type = "int64"
        self.index = np.zeros((5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisOpMulNotIncludeSelf(TestPutAlongAxisOp):
    def setUp(self):
        self.init_data()
        self.reduce_op = "mul"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        self.nums = np.zeros_like(self.target)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if self.nums[i, self.index[i, j, k], k] == 0:
                        self.target[i, self.index[i, j, k], k] = self.value[
                            i, j, k
                        ]
                    else:
                        self.target[i, self.index[i, j, k], k] *= self.value[
                            i, j, k
                        ]
                    self.nums[i, self.index[i, j, k], k] += 1
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
            'Value': self.value,
        }
        self.attrs = {
            'Axis': self.axis,
            'Reduce': self.reduce_op,
            'Include_self': False,
            'broadcast': False,
        }
        self.outputs = {'Result': self.target}

    def init_data(self):
        self.dtype = 'float64'
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.value_type = "float64"
        self.value = np.random.randint(1, 5, (5, 5, 5)).astype(self.value_type)
        self.index_type = "int64"
        self.index = np.zeros((5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisOpAdd(TestPutAlongAxisOp):
    def setUp(self):
        self.init_data()
        self.reduce_op = "add"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.target[i, self.index[i, j, k], k] += self.value[
                        i, j, k
                    ]
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
            'Value': self.value,
        }
        self.attrs = {
            'Axis': self.axis,
            'Reduce': self.reduce_op,
            'Include_self': True,
            'broadcast': False,
        }
        self.outputs = {'Result': self.target}

    def init_data(self):
        self.dtype = 'float64'
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.value_type = "float64"
        self.value = np.random.randint(1, 100, (5, 5, 5)).astype(
            self.value_type
        )
        self.index_type = "int64"
        self.index = np.zeros((5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisOpAddNotIncludeSelf(TestPutAlongAxisOp):
    def setUp(self):
        self.init_data()
        self.reduce_op = "add"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        self.nums = np.zeros_like(self.target)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if self.nums[i, self.index[i, j, k], k] == 0:
                        self.target[i, self.index[i, j, k], k] = self.value[
                            i, j, k
                        ]
                    else:
                        self.target[i, self.index[i, j, k], k] += self.value[
                            i, j, k
                        ]
                    self.nums[i, self.index[i, j, k], k] += 1
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
            'Value': self.value,
        }
        self.attrs = {
            'Axis': self.axis,
            'Reduce': self.reduce_op,
            'Include_self': False,
            'broadcast': False,
        }
        self.outputs = {'Result': self.target}

    def init_data(self):
        self.dtype = 'float64'
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.value_type = "float64"
        self.value = np.random.randint(1, 5, (5, 5, 5)).astype(self.value_type)
        self.index_type = "int64"
        self.index = np.zeros((5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisOpMean(TestPutAlongAxisOp):
    def setUp(self):
        self.init_data()
        self.reduce_op = "mean"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        self.nums = np.ones_like(self.target)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.target[i, self.index[i, j, k], k] += self.value[
                        i, j, k
                    ]
                    self.nums[i, self.index[i, j, k], k] += 1
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    self.target[i, j, k] /= self.nums[i, j, k]
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
            'Value': self.value,
        }
        self.attrs = {
            'Axis': self.axis,
            'Reduce': self.reduce_op,
            'Include_self': True,
            'broadcast': False,
        }
        self.outputs = {'Result': self.target}

    def init_data(self):
        self.dtype = 'float64'
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.value_type = "float64"
        self.value = np.random.randint(1, 5, (5, 5, 5)).astype(self.value_type)
        self.index_type = "int64"
        self.index = np.zeros((5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisOpMeanNotIncludeSelf(TestPutAlongAxisOp):
    def setUp(self):
        self.init_data()
        self.reduce_op = "mean"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        self.nums = np.zeros_like(self.target)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if self.nums[i, self.index[i, j, k], k] == 0:
                        self.target[i, self.index[i, j, k], k] = self.value[
                            i, j, k
                        ]
                    else:
                        self.target[i, self.index[i, j, k], k] += self.value[
                            i, j, k
                        ]
                    self.nums[i, self.index[i, j, k], k] += 1
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    if self.nums[i, j, k] > 0:
                        self.target[i, j, k] = (
                            self.target[i, j, k] / self.nums[i, j, k]
                        )
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
            'Value': self.value,
        }
        self.attrs = {
            'Axis': self.axis,
            'Reduce': self.reduce_op,
            'Include_self': False,
            'broadcast': False,
        }
        self.outputs = {'Result': self.target}

    def init_data(self):
        self.dtype = 'float64'
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.value_type = "float64"
        self.value = np.random.randint(1, 5, (5, 5, 5)).astype(self.value_type)
        self.index_type = "int64"
        self.index = np.zeros((5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisOpMin(TestPutAlongAxisOp):
    def setUp(self):
        self.init_data()
        self.reduce_op = "amin"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.target[i, self.index[i, j, k], k] = min(
                        self.target[i, self.index[i, j, k], k],
                        self.value[i, j, k],
                    )
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
            'Value': self.value,
        }
        self.attrs = {
            'Axis': self.axis,
            'Reduce': self.reduce_op,
            'include_self': True,
            'broadcast': False,
        }
        self.outputs = {'Result': self.target}

    def init_data(self):
        self.dtype = 'float64'
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.value_type = "float64"
        self.value = (
            np.arange(1, 126).reshape((5, 5, 5)).astype(self.value_type)
        )
        self.index_type = "int64"
        self.index = np.zeros((5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisOpMinNotIncludeSelf(TestPutAlongAxisOp):
    def setUp(self):
        self.init_data()
        self.reduce_op = "amin"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.target[i, self.index[i, j, k], k] = self.value[i, j, k]
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.target[i, self.index[i, j, k], k] = min(
                        self.target[i, self.index[i, j, k], k],
                        self.value[i, j, k],
                    )
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
            'Value': self.value,
        }
        self.attrs = {
            'Axis': self.axis,
            'Reduce': self.reduce_op,
            'Include_self': False,
            'broadcast': False,
        }
        self.outputs = {'Result': self.target}

    def init_data(self):
        self.dtype = 'float64'
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.value_type = "float64"
        self.value = (
            np.arange(1, 126).reshape((5, 5, 5)).astype(self.value_type)
        )
        self.index_type = "int64"
        self.index = np.zeros((5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisOpMax(TestPutAlongAxisOp):
    def setUp(self):
        self.init_data()
        self.reduce_op = "amax"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.target[i, self.index[i, j, k], k] = max(
                        self.target[i, self.index[i, j, k], k],
                        self.value[i, j, k],
                    )
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
            'Value': self.value,
        }
        self.attrs = {
            'Axis': self.axis,
            'Reduce': self.reduce_op,
            'include_self': True,
            'broadcast': False,
        }
        self.outputs = {'Result': self.target}

    def init_data(self):
        self.dtype = 'float64'
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.value_type = "float64"
        self.value = (
            np.arange(1, 126).reshape((5, 5, 5)).astype(self.value_type)
        )
        self.index_type = "int64"
        self.index = np.zeros((5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisOpMaxNotIncludeSelf(TestPutAlongAxisOp):
    def setUp(self):
        self.init_data()
        self.reduce_op = "amax"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.target[i, self.index[i, j, k], k] = self.value[i, j, k]
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.target[i, self.index[i, j, k], k] = max(
                        self.target[i, self.index[i, j, k], k],
                        self.value[i, j, k],
                    )
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
            'Value': self.value,
        }
        self.attrs = {
            'Axis': self.axis,
            'Reduce': self.reduce_op,
            'Include_self': False,
            'broadcast': False,
        }
        self.outputs = {'Result': self.target}

    def init_data(self):
        self.dtype = 'float64'
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.value_type = "float64"
        self.value = (
            np.arange(1, 126).reshape((5, 5, 5)).astype(self.value_type)
        )
        self.index_type = "int64"
        self.index = np.zeros((5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestPutAlongAxisBF16Op(OpTest):
    def setUp(self):
        self.init_data()
        self.reduce_op = "assign"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.xnp_result = copy.deepcopy(self.xnp)
        np.put_along_axis(self.xnp_result, self.index, self.value, self.axis)
        self.target = self.xnp_result
        broadcast_shape_list = list(self.x_shape)
        broadcast_shape_list[self.axis] = 1
        self.broadcast_shape = tuple(broadcast_shape_list)
        self.index_broadcast = np.broadcast_to(self.index, self.broadcast_shape)
        self.value_broadcast = np.broadcast_to(self.value, self.broadcast_shape)
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index_broadcast,
            'Value': self.value_broadcast,
        }
        self.attrs = {'Axis': self.axis, 'Reduce': self.reduce_op}
        self.outputs = {'Result': self.target}

        self.inputs['Input'] = convert_float_to_uint16(self.inputs['Input'])
        self.inputs['Value'] = convert_float_to_uint16(self.inputs['Value'])
        self.outputs['Result'] = convert_float_to_uint16(self.outputs['Result'])
        self.place = get_device_place()

    def test_check_output(self):
        self.check_output_with_place(
            self.place, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["Input", "Value"],
            "Result",
            check_pir=True,
            check_prim_pir=True,
        )

    def init_data(self):
        self.dtype = np.uint16
        self.x_type = "float32"
        self.x_shape = (10, 10, 10)
        self.value_type = "float32"
        self.value = np.array([99]).astype(self.value_type)
        self.index_type = "int32"
        self.index = np.array([[[0]]]).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [1, 3]
        self.index_shape = [1, 1]
        self.index_np = np.array([[0]]).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = get_places()
        self.axis = 0
        self.value_np = 99.0
        self.value_shape = []
        self.x_feed = copy.deepcopy(self.x_np)

    def test_api_static(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.shape)
                index = paddle.static.data('Index', self.index_shape, "int64")
                value = paddle.static.data('Value', self.value_shape)
                out = paddle.put_along_axis(x, index, value, self.axis)
                exe = paddle.static.Executor(self.place[0])
                res = exe.run(
                    feed={
                        'X': self.x_feed,
                        'Value': self.value_np,
                        'Index': self.index_np,
                    },
                    fetch_list=[out],
                )

            np.put_along_axis(
                self.x_np, self.index_np, self.value_np, self.axis
            )
            # numpy put_along_axis is an inplace operation.
            out_ref = self.x_np

            for out in res:
                np.testing.assert_allclose(out, out_ref, rtol=0.001)

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            index_tensor = paddle.to_tensor(self.index_np)
            value_tensor = paddle.to_tensor(self.value_np)
            out = paddle.put_along_axis(
                x_tensor, index_tensor, value_tensor, self.axis
            )
            np.array(
                np.put_along_axis(
                    self.x_np, self.index_np, self.value_np, self.axis
                )
            )
            out_ref = self.x_np
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

            # for ci coverage, numpy put_along_axis did not support argument of 'reduce'
            paddle.put_along_axis(
                x_tensor, index_tensor, value_tensor, self.axis, 'mul'
            )
            paddle.put_along_axis(
                x_tensor, index_tensor, value_tensor, self.axis, 'add'
            )

            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_inplace_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            index_tensor = paddle.to_tensor(self.index_np)
            value_tensor = paddle.to_tensor(self.value_np)

            x_tensor.put_along_axis_(index_tensor, value_tensor, self.axis)

            np.array(
                np.put_along_axis(
                    self.x_np, self.index_np, self.value_np, self.axis
                )
            )
            out_ref = self.x_np

            np.testing.assert_allclose(x_tensor.numpy(), out_ref, rtol=0.001)
            paddle.enable_static()

        for place in self.place:
            run(place)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestPutAlongAxisAPILargeCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [64, 1327104]
        self.index_shape = [64, 1327104]
        self.index_np = np.zeros(self.index_shape).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.axis = 1
        self.value_np = np.ones(self.index_shape).astype(np.float32)
        self.x_feed = copy.deepcopy(self.x_np)
        self.place = [get_device_place()]

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            index_tensor = paddle.to_tensor(self.index_np)
            value_tensor = paddle.to_tensor(self.value_np)
            out = paddle.put_along_axis(
                x_tensor, index_tensor, value_tensor, self.axis
            )
            np.array(
                np.put_along_axis(
                    self.x_np, self.index_np, self.value_np, self.axis
                )
            )
            out_ref = self.x_np
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

            paddle.enable_static()

        for place in self.place:
            run(place)


class TestPutAlongAxisAPICase2(TestPutAlongAxisAPI):
    def setUp(self):
        np.random.seed(0)
        self.shape = [2, 2]
        self.index_shape = [2, 2]
        self.index_np = np.array([[0, 0], [1, 0]]).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = get_places()
        self.axis = 0
        self.value_np = 99.0
        self.value_shape = []
        self.x_feed = copy.deepcopy(self.x_np)


class TestPutAlongAxisAPICase3(TestPutAlongAxisAPI):
    def setUp(self):
        np.random.seed(0)
        self.shape = [2, 2]
        self.index_shape = [4, 2]
        self.index_np = np.array([[0, 0], [1, 0], [0, 0], [1, 0]]).astype(
            'int64'
        )
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = get_places()
        self.axis = 0
        self.value_np = 99.0
        self.value_shape = []
        self.x_feed = copy.deepcopy(self.x_np)

    def test_inplace_dygraph(self):
        pass


class TestPutAlongAxisAPICase4(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [3, 5]
        self.index1_shape = [1, 4]
        self.index_np1 = np.array([[0, 1, 2, 0]]).astype('int64')
        self.index2_shape = [2, 3]
        self.index_np2 = np.array([[0, 1, 2], [0, 1, 4]]).astype('int64')
        self.x_np = np.zeros((3, 5)).astype(np.float32)
        self.value_shape = [2, 5]
        self.value = (
            np.arange(1, 11).reshape(self.value_shape).astype(np.float32)
        )
        self.place = get_places()

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            index_tensor1 = paddle.to_tensor(self.index_np1)
            value_tensor = paddle.to_tensor(self.value)
            out = paddle.put_along_axis(
                x_tensor, index_tensor1, value_tensor, 0, 'assign', True, False
            )
            out_ref = copy.deepcopy(self.x_np)
            for i in range(self.index1_shape[0]):
                for j in range(self.index1_shape[1]):
                    out_ref[self.index_np1[i, j], j] = self.value[i, j]
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

            # for ci coverage, numpy put_along_axis did not support argument of 'reduce'
            paddle.put_along_axis(
                x_tensor, index_tensor1, value_tensor, 0, 'mul', True, False
            )
            paddle.put_along_axis(
                x_tensor, index_tensor1, value_tensor, 0, 'add', True, False
            )

            index_tensor2 = paddle.to_tensor(self.index_np2)
            out = paddle.put_along_axis(
                x_tensor, index_tensor2, value_tensor, 1, 'assign', True, False
            )
            out_ref = copy.deepcopy(self.x_np)
            for i in range(self.index2_shape[0]):
                for j in range(self.index2_shape[1]):
                    out_ref[i, self.index_np2[i, j]] = self.value[i, j]
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

            # for ci coverage, numpy put_along_axis did not support argument of 'reduce'
            paddle.put_along_axis(
                x_tensor, index_tensor2, value_tensor, 1, 'mul', True, False
            )
            paddle.put_along_axis(
                x_tensor, index_tensor2, value_tensor, 1, 'add', True, False
            )

            paddle.enable_static()

        def run_inplace(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            index_tensor1 = paddle.to_tensor(self.index_np1)
            value_tensor = paddle.to_tensor(self.value)
            x_tensor.put_along_axis_(
                index_tensor1, value_tensor, 0, 'assign', True, False
            )
            out_ref = copy.deepcopy(self.x_np)
            for i in range(self.index1_shape[0]):
                for j in range(self.index1_shape[1]):
                    out_ref[self.index_np1[i, j], j] = self.value[i, j]
            np.testing.assert_allclose(x_tensor.numpy(), out_ref, rtol=0.001)

            x_tensor = paddle.to_tensor(self.x_np)
            index_tensor2 = paddle.to_tensor(self.index_np2)
            x_tensor.put_along_axis_(
                index_tensor2, 10, 1, 'assign', True, False
            )
            out_ref = copy.deepcopy(self.x_np)
            for i in range(self.index2_shape[0]):
                for j in range(self.index2_shape[1]):
                    out_ref[i, self.index_np2[i, j]] = 10
            np.testing.assert_allclose(x_tensor.numpy(), out_ref, rtol=0.001)

            paddle.enable_static()

        for place in self.place:
            run(place)
            run_inplace(place)

    def test_api_static(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                x1 = paddle.static.data('X', self.shape)
                index1 = paddle.static.data('Index', self.index1_shape, "int64")
                value_tensor = paddle.to_tensor(self.value)
                out1 = paddle.put_along_axis(
                    x1, index1, value_tensor, 0, 'assign', True, False
                )
                exe = paddle.static.Executor(place)
                res = exe.run(
                    feed={
                        'X': self.x_np,
                        'Value': self.value,
                        'Index': self.index_np1,
                    },
                    fetch_list=[out1],
                )
            out_ref = copy.deepcopy(self.x_np)
            for i in range(self.index1_shape[0]):
                for j in range(self.index1_shape[1]):
                    out_ref[self.index_np1[i, j], j] = self.value[i, j]

            for out in res:
                np.testing.assert_allclose(out, out_ref, rtol=0.001)

            with paddle.static.program_guard(paddle.static.Program()):
                x2 = paddle.static.data('X', self.shape)
                index2 = paddle.static.data('Index', self.index2_shape, "int64")
                value_tensor = paddle.to_tensor(self.value)
                out2 = paddle.put_along_axis(
                    x2, index2, value_tensor, 1, 'assign', True, False
                )
                exe = paddle.static.Executor(place)
                res = exe.run(
                    feed={
                        'X': self.x_np,
                        'Value': self.value,
                        'Index': self.index_np2,
                    },
                    fetch_list=[out2],
                )
            out_ref = copy.deepcopy(self.x_np)
            for i in range(self.index2_shape[0]):
                for j in range(self.index2_shape[1]):
                    out_ref[i, self.index_np2[i, j]] = self.value[i, j]

            for out in res:
                np.testing.assert_allclose(out, out_ref, rtol=0.001)

        for place in self.place:
            run(place)

    def _check_error(self, index_too_large_error):
        """The body of ``test_error``, run once per mode.

        ``index_too_large_error`` is what tells the two modes apart: in dygraph
        the wrapper rejects the index itself and raises ``RuntimeError``, while
        in static and PIR that check is skipped -- it is guarded by
        ``in_dynamic_mode()`` -- and ``PutAlongAxisInferMeta`` reports the
        violation as ``InvalidArgument``, which surfaces in python as
        ``ValueError``.
        """
        tensorx = paddle.to_tensor([[1, 2, 3], [4, 5, 6]]).astype("float32")
        indices = paddle.to_tensor([1]).astype("int32")
        values = paddle.to_tensor([2])
        # len(arr.shape) != len(indices.shape)
        with self.assertRaises(ValueError):
            paddle.put_along_axis(
                tensorx, indices, 1.0, 0, 'assign', True, False
            )
        indices = paddle.to_tensor([[1]]).astype("int32")
        # len(values.shape) != len(indices.shape)
        with self.assertRaises(ValueError):
            paddle.put_along_axis(
                tensorx, indices, values, 0, 'assign', True, False
            )
        # len(values.shape) != len(indices.shape)
        with self.assertRaises(ValueError):
            tensorx.put_along_axis_(indices, values, 0, 'assign', True, False)
        indices = paddle.to_tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        ).astype("int32")
        # indices too large
        with self.assertRaises(index_too_large_error):
            paddle.put_along_axis(
                tensorx, indices, 1.0, 0, 'assign', True, False
            )
        # indices too large
        with self.assertRaises(index_too_large_error):
            tensorx.put_along_axis_(indices, 1.0, 0, 'assign', True, False)
        if not paddle.in_dynamic_mode():
            # What is left is a check on the index *values*, which only the
            # wrapper performs and only in dygraph. Outside it there is nothing
            # to assert: the shapes below are legal, so InferMeta accepts them
            # and an out-of-range element would only be seen by the kernel, at
            # a point this test never reaches.
            return
        indices = paddle.to_tensor([[10]]).astype("int32")
        # the element of indices out of range
        with self.assertRaises(RuntimeError):
            paddle.put_along_axis(
                tensorx, indices, 1.0, 0, 'assign', True, False
            )
        # the element of indices out of range
        with self.assertRaises(RuntimeError):
            tensorx.put_along_axis_(indices, 1.0, 0, 'assign', True, False)

    def test_error(self):
        # Pin the mode rather than inherit whichever one the previously run
        # test left enabled: which layer rejects an illegal input depends on
        # it, so an inherited mode would decide what this test asserts.
        with dygraph_guard():
            self._check_error(RuntimeError)
        with (
            static_guard(),
            paddle.static.program_guard(paddle.static.Program()),
        ):
            self._check_error(ValueError)

    def test_index_type_error(self):
        tensorx = paddle.to_tensor([[1, 2, 3], [4, 5, 6]]).astype("float32")
        indices = paddle.to_tensor([[1]]).astype("float32")
        values = paddle.to_tensor([[2]])
        with self.assertRaises(TypeError):
            res = paddle.put_along_axis(
                tensorx, indices, values, 0, 'mul', True, False
            )


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestPutAlongAxisAPIMulFloat32(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.dtype = 'float32'
        self.x_type = "float32"
        self.x_shape = (10, 10, 10)
        self.value_type = "float32"
        self.value = np.random.randint(1, 5, (5, 5, 5)).astype(self.value_type)
        self.index_type = "int64"
        self.index = np.random.randint(0, 5, (5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.target[i, self.index[i, j, k], k] *= self.value[
                        i, j, k
                    ]

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.xnp)
            index_tensor = paddle.to_tensor(self.index)
            value_tensor = paddle.to_tensor(self.value)
            out = paddle.put_along_axis(
                x_tensor,
                index_tensor,
                value_tensor,
                self.axis,
                "mul",
                True,
                False,
            )
            out_ref = self.target
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

        run(get_device_place())


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestPutAlongAxisAPIMulBF16(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.dtype = 'float32'
        self.x_type = "float32"
        self.x_shape = (10, 10, 10)
        self.value_type = "float32"
        self.value = np.random.randint(1, 3, (3, 3, 3)).astype(self.value_type)
        self.index_type = "int64"
        self.index = np.random.randint(0, 3, (3, 3, 3)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.target = copy.deepcopy(self.xnp)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.target[i, self.index[i, j, k], k] *= self.value[
                        i, j, k
                    ]
        self.xnp = convert_float_to_uint16(self.xnp)
        self.value = convert_float_to_uint16(self.value)
        self.target = convert_float_to_uint16(self.target)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.xnp)
            index_tensor = paddle.to_tensor(self.index)
            value_tensor = paddle.to_tensor(self.value)
            out = paddle.put_along_axis(
                x_tensor,
                index_tensor,
                value_tensor,
                self.axis,
                "mul",
                True,
                False,
            )
            out_ref = self.target
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

        run(get_device_place())


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestPutAlongAxisAPIMulInt32(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.dtype = 'int32'
        self.x_type = "int32"
        self.x_shape = (10, 10, 10)
        self.value_type = "int32"
        self.value = np.random.randint(1, 5, (5, 5, 5)).astype(self.value_type)
        self.index_type = "int32"
        self.index = np.zeros((5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.randint(1, 5, self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.target[i, self.index[i, j, k], k] *= self.value[
                        i, j, k
                    ]

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.xnp)
            index_tensor = paddle.to_tensor(self.index)
            value_tensor = paddle.to_tensor(self.value)
            out = paddle.put_along_axis(
                x_tensor,
                index_tensor,
                value_tensor,
                self.axis,
                "mul",
                True,
                False,
            )
            out_ref = self.target
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

        run(get_device_place())


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestPutAlongAxisAPIMulInt64(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.dtype = 'int64'
        self.x_type = "int64"
        self.x_shape = (10, 10, 10)
        self.value_type = "int64"
        self.value = np.random.randint(1, 5, (5, 5, 5)).astype(self.value_type)
        self.index_type = "int64"
        self.index = np.zeros((5, 5, 5)).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.randint(1, 5, self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    self.target[i, self.index[i, j, k], k] *= self.value[
                        i, j, k
                    ]

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.xnp)
            index_tensor = paddle.to_tensor(self.index)
            value_tensor = paddle.to_tensor(self.value)
            out = paddle.put_along_axis(
                x_tensor,
                index_tensor,
                value_tensor,
                self.axis,
                "mul",
                True,
                False,
            )
            out_ref = self.target
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

        run(get_device_place())


class TestPutAlongAxisAPIReduceLowBits(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.setup_dtype()
        self.set_range()
        self.set_op_to_test()
        self.x_shape = (8, 8)
        self.value = np.random.randint(*self.ranges, (8, 8)).astype(
            self.value_type
        )
        self.index_type = "int64"
        self.index = np.ones((8, 8), dtype=np.int64)
        self.axis = 1
        self.axis_type = "int64"
        self.op_type = "put_along_axis"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.tensor.put_along_axis
        self.python_api = paddle.tensor.put_along_axis
        self.xnp = np.random.randint(*self.ranges, self.x_shape).astype(
            self.x_type
        )
        self.input_filter()
        # numpy put_along_axis is an inplace operation.
        self.target = copy.deepcopy(self.xnp)
        if self.op == "mul":
            host_op = lambda x, y: x * y
        elif self.op == "amax":
            host_op = lambda x, y: max(x, y)
        elif self.op == "amin":
            host_op = lambda x, y: min(x, y)
        else:
            raise ValueError(
                f"Unsupported reduce op for put along axis: {self.op}"
            )
        for i in range(8):
            for j in range(8):
                self.target[i, self.index[i, j]] = host_op(
                    self.target[i, self.index[i, j]], self.value[i, j]
                )

    def input_filter(self):
        if self.ranges[0] <= 0 and self.op == "mul":
            is_zero = self.values == 0
            self.values[is_zero] = 1
            is_zero = self.xnp == 0
            self.xnp[is_zero] = 1

    def setup_dtype(self):
        self.dtype = 'uint8'
        self.x_type = "uint8"
        self.value_type = "uint8"

    def set_range(self):
        self.ranges = [1, 5]

    def set_op_to_test(self):
        self.op = "mul"

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.xnp)
            index_tensor = paddle.to_tensor(self.index)
            value_tensor = paddle.to_tensor(self.value)
            out = paddle.put_along_axis(
                x_tensor,
                index_tensor,
                value_tensor,
                self.axis,
                self.op,
                True,
                False,
            )
            out_ref = self.target
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

        run(
            get_device_place()
            if (core.is_compiled_with_cuda() or is_custom_device())
            else paddle.CPUPlace()
        )


class TestPutAlongAxisAPIMulInt16(TestPutAlongAxisAPIReduceLowBits):
    def setup_dtype(self):
        self.dtype = 'int16'
        self.x_type = "int16"
        self.value_type = "int16"


class TestPutAlongAxisAPIMinInt16(TestPutAlongAxisAPIMulInt16):
    def set_range(self):
        self.ranges = [-32760, 32761]

    def set_op_to_test(self):
        self.op = "amin"


class TestPutAlongAxisAPIMaxInt16(TestPutAlongAxisAPIMinInt16):
    def set_op_to_test(self):
        self.op = "amax"


class TestPutAlongAxisAPIMinUInt8(TestPutAlongAxisAPIReduceLowBits):
    def set_range(self):
        self.ranges = [0, 256]

    def set_op_to_test(self):
        self.op = "amin"


class TestPutAlongAxisAPIMaxUInt8(TestPutAlongAxisAPIMinUInt8):
    def set_op_to_test(self):
        self.op = "amax"


class TestPutAlongAxisDynamicShape(unittest.TestCase):
    def setUp(self):
        np.random.seed(2024)
        self.net = put_along_axis_net
        self.enable_cinn = False
        self.tol = 1e-6
        self.dtype = "float32"
        self.axis = -2
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=self.dtype,
                stop_gradient=False,
            )
        ]
        self.arr = np.random.random([10, 10, 10, 10]).astype(self.dtype)

    def train(self, to_static):
        arr = paddle.to_tensor(self.arr, stop_gradient=False)
        if to_static:
            backend = "CINN" if self.enable_cinn else None
            net = paddle.jit.to_static(
                self.net,
                input_spec=self.input_specs,
                backend=backend,
                full_graph=True,
            )
            net.train()
        else:
            net = self.net

        res = net(arr, self.axis)
        res.backward()
        arr_grad = arr.gradient()
        return res, arr_grad

    def test_dynamic_static(self):
        with dygraph_guard():
            st_out, st_grads = self.train(to_static=True)
            dy_out, dy_grads = self.train(to_static=False)

            for ref, actual in zip(dy_out, st_out):
                np.testing.assert_allclose(
                    ref, actual, rtol=self.tol, atol=self.tol
                )

            for dr, d in zip(dy_grads, st_grads):
                np.testing.assert_allclose(dr, d, rtol=self.tol, atol=self.tol)


class TestPutAlongAxisDynamicShape1(TestPutAlongAxisDynamicShape):
    def setUp(self):
        np.random.seed(2024)
        self.net = put_along_axis_net
        self.enable_cinn = False
        self.tol = 1e-6
        self.dtype = "float32"
        self.axis = 0
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=self.dtype,
                stop_gradient=False,
            )
        ]
        self.arr = np.random.random([16, 16, 16, 16]).astype(self.dtype)


class TestPutAlongAxisDynamicShape2(TestPutAlongAxisDynamicShape):
    def setUp(self):
        np.random.seed(2024)
        self.net = put_along_axis_net
        self.enable_cinn = False
        self.tol = 1e-6
        self.dtype = "float32"
        self.axis = -1
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=self.dtype,
                stop_gradient=False,
            )
        ]
        self.arr = np.random.random([20, 20, 20, 20]).astype(self.dtype)


class TestPutAlongAxisDynamicShape3(TestPutAlongAxisDynamicShape):
    def setUp(self):
        np.random.seed(2024)
        self.net = put_along_axis_net
        self.enable_cinn = False
        self.tol = 1e-6
        self.dtype = "float32"
        self.axis = 3
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=self.dtype,
                stop_gradient=False,
            )
        ]
        self.arr = np.random.random([32, 32, 32, 32]).astype(self.dtype)


class TestPutAlongAxisDynamicShape_ZeroSize(TestPutAlongAxisDynamicShape):
    """``arr`` is 0-size while ``put_along_axis_net`` scatters a [1, 1, 1, 1]
    index, which exceeds ``arr`` on dimension 0.

    Both paths reject it, from different places. Dygraph knows the shapes and
    reports it in ``check_put_along_axis_index_shape``. The to_static path
    traces ``arr`` with a fully dynamic ``(-1, -1, -1, -1)`` spec, so nothing
    can be decided at compile time; the broadcast target keeps the larger of
    the two sizes apart from ``axis``, which leaves the violation for
    ``PutAlongAxisInferMeta`` to report when the executor runs it again with
    the runtime shapes. Before, ``arr``'s 0 won, the index was collapsed to
    0-size and the scatter became a silent no-op.
    """

    def setUp(self):
        np.random.seed(2024)
        self.net = put_along_axis_net
        self.enable_cinn = False
        self.tol = 1e-6
        self.dtype = "float32"
        self.axis = -2
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=self.dtype,
                stop_gradient=False,
            )
        ]
        self.arr = np.random.random([0, 10, 10, 10]).astype(self.dtype)

    def test_dynamic_static(self):
        with dygraph_guard():
            for to_static in (False, True):
                with self.assertRaisesRegex(
                    (RuntimeError, ValueError), "no larger than self"
                ):
                    self.train(to_static=to_static)


class TestPutAlongAxisDynamicShapeIndexLargerThanInput(
    TestPutAlongAxisDynamicShape
):
    """``indices`` exceeds ``arr`` on a dimension other than ``axis`` while
    ``arr`` is traced with a dynamic shape.

    The broadcast target keeps ``indices``' larger size instead of overriding
    it with ``arr``'s, so the violation stays visible to
    ``PutAlongAxisInferMeta`` and is reported the way torch does. Before, the
    index was silently shrunk to fit ``arr``.
    """

    def setUp(self):
        np.random.seed(2024)

        def net(arr, axis=-1):
            indices = paddle.to_tensor(
                [[[[2]]]] * 5, dtype='int32', stop_gradient=False
            )
            return paddle.put_along_axis(
                arr, indices=indices, values=-4.0, axis=axis, reduce='add'
            )

        self.net = net
        self.enable_cinn = False
        self.tol = 1e-6
        self.dtype = "float32"
        self.axis = -2
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=self.dtype,
                stop_gradient=False,
            )
        ]
        self.arr = np.random.random([3, 10, 10, 10]).astype(self.dtype)

    def test_dynamic_static(self):
        with dygraph_guard():
            for to_static in (False, True):
                with self.assertRaisesRegex(
                    (RuntimeError, ValueError), "no larger than self"
                ):
                    self.train(to_static=to_static)


class TestPutAlongAxisZeroSizeIndex(unittest.TestCase):
    """
    When indices has a 0-size dimension (numel == 0), put_along_axis should
    return a copy of arr unchanged regardless of the shape of values.

    Before the fix, the Python wrapper tried to broadcast_to(values, [2, 0])
    which failed because a non-1/non-0 input dimension cannot be expanded to 0
    (that is a valid constraint in expand). The fix adds an early return when
    indices.numel() == 0, matching PyTorch scatter_ behaviour.
    """

    def setUp(self):
        paddle.disable_static()

    def _check(self, arr_shape, index_shape, val_shape, axis):
        arr = paddle.rand(arr_shape, dtype='float32')
        idx = paddle.zeros(index_shape, dtype='int64')
        val = paddle.rand(val_shape, dtype='float32')
        out = paddle.put_along_axis(arr, idx, val, axis=axis)
        # Output shape must equal arr shape; values must be unchanged.
        np.testing.assert_equal(list(out.shape), arr_shape)
        np.testing.assert_allclose(
            out.numpy(), arr.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_index_zero_dim_values_non_zero(self):
        """Original bug: arr[2,60] idx[2,0] val[2,4] axis=1."""
        self._check([2, 60], [2, 0], [2, 4], axis=1)

    def test_index_zero_first_dim(self):
        self._check([60, 2], [0, 2], [4, 2], axis=0)

    def test_index_zero_mid_dim(self):
        self._check([3, 5, 7], [3, 0, 7], [3, 4, 7], axis=1)

    def test_all_zero_size(self):
        self._check([2, 60], [2, 0], [2, 0], axis=1)

    def test_inplace_zero_index(self):
        """Inplace variant should also return arr unchanged."""
        arr = paddle.rand([2, 60], dtype='float32')
        arr_copy = arr.clone()
        idx = paddle.zeros([2, 0], dtype='int64')
        val = paddle.rand([2, 4], dtype='float32')
        arr.put_along_axis_(idx, val, axis=1)
        np.testing.assert_allclose(
            arr.numpy(), arr_copy.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_reduce_mul_zero_index(self):
        arr = paddle.ones([2, 60], dtype='float32')
        idx = paddle.zeros([2, 0], dtype='int64')
        val = paddle.rand([2, 4], dtype='float32') + 2.0
        out = paddle.put_along_axis(arr, idx, val, axis=1, reduce='mul')
        np.testing.assert_equal(list(out.shape), [2, 60])
        np.testing.assert_allclose(
            out.numpy(), arr.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_invalid_parameters_are_checked_before_early_return(self):
        arr = paddle.ones([2, 3], dtype='float32')
        idx = paddle.zeros([2, 0], dtype='int64')
        val = paddle.ones([2, 0], dtype='float32')

        for api in (paddle.put_along_axis, paddle.tensor.put_along_axis_):
            invalid_idx = paddle.zeros([2, 0], dtype='float32')
            with self.assertRaises(TypeError):
                api(arr, invalid_idx, val, axis=1)
            with self.assertRaises(TypeError):
                api(arr, idx, val, axis=1, reduce=1)
            with self.assertRaises(ValueError):
                api(arr, idx, val, axis=1, reduce='invalid')
            with self.assertRaises(TypeError):
                api(arr, idx, val, axis=1, include_self='invalid')

        bool_arr = paddle.ones([2, 3], dtype='bool')
        with self.assertRaises(TypeError):
            paddle.put_along_axis(bool_arr, idx, val, axis=1)

    def test_zero_index_xpu(self):
        if not paddle.device.is_compiled_with_xpu():
            return
        place = paddle.XPUPlace(0)
        arr = paddle.to_tensor(
            np.random.random([2, 60]).astype('float32'), place=place
        )
        idx = paddle.to_tensor(np.zeros([2, 0], dtype='int64'), place=place)
        val = paddle.to_tensor(
            np.random.random([2, 4]).astype('float32'), place=place
        )
        out = paddle._C_ops.put_along_axis(arr, idx, val, 1, 'assign', True)
        np.testing.assert_equal(list(out.shape), [2, 60])
        np.testing.assert_allclose(
            out.numpy(), arr.numpy(), rtol=1e-6, atol=1e-6
        )


class TestPutAlongAxisZeroIndexGrad(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def _check(self, place):
        arr = paddle.to_tensor(
            np.random.rand(2, 3).astype('float32'), place=place
        )
        arr.stop_gradient = False
        index = paddle.to_tensor(np.zeros((2, 0), dtype='int64'), place=place)
        values = paddle.to_tensor(
            np.zeros((2, 0), dtype='float32'), place=place
        )
        values.stop_gradient = False
        out = paddle._C_ops.put_along_axis(
            arr, index, values, 1, 'assign', True
        )
        out.sum().backward()
        np.testing.assert_allclose(
            arr.grad.numpy(), np.ones((2, 3), dtype='float32')
        )
        self.assertIsNotNone(values.grad)
        self.assertEqual(list(values.grad.shape), [2, 0])

    def test_cpu(self):
        self._check(paddle.CPUPlace())

    def test_empty_input_cpu(self):
        place = paddle.CPUPlace()
        arr = paddle.to_tensor(np.zeros([0, 3], dtype='float32'), place=place)
        arr.stop_gradient = False
        index = paddle.to_tensor(np.zeros([0, 3], dtype='int64'), place=place)
        values = paddle.to_tensor(
            np.zeros([0, 3], dtype='float32'), place=place
        )
        values.stop_gradient = False

        out = paddle._C_ops.put_along_axis(
            arr, index, values, 1, 'assign', True
        )
        out.sum().backward()

        self.assertEqual(list(arr.grad.shape), [0, 3])
        self.assertIsNotNone(values.grad)
        self.assertEqual(list(values.grad.shape), [0, 3])

    def test_gpu(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            self._check(get_device_place())


class TestPutAlongAxisMulIntegerDivByZero(unittest.TestCase):
    """
    Bug A: reduce='mul' backward with integer dtypes crashes (SIGFPE) when
    x or value contains zero, because the grad formula divides by x or value
    without a zero guard.

    Fixed in gather_scatter_functor.cc/.cu: a zero factor is handled by the
    zero-count path and never reaches the division.

    For integer dtypes (uint8/int32/int64), Paddle does not support autograd
    (no sum_grad kernel registered for these types), so we verify only that the
    forward pass does not crash.  The div-by-zero guard in the grad kernel is
    exercised via the float32 backward test below.
    """

    def setUp(self):
        paddle.disable_static()

    def _run_forward_only(self, dtype):
        """Verify forward does not SIGFPE for integer dtypes with zeros."""
        x = paddle.to_tensor(np.array([[[1, 0, 3], [0, 5, 6]]]), dtype=dtype)
        index = paddle.to_tensor(
            np.array([[[0, 1, 0], [1, 0, 1]]]), dtype='int64'
        )
        value = paddle.to_tensor(
            np.array([[[2, 0, 4], [0, 3, 5]]]), dtype=dtype
        )
        out = paddle.put_along_axis(
            x, index, value, axis=2, reduce='mul', include_self=True
        )
        self.assertEqual(list(out.shape), [1, 2, 3])

    def test_uint8_cpu(self):
        self._run_forward_only('uint8')

    def test_int32_cpu(self):
        self._run_forward_only('int32')

    def test_int64_cpu(self):
        self._run_forward_only('int64')


class TestPutAlongAxisZeroSizeInputGrad(unittest.TestCase):
    """
    Bug B: when input has a 0-size dimension but index is non-empty,
    CPU backward crashes due to out-of-bounds memory access because
    the CPU grad kernel lacked a numel==0 early-return guard.

    Fixed in put_along_axis_grad_kernel.cc by adding the same guard
    that the GPU grad kernel already had. The guard is kept as defense in
    depth, but such shapes are now rejected up front: a 0-size input with a
    non-empty index is always out of bounds, either because index exceeds
    input on a non-axis dimension or because the scatter dimension has no
    valid index value. See ``TestPutAlongAxisZeroSizeInputInvalidIndex``.
    """

    def setUp(self):
        paddle.disable_static()

    def _run_zero_size_input(
        self, x_shape, idx_shape, val_shape, axis, reduce, place=None
    ):
        if place is None:
            place = paddle.CPUPlace()
        x = paddle.to_tensor(
            np.random.rand(*x_shape).astype('float32'), place=place
        )
        x.stop_gradient = False
        index = paddle.to_tensor(
            np.zeros(idx_shape, dtype='int64'), place=place
        )
        value = paddle.to_tensor(
            np.random.rand(*val_shape).astype('float32'), place=place
        )
        value.stop_gradient = False
        out = paddle.put_along_axis(
            x, index, value, axis=axis, reduce=reduce, include_self=True
        )
        self.assertEqual(list(out.shape), x_shape)
        loss = out.sum()
        loss.backward()
        if np.prod(idx_shape) == 0:
            self.assertIsNone(value.grad)
        else:
            self.assertIsNotNone(value.grad)
            np.testing.assert_array_equal(
                value.grad.numpy(), np.zeros(idx_shape, dtype='float32')
            )

    def test_input_first_dim_zero_assign(self):
        self._run_zero_size_input(
            [0, 60], [0, 4], [0, 4], axis=1, reduce='assign'
        )

    def test_input_first_dim_zero_add(self):
        self._run_zero_size_input([0, 60], [0, 4], [0, 4], axis=1, reduce='add')

    def test_input_first_dim_zero_mul(self):
        self._run_zero_size_input([0, 60], [0, 4], [0, 4], axis=1, reduce='mul')

    def test_input_mid_dim_zero(self):
        self._run_zero_size_input(
            [4, 0, 4], [1, 0, 1], [1, 0, 1], axis=0, reduce='assign'
        )

    def test_input_last_dim_zero(self):
        self._run_zero_size_input(
            [4, 4, 0], [1, 1, 0], [1, 1, 0], axis=0, reduce='assign'
        )


class TestPutAlongAxisZeroSizeInputInvalidIndex(unittest.TestCase):
    """
    A 0-size input combined with a non-empty index can never be valid, and
    ``broadcast=True`` used to silently ignore it while ``broadcast=False``
    and ``torch.scatter_`` both raise. Both flavours now agree.

    Two distinct reasons to reject:

    - the 0-size dimension is not ``axis``, so index exceeds input there and
      the scatter would write out of bounds. Rejected up front from the shape
      alone, mirroring the wording of the ``broadcast=False`` branch;
    - the 0-size dimension *is* ``axis``, so no index value can be in range.
      This one is reported by the kernel as an index out-of-bounds error, the
      way ``torch.scatter_`` does, which surfaces as ``IndexError``. The
      ``broadcast=False`` branch checks the index values in python first and
      raises ``RuntimeError``, so both types are accepted.
    """

    def setUp(self):
        paddle.disable_static()

    def _places(self):
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda() or is_custom_device():
            places.append(get_device_place())
        if paddle.device.is_compiled_with_xpu():
            places.append(paddle.XPUPlace(0))
        return places

    def _run(self, x_shape, idx_shape, val_shape, axis, place, broadcast):
        x = paddle.to_tensor(
            np.random.rand(*x_shape).astype('float32'), place=place
        )
        index = paddle.to_tensor(
            np.zeros(idx_shape, dtype='int64'), place=place
        )
        value = paddle.to_tensor(
            np.random.rand(*val_shape).astype('float32'), place=place
        )
        paddle.put_along_axis(
            x,
            index,
            value,
            axis=axis,
            reduce='assign',
            include_self=True,
            broadcast=broadcast,
        )

    def test_non_axis_dim_zero(self):
        for place in self._places():
            for broadcast in (True, False):
                with self.assertRaisesRegex(
                    RuntimeError, "apart from dimension"
                ):
                    self._run(
                        [0, 20], [1024, 6], [1024, 6], 1, place, broadcast
                    )

    def test_axis_dim_zero(self):
        for place in self._places():
            for broadcast in (True, False):
                with self.assertRaisesRegex(
                    (RuntimeError, IndexError), "out of bounds"
                ):
                    self._run(
                        [2, 0, 3], [2, 1, 3], [2, 1, 3], 1, place, broadcast
                    )

    def test_inplace_rejects_too(self):
        x = paddle.zeros([0, 20], dtype='float32')
        index = paddle.zeros([1024, 6], dtype='int64')
        value = paddle.zeros([1024, 6], dtype='float32')
        with self.assertRaisesRegex(RuntimeError, "apart from dimension"):
            x.put_along_axis_(index, value, axis=1)


class TestPutAlongAxisIndexLargerThanInput(unittest.TestCase):
    """
    ``indices`` may only exceed ``arr`` along ``axis``. On any other dimension
    its coordinates address ``arr`` directly, so a larger size made the scatter
    kernel write past the end of ``arr`` -- a segfault with a large enough
    index, silently ignored when ``arr`` was 0-size. ``infer_broadcast_shape``
    returns None for both situations, and the ``broadcast=True`` path used to
    read that as "just skip broadcasting" without validating anything.
    """

    def setUp(self):
        paddle.disable_static()

    def test_index_rows_exceed_input_rows(self):
        for broadcast in (True, False):
            arr = paddle.zeros([3, 4], dtype='float32')
            index = paddle.zeros([4096, 2], dtype='int64')
            value = paddle.ones([4096, 2], dtype='float32')
            with self.assertRaisesRegex(RuntimeError, "apart from dimension"):
                paddle.put_along_axis(
                    arr, index, value, axis=1, broadcast=broadcast
                )

    def test_index_longer_along_axis_is_allowed(self):
        """Exceeding ``arr`` on ``axis`` itself stays legal."""
        arr = paddle.zeros([3, 4], dtype='float32')
        index = paddle.to_tensor(
            [[0, 1, 2, 3, 0, 1], [0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3]],
            dtype='int64',
        )
        value = paddle.arange(18).astype('float32').reshape([3, 6])
        out = paddle.put_along_axis(arr, index, value, axis=1)
        expected = np.array(
            [[4.0, 5.0, 2.0, 3.0], [11.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 17.0]]
        )
        np.testing.assert_allclose(out.numpy(), expected)


class TestPutAlongAxisIncludeSelfFalseInvalidIndex(unittest.TestCase):
    """``include_self=False`` initializes scatter targets before reducing.

    That initialization has to reject index values before computing the target
    offset, otherwise an invalid index writes out of bounds before the reduce
    step ever looks at it.

    Only the CPU place is exercised. On GPU the same check is a device-side
    ``PADDLE_ENFORCE``, which prints and then traps; the trap destroys the CUDA
    context, so the failure surfaces as ``CUDA error(719)`` instead of the
    diagnostic, and every later CUDA call in the process fails as well. torch
    reports an out-of-range scatter index the same way, and for the same
    reason, so there is nothing in-process to assert there.
    """

    def setUp(self):
        paddle.disable_static()

    def test_reduce_mul_checks_index_before_initialization(self):
        place = paddle.CPUPlace()
        x = paddle.zeros([2, 3], dtype='float32').cpu()
        index = paddle.to_tensor([[10, 0, 0]], dtype='int64', place=place)
        value = paddle.ones([1, 3], dtype='float32').cpu()

        with self.assertRaisesRegex(IndexError, "expected >= -2 and < 2"):
            out = paddle._C_ops.put_along_axis(x, index, value, 0, 'mul', False)
            out.numpy()


class TestPutAlongAxisInferMetaShapeCheck(unittest.TestCase):
    """``PutAlongAxisInferMeta`` mirrors ``scatter_shape_check`` in torch.

    The python wrapper rejects a rank mismatch and normalizes ``values`` by
    broadcasting, so these constraints are only observable when the op is
    called directly. Going through ``_C_ops`` covers the static graph, PIR and
    custom-pass entries at the same time, since they all share the InferMeta.

    Shape violations are reported as ``InvalidArgument``, which surfaces as
    ``ValueError`` in python.
    """

    def setUp(self):
        paddle.disable_static()

    def _call(self, x_shape, idx_shape, val_shape, axis):
        def make(shape, dtype):
            if shape is None:  # 0-D tensor
                return paddle.zeros([], dtype=dtype)
            return paddle.zeros(shape, dtype=dtype)

        return paddle._C_ops.put_along_axis(
            make(x_shape, 'float32'),
            make(idx_shape, 'int64'),
            make(val_shape, 'float32'),
            axis,
            'assign',
            True,
        )

    def test_index_rank_mismatch(self):
        with self.assertRaisesRegex(
            ValueError, "same number of dimensions as self"
        ):
            self._call([3, 4], [3], [3], 1)

    def test_value_rank_mismatch(self):
        with self.assertRaisesRegex(
            ValueError, "same number of dimensions as value"
        ):
            self._call([3, 4], [3, 2], [3], 1)

    def test_zero_dim_input_rank_mismatch(self):
        """A 0-D tensor counts as a 1-D tensor holding a single element."""
        with self.assertRaisesRegex(
            ValueError, "same number of dimensions as self"
        ):
            self._call(None, [2, 3], [2, 3], 0)

    def test_index_larger_than_value(self):
        """Unlike ``x``, ``value`` is constrained on ``axis`` as well."""
        with self.assertRaisesRegex(ValueError, "no larger than value"):
            self._call([3, 4], [3, 2], [3, 1], 1)

    def test_index_larger_than_input_off_axis(self):
        with self.assertRaisesRegex(ValueError, "apart from dimension"):
            self._call([3, 4], [4096, 2], [4096, 2], 1)

    def test_axis_out_of_range(self):
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            self._call([3, 4], [3, 2], [3, 2], 2)

    def test_empty_index_skips_every_check(self):
        """An empty index scatters nothing, so no shape constraint applies."""
        out = self._call([8192, 32], [0, 10], [8192, 10], 1)
        self.assertEqual(list(out.shape), [8192, 32])

    def test_index_longer_along_axis_is_allowed(self):
        out = self._call([3, 4], [3, 6], [3, 6], 1)
        self.assertEqual(list(out.shape), [3, 4])

    def test_empty_index_still_checks_axis(self):
        """torch normalizes and range-checks ``dim`` in ``scatter_meta_impl``
        before the empty-index short circuit of ``scatter_shape_check``, so an
        illegal axis is reported even when nothing would be scattered."""
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            self._call([3, 4], [0, 2], [3, 2], 2)

    def test_zero_dim_input_allows_axis_minus_one(self):
        """``maybe_wrap_dim(dim, 0)`` treats a 0-D self as rank 1, so both -1
        and 0 are legal."""
        out = self._call(None, None, None, -1)
        self.assertEqual(list(out.shape), [])


def zero_dim_operand_places():
    """The places the 0-D promotion cases have to run on.

    ``get_places`` leaves CPU out unless ``FLAGS_CI_both_cpu_and_gpu`` is set,
    so on a GPU build these cases would only ever enter the ``.cu`` kernel. The
    promotion they are about lives in ``gather_scatter_functor.h``, shared by
    every backend, and only the CPU kernel is a host translation unit, so CPU
    has to be in the list for the coverage build to record those lines as run.
    """
    places = get_places()
    if not any(isinstance(place, paddle.CPUPlace) for place in places):
        places.insert(0, paddle.CPUPlace())
    return places


class TestPutAlongAxisZeroDimOperands(unittest.TestCase):
    """A 0-D operand counts as a 1-D operand holding a single element.

    ``PutAlongAxisInferMeta`` accepts these shapes, so the kernel has to be
    able to run them: it promotes the 0-D operands to rank 1 before entering
    the scatter functor, which indexes ``dims()`` and ``strides()`` directly
    and cannot address a rank-0 tensor.

    The python wrapper rejects both combinations before they get this far, so
    they are only reachable through ``_C_ops``.
    """

    def setUp(self):
        paddle.disable_static()
        self.places = zero_dim_operand_places()

    def test_all_zero_dim(self):
        for place in self.places:
            out = paddle._C_ops.put_along_axis(
                paddle.to_tensor(5.0, place=place),
                paddle.to_tensor(0, dtype='int64', place=place),
                paddle.to_tensor(9.0, place=place),
                0,
                'assign',
                True,
            )
            self.assertEqual(list(out.shape), [])
            np.testing.assert_allclose(out.numpy(), np.array(9.0))

    def test_zero_dim_input_with_rank_one_index(self):
        for place in self.places:
            out = paddle._C_ops.put_along_axis(
                paddle.to_tensor(5.0, place=place),
                paddle.to_tensor([0], dtype='int64', place=place),
                paddle.to_tensor([9.0], place=place),
                0,
                'assign',
                True,
            )
            self.assertEqual(list(out.shape), [])
            np.testing.assert_allclose(out.numpy(), np.array(9.0))

    def test_zero_dim_operands_reduce_add(self):
        for place in self.places:
            out = paddle._C_ops.put_along_axis(
                paddle.to_tensor(5.0, place=place),
                paddle.to_tensor(0, dtype='int64', place=place),
                paddle.to_tensor(9.0, place=place),
                0,
                'add',
                True,
            )
            self.assertEqual(list(out.shape), [])
            np.testing.assert_allclose(out.numpy(), np.array(14.0))

    def test_negative_axis(self):
        """``axis`` reaches the kernel as written, which normalizes it."""
        expected = np.zeros([3, 4], dtype='float32')
        expected[0, 0] = 1.0
        expected[1, 1] = 2.0
        expected[2, 2] = 3.0
        for place in self.places:
            x = paddle.to_tensor(np.zeros([3, 4], dtype='float32'), place=place)
            index = paddle.to_tensor(
                [[0], [1], [2]], dtype='int64', place=place
            )
            value = paddle.to_tensor([[1.0], [2.0], [3.0]], place=place)
            out = paddle._C_ops.put_along_axis(
                x, index, value, -1, 'assign', True
            )
            np.testing.assert_allclose(out.numpy(), expected)


class TestPutAlongAxisZeroDimOperandsGrad(unittest.TestCase):
    """The backward pass promotes 0-D operands as well.

    A 0-D tensor has ``numel() == 1``, so it gets past both 0-size early
    returns of ``PutAlongAxisGradKernel`` and reaches the same scatter functor,
    which indexes ``dims()`` and ``strides()`` directly. Reachable only through
    ``_C_ops``, like the forward cases above.
    """

    def setUp(self):
        paddle.disable_static()
        self.places = zero_dim_operand_places()

    def _run(self, reduce, place, axis=0):
        x = paddle.to_tensor(5.0, place=place)
        x.stop_gradient = False
        value = paddle.to_tensor(9.0, place=place)
        value.stop_gradient = False
        index = paddle.to_tensor(0, dtype='int64', place=place)
        out = paddle._C_ops.put_along_axis(x, index, value, axis, reduce, True)
        out.backward()
        self.assertEqual(list(x.grad.shape), [])
        self.assertEqual(list(value.grad.shape), [])
        return x.grad.numpy(), value.grad.numpy()

    def test_assign_grad(self):
        """``scatter_input_grad_kernel`` + ``scatter_value_grad_kernel``."""
        for place in self.places:
            x_grad, value_grad = self._run('assign', place)
            np.testing.assert_allclose(x_grad, np.array(0.0))
            np.testing.assert_allclose(value_grad, np.array(1.0))

    def test_add_grad(self):
        """``scatter_mean_input_grad_kernel`` is skipped for ``add``, so this
        covers ``scatter_add_mean_value_grad_kernel``."""
        for place in self.places:
            x_grad, value_grad = self._run('add', place)
            np.testing.assert_allclose(x_grad, np.array(1.0))
            np.testing.assert_allclose(value_grad, np.array(1.0))

    def test_mul_grad(self):
        """``scatter_mul_min_max_{input,value}_grad_kernel``, the pair that
        also restrides ``value``."""
        for place in self.places:
            x_grad, value_grad = self._run('mul', place)
            # out == x * value, so the gradient of each operand is the other
            # one.
            np.testing.assert_allclose(x_grad, np.array(9.0))
            np.testing.assert_allclose(value_grad, np.array(5.0))

    def test_negative_axis_grad(self):
        """``axis`` is normalized in the grad kernel too, not only the forward
        one: it arrives as the caller wrote it there as well.

        ``-1`` is the only negative axis a 0-D ``x`` admits, and it has to
        select the same element as ``0``.
        """
        for place in self.places:
            x_grad, value_grad = self._run('add', place, axis=-1)
            np.testing.assert_allclose(x_grad, np.array(1.0))
            np.testing.assert_allclose(value_grad, np.array(1.0))


class TestPutAlongAxisMulFloat32DivByZeroGrad(unittest.TestCase):
    """``reduce='mul'`` never divides by zero, whichever factor vanishes.

    The backward of a product is the product of the *other* factors, computed
    as ``out / factor`` while every factor is non-zero. A zero factor takes the
    zero-count path instead of the division, so the gradients stay finite.
    ``TestPutAlongAxisMulZeroFactorGrad`` pins down their values; these two
    only guard the multi-dimensional, non-contiguous-index shape against
    crashes and infinities.
    """

    def setUp(self):
        paddle.disable_static()

    def test_x_grad_with_zero_in_x(self):
        """x contains 0, so both the zero and the division path run."""
        cpu = paddle.CPUPlace()
        x = paddle.to_tensor(
            [[[1.0, 0.0, 3.0], [0.0, 5.0, 6.0]]],
            dtype='float32',
            place=cpu,
        )
        x.stop_gradient = False
        index = paddle.to_tensor(
            [[[0, 1, 0], [1, 0, 1]]], dtype='int64', place=cpu
        )
        value = paddle.to_tensor(
            [[[2.0, 1.0, 4.0], [1.0, 3.0, 5.0]]],
            dtype='float32',
            place=cpu,
        )
        value.stop_gradient = True
        out = paddle.put_along_axis(
            x, index, value, axis=2, reduce='mul', include_self=True
        )
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(paddle.isfinite(x.grad).all())

    def test_value_grad_with_zero_in_value(self):
        """value contains 0, so both the zero and the division path run."""
        cpu = paddle.CPUPlace()
        x = paddle.to_tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
            dtype='float32',
            place=cpu,
        )
        x.stop_gradient = True
        index = paddle.to_tensor(
            [[[0, 1, 0], [1, 0, 1]]], dtype='int64', place=cpu
        )
        value = paddle.to_tensor(
            [[[2.0, 0.0, 4.0], [0.0, 3.0, 5.0]]],
            dtype='float32',
            place=cpu,
        )
        value.stop_gradient = False
        out = paddle.put_along_axis(
            x, index, value, axis=2, reduce='mul', include_self=True
        )
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(value.grad)
        self.assertTrue(paddle.isfinite(value.grad).all())


class TestPutAlongAxisZeroDimPublicAPI(unittest.TestCase):
    """The public API reaches the 0-D kernels instead of rejecting them.

    ``non_negative_axis`` derives the legal ``axis`` range from
    ``len(arr.shape)``, which is 0 here, so it used to leave no legal value at
    all and the 0-D support of the kernel was reachable through ``_C_ops``
    only. ``torch`` treats a 0-D tensor as holding a single element on one
    dimension and accepts ``axis`` in ``[-1, 0]``.
    """

    def setUp(self):
        paddle.disable_static()
        self.places = zero_dim_operand_places()

    def _operands(self, place):
        return (
            paddle.to_tensor(5.0, place=place),
            paddle.to_tensor(0, dtype='int64', place=place),
            paddle.to_tensor(3.0, place=place),
        )

    def test_put_along_axis(self):
        for place in self.places:
            for axis in (0, -1):
                arr, index, value = self._operands(place)
                out = paddle.put_along_axis(arr, index, value, axis, 'add')
                self.assertEqual(list(out.shape), [])
                np.testing.assert_allclose(out.numpy(), np.array(8.0))

    def test_put_along_axis_no_broadcast(self):
        """The ``broadcast=False`` path has no ``arr.shape[axis]`` to read."""
        for place in self.places:
            arr, index, value = self._operands(place)
            out = paddle.put_along_axis(
                arr, index, value, 0, 'add', broadcast=False
            )
            np.testing.assert_allclose(out.numpy(), np.array(8.0))

    def test_put_along_axis_scalar_value(self):
        """``values`` is turned into a tensor on the 0-D path as well."""
        for place in self.places:
            arr, index, _ = self._operands(place)
            out = paddle.put_along_axis(arr, index, 3.0, 0, 'add')
            np.testing.assert_allclose(out.numpy(), np.array(8.0))

    def test_put_along_axis_inplace(self):
        for place in self.places:
            arr, index, value = self._operands(place)
            paddle.tensor.put_along_axis_(arr, index, value, 0, 'add')
            np.testing.assert_allclose(arr.numpy(), np.array(8.0))

    def test_scatter_inplace(self):
        """``Tensor.scatter_`` normalizes ``dim`` in its own wrapper."""
        for place in self.places:
            arr, index, value = self._operands(place)
            arr.scatter_(0, index, value, reduce='add')
            np.testing.assert_allclose(arr.numpy(), np.array(8.0))

    def test_grad(self):
        for place in self.places:
            arr, index, value = self._operands(place)
            arr.stop_gradient = False
            value.stop_gradient = False
            out = paddle.put_along_axis(arr, index, value, 0, 'add')
            out.backward()
            self.assertEqual(list(arr.grad.shape), [])
            np.testing.assert_allclose(arr.grad.numpy(), np.array(1.0))
            np.testing.assert_allclose(value.grad.numpy(), np.array(1.0))

    def test_axis_out_of_range(self):
        """``[-1, 1)`` is the only legal range, as ``torch`` reports too."""
        arr, index, value = self._operands(paddle.CPUPlace())
        for axis in (1, -2):
            with self.assertRaises(IndexError):
                paddle.put_along_axis(arr, index, value, axis)
            with self.assertRaises(IndexError):
                paddle.tensor.put_along_axis_(arr, index, value, axis)
            with self.assertRaises(IndexError):
                arr.scatter_(axis, index, value)


class TestPutAlongAxisMulZeroFactorGrad(unittest.TestCase):
    """The gradient of a product is the product of the *other* factors.

    ``reduce='mul'`` used to compute it as ``out / factor``, which is only the
    product of the others while none of them is zero: as soon as one factor is
    zero ``out`` is zero too and no longer carries the rest. The old kernel
    answered 0 for every factor in that case, but the derivative with respect to
    the single zero factor is the (generally non-zero) product of the others.
    Only two or more zeros make every gradient vanish.

    Every expectation below is what ``torch.Tensor.scatter_reduce(...,
    'prod')`` returns for the same inputs.
    """

    # (note, arr, index, value, x_grad, value_grad), include_self=True
    CASES = [
        ("no zeros", [2, 3], [0, 0], [5, 11], [55, 1], [22, 10]),
        ("only x is zero", [0, 3], [0], [7], [7, 1], [0]),
        ("only value is zero", [2, 3], [0, 0], [0, 11], [0, 1], [22, 0]),
        ("x and value zero", [0, 3], [0], [0], [0, 1], [0]),
        ("x and one value zero", [0, 3], [0, 0], [0, 11], [0, 1], [0, 0]),
        ("two values zero", [2, 3], [0, 0], [0, 0], [0, 1], [0, 0]),
    ]
    # ``include_self=False`` drops ``x`` from the product, so a zero ``x`` must
    # not be counted as a zero factor there.
    CASES_NO_SELF = [
        ("value is zero", [5, 3], [0], [0], [0, 1], [1]),
        ("one value is zero", [5, 3], [0, 0], [0, 11], [0, 1], [11, 0]),
    ]

    def setUp(self):
        paddle.disable_static()
        self.places = zero_dim_operand_places()

    def test_grad(self):
        for place in self.places:
            for inc, cases in ((True, self.CASES), (False, self.CASES_NO_SELF)):
                for note, arr, index, value, x_grad, v_grad in cases:
                    self._check(
                        place, inc, note, arr, index, value, x_grad, v_grad
                    )

    def _check(self, place, inc, note, arr, index, value, x_grad, v_grad):
        a = paddle.to_tensor(arr, dtype='float32', place=place)
        a.stop_gradient = False
        v = paddle.to_tensor(value, dtype='float32', place=place)
        v.stop_gradient = False
        out = paddle.put_along_axis(
            a,
            paddle.to_tensor(index, dtype='int64', place=place),
            v,
            0,
            'mul',
            include_self=inc,
        )
        out.backward(paddle.ones_like(out))
        where = f"'{note}' with include_self={inc} on {place}"
        np.testing.assert_allclose(
            a.grad.numpy(),
            np.array(x_grad, dtype='float32'),
            err_msg=f"x_grad of {where}",
        )
        np.testing.assert_allclose(
            v.grad.numpy(),
            np.array(v_grad, dtype='float32'),
            err_msg=f"value_grad of {where}",
        )


class TestPutAlongAxisMulInt16NegativeValue(unittest.TestCase):
    """``mul`` on int16 must not corrupt the neighbouring element.

    The GPU reduce is a sub-word ``atomicCAS`` that rebuilds the whole 4-byte
    word around the element. Widening a negative ``int16_t`` result with a plain
    ``static_cast<uint32_t>`` sign extends it, and OR-ing that in overwrites the
    other half of the word with 0xFFFF. Only an even element offset is affected,
    so element 0 is the one scattered into here and element 1 is the witness.
    """

    def setUp(self):
        paddle.disable_static()
        self.places = zero_dim_operand_places()

    def test_negative_product_keeps_neighbour(self):
        for place in self.places:
            x = paddle.to_tensor([3, 7], dtype='int16', place=place)
            index = paddle.to_tensor([0], dtype='int64', place=place)
            value = paddle.to_tensor([-2], dtype='int16', place=place)
            out = paddle.put_along_axis(x, index, value, 0, 'mul')
            np.testing.assert_array_equal(
                out.numpy(), np.array([-6, 7], dtype='int16')
            )


class TestPutAlongAxisNegativeIndexGrad(unittest.TestCase):
    """A negative subscript has to mean the same thing in the backward pass.

    ``if (index < 0) index += size`` used to live in the forward gather/scatter
    kernels only. The backward ones handed the raw subscript to the offset
    arithmetic, so a negative one became a negative offset -- an access before
    the start of the tensor. Both gradients came out wrong on CPU and on GPU,
    and the GPU reduce kernels could fault outright.

    The normalization needs the length along the axis of the tensor the
    subscript addresses, which is never ``index`` itself: the ``index shorter
    than arr`` case below is the one that tells the two apart.
    """

    REDUCES = ['assign', 'add', 'mean', 'mul', 'amin', 'amax']

    # (note, arr, index, value). ``index`` holds negative subscripts only; the
    # reference run replaces each of them with ``i + arr.shape[axis]``.
    CASES = [
        ("single element", [1.0, 2.0, 3.0], [-1], [10.0]),
        ("every position", [1.0, 2.0, 3.0], [-1, -2, -3], [10.0, 20.0, 30.0]),
        (
            "index shorter than arr",
            [1.0, 2.0, 3.0, 4.0],
            [-1, -4],
            [10.0, 20.0],
        ),
        ("duplicate subscripts", [1.0, 2.0, 3.0], [-2, -2], [10.0, 20.0]),
    ]

    def setUp(self):
        paddle.disable_static()
        self.places = zero_dim_operand_places()

    def _run(self, place, include_self, reduce, arr, index, value):
        a = paddle.to_tensor(arr, dtype='float32', place=place)
        a.stop_gradient = False
        v = paddle.to_tensor(value, dtype='float32', place=place)
        v.stop_gradient = False
        out = paddle.put_along_axis(
            a,
            paddle.to_tensor(index, dtype='int64', place=place),
            v,
            0,
            reduce,
            include_self=include_self,
        )
        out.backward(paddle.ones_like(out))
        return out.numpy(), a.grad.numpy(), v.grad.numpy()

    def test_matches_the_positive_subscript(self):
        # The positive path is the reference: same offsets, so the two runs have
        # to agree exactly, not just to a tolerance.
        for place in self.places:
            for include_self in (True, False):
                for reduce in self.REDUCES:
                    for note, arr, index, value in self.CASES:
                        got = self._run(
                            place, include_self, reduce, arr, index, value
                        )
                        want = self._run(
                            place,
                            include_self,
                            reduce,
                            arr,
                            [i + len(arr) for i in index],
                            value,
                        )
                        where = (
                            f"'{note}' reduce={reduce} "
                            f"include_self={include_self} on {place}"
                        )
                        for name, g, w in zip(
                            ('out', 'x_grad', 'value_grad'), got, want
                        ):
                            np.testing.assert_array_equal(
                                g, w, err_msg=f"{name} of {where}"
                            )

    def test_assign_grad_values(self):
        # Anchors the comparison above: ``index=[-1]`` on a 3 element input has
        # to zero the last position of ``x_grad`` and nothing else.
        for place in self.places:
            out, x_grad, v_grad = self._run(
                place, True, 'assign', [1.0, 2.0, 3.0], [-1], [10.0]
            )
            where = f"on {place}"
            np.testing.assert_array_equal(
                out, np.array([1.0, 2.0, 10.0], dtype='float32'), err_msg=where
            )
            np.testing.assert_array_equal(
                x_grad,
                np.array([1.0, 1.0, 0.0], dtype='float32'),
                err_msg=where,
            )
            np.testing.assert_array_equal(
                v_grad, np.array([1.0], dtype='float32'), err_msg=where
            )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
