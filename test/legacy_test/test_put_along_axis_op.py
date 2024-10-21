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
import os
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16
from utils import dygraph_guard

import paddle
from paddle.framework import core
from paddle.static import InputSpec

paddle.enable_static()


def put_along_axis_net(arr):
    indices = paddle.to_tensor([[[[2]]]], dtype='int32', stop_gradient=False)
    return paddle.tensor.put_along_axis(
        arr, indices=indices, values=-4.0, axis=-2, reduce='add'
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
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
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
        self.place = core.CUDAPlace(0)

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
        self.place = []
        self.axis = 0
        self.value_np = 99.0
        self.value_shape = []
        self.x_feed = copy.deepcopy(self.x_np)
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

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
    not core.is_compiled_with_cuda(),
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
        self.place = [paddle.CUDAPlace(0)]

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
        self.place = []
        self.axis = 0
        self.value_np = 99.0
        self.value_shape = []
        self.x_feed = copy.deepcopy(self.x_np)
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))


class TestPutAlongAxisAPICase3(TestPutAlongAxisAPI):
    def setUp(self):
        np.random.seed(0)
        self.shape = [2, 2]
        self.index_shape = [4, 2]
        self.index_np = np.array([[0, 0], [1, 0], [0, 0], [1, 0]]).astype(
            'int64'
        )
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = []
        self.axis = 0
        self.value_np = 99.0
        self.value_shape = []
        self.x_feed = copy.deepcopy(self.x_np)
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

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
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

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

    def test_error(self):
        tensorx = paddle.to_tensor([[1, 2, 3], [4, 5, 6]]).astype("float32")
        indices = paddle.to_tensor([1]).astype("int32")
        values = paddle.to_tensor([2])
        # len(arr.shape) != len(indices.shape)
        try:
            res = paddle.put_along_axis(
                tensorx, indices, 1.0, 0, 'assign', True, False
            )
        except Exception as error:
            self.assertIsInstance(error, ValueError)
        indices = paddle.to_tensor([[1]]).astype("int32")
        # len(values.shape) != len(indices.shape)
        try:
            res = paddle.put_along_axis(
                tensorx, indices, values, 0, 'assign', True, False
            )
        except Exception as error:
            self.assertIsInstance(error, ValueError)
        # len(values.shape) != len(indices.shape)
        try:
            tensorx.put_along_axis_(indices, values, 0, 'assign', True, False)
        except Exception as error:
            self.assertIsInstance(error, ValueError)
        indices = paddle.to_tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        ).astype("int32")
        # indices too large
        try:
            res = paddle.put_along_axis(
                tensorx, indices, 1.0, 0, 'assign', True, False
            )
        except Exception as error:
            self.assertIsInstance(error, RuntimeError)
        # indices too large
        try:
            tensorx.put_along_axis_(indices, 1.0, 0, 'assign', True, False)
        except Exception as error:
            self.assertIsInstance(error, RuntimeError)
        indices = paddle.to_tensor([[10]]).astype("int32")
        # the element of indices out of range
        try:
            res = paddle.put_along_axis(
                tensorx, indices, 1.0, 0, 'assign', True, False
            )
        except Exception as error:
            self.assertIsInstance(error, RuntimeError)
        # the element of indices out of range
        try:
            tensorx.put_along_axis_(indices, 1.0, 0, 'assign', True, False)
        except Exception as error:
            self.assertIsInstance(error, RuntimeError)

    def test_index_type_error(self):
        tensorx = paddle.to_tensor([[1, 2, 3], [4, 5, 6]]).astype("float32")
        indices = paddle.to_tensor([[1]]).astype("float32")
        values = paddle.to_tensor([[2]])
        with self.assertRaises(TypeError):
            res = paddle.put_along_axis(
                tensorx, indices, values, 0, 'mul', True, False
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
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

        run(paddle.CUDAPlace(0))


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
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

        run(paddle.CUDAPlace(0))


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
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

        run(paddle.CUDAPlace(0))


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
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

        run(paddle.CUDAPlace(0))


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestPutAlongAxisAPIMulUint8(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.dtype = 'uint8'
        self.x_type = "uint8"
        self.x_shape = (10, 10, 10)
        self.value_type = "uint8"
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

        run(paddle.CUDAPlace(0))


class TestPutAlongAxisDynamicShape(unittest.TestCase):
    def setUp(self):
        np.random.seed(2024)
        self.net = put_along_axis_net
        self.enable_cinn = False
        self.tol = 1e-6
        self.dtype = "float32"
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
            build_strategy = paddle.static.BuildStrategy()
            build_strategy.build_cinn_pass = self.enable_cinn
            net = paddle.jit.to_static(
                self.net,
                input_spec=self.input_specs,
                build_strategy=build_strategy,
                full_graph=True,
            )
            net.train()
        else:
            net = self.net

        res = net(arr)
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


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
