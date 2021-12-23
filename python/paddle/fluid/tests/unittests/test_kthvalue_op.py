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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid


def cal_kthvalue(x, k, axis, keepdim=False):
    if axis is None:
        axis = -1
    if axis < 0:
        axis = len(x.shape) + axis
    indices = np.argsort(x, axis=axis)
    value = np.sort(x, axis=axis)
    indices = indices.take(indices=k - 1, axis=axis)
    value = value.take(indices=k - 1, axis=axis)
    if keepdim:
        indices = np.expand_dims(indices, axis)
        value = np.expand_dims(value, axis)
    return value, indices


class TestKthvalueOp(OpTest):
    def init_args(self):
        self.k = 2
        self.axis = 1

    def setUp(self):
        self.op_type = "kthvalue"
        self.dtype = np.float64
        self.input_data = np.random.random((2, 50, 32))
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis}
        output, indices = cal_kthvalue(
            self.input_data, k=self.k, axis=self.axis)
        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output()

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad(set(['X']), 'Out')


class TestKthvalueOpWithKeepdim(OpTest):
    def init_args(self):
        self.k = 2
        self.axis = 1

    def setUp(self):
        self.init_args()
        self.op_type = "kthvalue"
        self.dtype = np.float64
        self.input_data = np.random.random((2, 50, 32))
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'keepdim': True}
        output, indices = cal_kthvalue(
            self.input_data, k=self.k, axis=self.axis, keepdim=True)
        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output()

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad(set(['X']), 'Out')


class TestKthvalueOpKernels(unittest.TestCase):
    def setUp(self):
        self.axises = [2, -1, None]

    def test_kthvalue_op(self):
        paddle.disable_static()

        def test_cpu_kernel():
            shape = (2, 128, 10)
            k = 2
            paddle.set_device('cpu')
            inputs = np.random.random(shape)
            tensor = paddle.to_tensor(inputs)
            for axis in self.axises:
                value_expect, indice_expect = cal_kthvalue(inputs, k, axis)
                v, inds = paddle.kthvalue(tensor, k, axis)
                self.assertTrue(np.allclose(v.numpy(), value_expect))
                self.assertTrue(np.allclose(inds.numpy(), indice_expect))

        def test_gpu_cub_kernel():
            # when nums >= 128 and k > 0.75 * nums, it will use cub sort.
            shape = (2, 30, 250)
            k = 244
            paddle.set_device('gpu')
            inputs = np.random.random(shape)
            tensor = paddle.to_tensor(inputs)
            for axis in self.axises:
                value_expect, indice_expect = cal_kthvalue(inputs, k, axis)
                v, inds = paddle.kthvalue(tensor, k, axis)
                self.assertTrue(np.allclose(v.numpy(), value_expect))
                self.assertTrue(np.allclose(inds.numpy(), indice_expect))

        def test_gpu_hand_kernel():
            # otherwise, it will use handwrite kernel
            shape = (2, 20, 342)
            k = 28
            paddle.set_device('gpu')
            inputs = np.random.random(shape)
            tensor = paddle.to_tensor(inputs)
            for axis in self.axises:
                value_expect, indice_expect = cal_kthvalue(inputs, k, axis)
                v, inds = paddle.kthvalue(tensor, k, axis)
                self.assertTrue(np.allclose(v.numpy(), value_expect))
                self.assertTrue(np.allclose(inds.numpy(), indice_expect))

        test_cpu_kernel()
        if fluid.core.is_compiled_with_cuda():
            test_gpu_cub_kernel()
            test_gpu_hand_kernel()


class TestKthvalueOpWithNaN(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x = paddle.uniform([2, 200, 10], dtype='float32')

    def test_errors(self):
        def test_nan_in_cpu_kernel():
            paddle.set_device('cpu')
            nan_position = 100
            self.x[0, nan_position, 2] = float('nan')
            v, inds = self.x.kthvalue(k=200, axis=1)
            self.assertTrue(np.isnan(v[0, 2].numpy()[0]))
            self.assertEqual(inds[0, 2].numpy()[0], nan_position)

        def test_nan_in_gpu_cub_kernel():
            paddle.set_device('gpu')
            nan_position = 100
            self.x[0, nan_position, 2] = float('nan')
            v, inds = self.x.kthvalue(k=200, axis=1)
            self.assertTrue(np.isnan(v[0, 2].numpy()[0]))
            self.assertEqual(inds[0, 2].numpy()[0], nan_position)

        def test_nan_in_gpu_hand_kernel():
            paddle.set_device('gpu')
            x = paddle.uniform([2, 20, 10], dtype='float32')
            nan_position = 15
            x[0, nan_position, 2] = float('nan')
            v, inds = x.kthvalue(k=20, axis=1)
            self.assertTrue(np.isnan(v[0, 2].numpy()[0]))
            self.assertEqual(inds[0, 2].numpy()[0], nan_position)

        test_nan_in_cpu_kernel()
        if fluid.core.is_compiled_with_cuda():
            test_nan_in_gpu_cub_kernel()
            test_nan_in_gpu_hand_kernel()


class TestKthvalueOpErrors(unittest.TestCase):
    def setUp(self):
        self.x = paddle.uniform([2, 10, 20, 25], dtype='float32')

    def test_errors(self):
        paddle.disable_static()

        def test_k_lowrange_error():
            self.x.kthvalue(k=0, axis=2)

        self.assertRaises(ValueError, test_k_lowrange_error)

        def test_k_uprange_error():
            self.x.kthvalue(k=500, axis=2)

        self.assertRaises(ValueError, test_k_uprange_error)

        def test_dim_range_error():
            self.x.kthvalue(k=10, axis=5)

        self.assertRaises(ValueError, test_dim_range_error)


if __name__ == '__main__':
    unittest.main()
