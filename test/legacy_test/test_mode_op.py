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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float

import paddle
from paddle import base
from paddle.base import core


def _mode1D(a):
    sorted_inds = np.argsort(a, kind='stable')
    sorted_array = a[sorted_inds]
    max_freq = 0
    cur_freq = 0
    mode = -1
    for i in range(len(sorted_array)):
        cur_freq += 1
        if i == len(sorted_array) - 1 or sorted_array[i] != sorted_array[i + 1]:
            if cur_freq > max_freq:
                mode = sorted_array[i]
                index = sorted_inds[i]
                max_freq = cur_freq
            cur_freq = 0
    return mode, index


def cal_mode(a, axis, keepdim=False):
    if axis < 0:
        axis = len(a.shape) + axis
    in_dims = list(range(a.ndim))
    a_view = np.transpose(a, in_dims[:axis] + in_dims[axis + 1 :] + [axis])
    inds = np.ndindex(a_view.shape[:-1])
    modes = np.empty(a_view.shape[:-1], dtype=a.dtype)
    indexes = np.empty(a_view.shape[:-1], dtype=np.int64)
    for ind in inds:
        modes[ind], indexes[ind] = _mode1D(a_view[ind])
    if keepdim:
        newshape = list(a.shape)
        newshape[axis] = 1
        modes = modes.reshape(newshape)
        indexes = indexes.reshape(newshape)
    return modes, indexes


class TestModeOp(OpTest):
    def init_args(self):
        self.axis = 1
        self.input_shape = (2, 64, 1)

    def init_input_data(self):
        self.input_data = np.random.rand(*self.input_shape).astype(self.dtype)
        self.inputs = {'X': self.input_data}

    def init_dtype(self):
        self.dtype = np.float64

    def setUp(self):
        self.op_type = "mode"
        self.python_api = paddle.mode
        self.init_dtype()
        self.init_args()
        self.init_input_data()
        self.attrs = {'axis': self.axis}
        output, indices = cal_mode(self.input_data, axis=self.axis)
        self.outputs = {'Out': output, 'Indices': indices}

    def init_numeric_grads(self):
        if self.axis < 0:
            axis = len(self.input_data.shape) + self.axis
        else:
            axis = self.axis
        if self.dtype == np.float64:
            dtype = np.float64
        else:
            dtype = np.float32
        grad = np.zeros(self.input_data.shape).astype(dtype)
        in_dims = list(range(grad.ndim))
        if axis == len(self.input_data.shape) - 1:
            a_view = grad
        else:
            a_view = np.transpose(
                grad,
                in_dims[:axis] + in_dims[axis + 1 :] + [axis],
            )
        idx = np.array(self.outputs['Indices']).flatten()
        inds = np.ndindex(a_view.shape[:-1])
        for i, ind in enumerate(inds):
            a_view[ind][idx[i]] = 1 / np.prod(self.outputs['Indices'].shape)
        if axis == len(self.input_data.shape) - 1:
            grad = a_view
        else:
            grad = np.transpose(
                a_view,
                in_dims[:axis] + in_dims[-1:] + in_dims[axis:-1],
            )
        return grad

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_pir=True)

    def test_check_grad(self):
        paddle.enable_static()
        grad = self.init_numeric_grads()
        self.check_grad({'X'}, 'Out', user_defined_grads=[grad], check_pir=True)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestModeFP16Op(TestModeOp):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestModeBF16Op(TestModeOp):
    def init_dtype(self):
        self.dtype = np.uint16

    def init_input_data(self):
        self.input_data = np.random.rand(*self.input_shape).astype(np.float32)
        self.input_data = convert_uint16_to_float(
            convert_float_to_uint16(self.input_data)
        )
        self.inputs = {'X': convert_float_to_uint16(self.input_data)}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        paddle.enable_static()
        if core.is_bfloat16_supported(place):
            self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        paddle.enable_static()
        grad = self.init_numeric_grads()

        if core.is_bfloat16_supported(place):
            self.check_grad_with_place(
                place, {'X'}, 'Out', user_defined_grads=[grad], check_pir=True
            )


class TestModeOpLastdim(TestModeOp):
    def init_args(self):
        self.axis = -1
        self.input_shape = (2, 1, 1, 2, 30)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestModeFP16OpLastdim(TestModeFP16Op):
    def init_args(self):
        self.axis = -1
        self.input_shape = (2, 1, 1, 2, 30)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestModeBF16OpLastdim(TestModeBF16Op):
    def init_args(self):
        self.axis = -1
        self.input_shape = (2, 1, 1, 2, 30)


class TestModeOpKernels(unittest.TestCase):
    def setUp(self):
        self.axes = [-1, 1]
        np.random.seed(666)
        self.inputs = np.ceil(np.random.rand(2, 10, 10) * 1000)

    def test_mode_op(self):
        def test_cpu_kernel():
            paddle.set_device('cpu')
            tensor = paddle.to_tensor(self.inputs)
            for axis in self.axes:
                value_expect, indice_expect = cal_mode(self.inputs, axis)
                v, inds = paddle.mode(tensor, axis)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)

                value_expect, indice_expect = cal_mode(
                    self.inputs, axis, keepdim=True
                )
                v, inds = paddle.mode(tensor, axis, keepdim=True)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)

        def test_gpu_kernel():
            paddle.set_device('gpu')
            tensor = paddle.to_tensor(self.inputs)
            for axis in self.axes:
                value_expect, indice_expect = cal_mode(self.inputs, axis)
                v, inds = paddle.mode(tensor, axis)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)

                value_expect, indice_expect = cal_mode(
                    self.inputs, axis, keepdim=True
                )
                v, inds = paddle.mode(tensor, axis, keepdim=True)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)

        paddle.disable_static()
        test_cpu_kernel()
        if base.core.is_compiled_with_cuda():
            test_gpu_kernel()


class TestModeOpErrors(unittest.TestCase):
    def setUp(self):
        self.x = paddle.uniform([2, 10, 20, 25], dtype='float32')

        def test_dim_range_error():
            self.x.mode(axis=5)

        self.assertRaises(ValueError, test_dim_range_error)


class TestModeOpInStatic(unittest.TestCase):
    def setUp(self):
        np.random.seed(666)
        self.input_data = np.ceil(
            np.random.random((2, 10, 10)) * 1000, dtype=np.float64
        )

    def test_run_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input_tensor = paddle.static.data(
                name="x", shape=[2, 10, 10], dtype="float64"
            )

            result = paddle.mode(input_tensor, axis=1)
            expect_value = cal_mode(self.input_data, axis=1)[0]
            exe = paddle.static.Executor(paddle.CPUPlace())
            paddle_result = exe.run(
                feed={"x": self.input_data}, fetch_list=[result]
            )[0]
            np.testing.assert_allclose(paddle_result, expect_value, rtol=1e-05)


class TestModeZeroError(unittest.TestCase):
    def test_errors(self):
        with paddle.base.dygraph.guard():

            def test_0_size():
                array = np.array([], dtype=np.float32)
                x = paddle.to_tensor(np.reshape(array, [0, 0]), dtype='float32')
                paddle.mode(x, axis=0)

            self.assertRaises(ValueError, test_0_size)


if __name__ == '__main__':
    unittest.main()
