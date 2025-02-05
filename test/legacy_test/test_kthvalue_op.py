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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import core


def cal_kthvalue(x, k, axis, keepdim=False):
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
        self.k = 5
        self.axis = -1

    def init_dtype(self):
        self.dtype = np.float64

    def setUp(self):
        self.op_type = "kthvalue"
        self.prim_op_type = "prim"
        self.python_api = paddle.kthvalue
        self.public_python_api = paddle.kthvalue
        self.init_dtype()
        self.input_data = np.random.random([2, 1, 2, 4, 10]).astype(self.dtype)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis}
        output, indices = cal_kthvalue(
            self.input_data, k=self.k, axis=self.axis
        )
        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_pir=True)

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad(
            ['X'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
        )


class TestKthvalueOpFp16(TestKthvalueOp):
    def init_dtype(self):
        self.dtype = np.float16


class TestKthvalueOpWithKeepdim(OpTest):
    def init_args(self):
        self.k = 2
        self.axis = 1

    def init_dtype(self):
        self.dtype = np.float64

    def setUp(self):
        self.init_args()
        self.init_dtype()
        self.op_type = "kthvalue"
        self.prim_op_type = "prim"
        self.python_api = paddle.kthvalue
        self.public_python_api = paddle.kthvalue
        self.input_data = np.random.random([1, 3, 2, 4, 10]).astype(self.dtype)
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'keepdim': True}
        output, indices = cal_kthvalue(
            self.input_data, k=self.k, axis=self.axis, keepdim=True
        )
        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_pir=True)

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad(
            ['X'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
        )


class TestKthvalueOpWithKeepdimFp16(TestKthvalueOpWithKeepdim):
    def init_dtype(self):
        self.dtype = np.float16


class TestKthvalueOpKernels(unittest.TestCase):
    def setUp(self):
        self.axes = [2, -1]

    def test_kthvalue_op(self):
        paddle.disable_static()

        def test_cpu_kernel():
            shape = (2, 128, 10)
            k = 2
            paddle.set_device('cpu')
            inputs = np.random.random(shape)
            tensor = paddle.to_tensor(inputs)
            for axis in self.axes:
                value_expect, indice_expect = cal_kthvalue(inputs, k, axis)
                v, inds = paddle.kthvalue(tensor, k, axis)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)
                np.testing.assert_allclose(
                    inds.numpy(), indice_expect, rtol=1e-05
                )

        def test_gpu_kernel():
            shape = (2, 30, 250)
            k = 244
            paddle.set_device('gpu')
            inputs = np.random.random(shape)
            tensor = paddle.to_tensor(inputs)
            for axis in self.axes:
                value_expect, indice_expect = cal_kthvalue(inputs, k, axis)
                v, inds = paddle.kthvalue(tensor, k, axis)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)
                np.testing.assert_allclose(
                    inds.numpy(), indice_expect, rtol=1e-05
                )

        test_cpu_kernel()
        if base.core.is_compiled_with_cuda():
            test_gpu_kernel()


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
            self.assertTrue(np.isnan(v[0, 2].numpy()))
            self.assertEqual(inds[0, 2].numpy(), nan_position)

        def test_nan_in_gpu_kernel():
            paddle.set_device('gpu')
            nan_position = 100
            self.x[0, nan_position, 2] = float('nan')
            v, inds = self.x.kthvalue(k=200, axis=1)
            self.assertTrue(np.isnan(v[0, 2].numpy()))
            self.assertEqual(inds[0, 2].numpy(), nan_position)

        test_nan_in_cpu_kernel()
        if base.core.is_compiled_with_cuda():
            test_nan_in_gpu_kernel()


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

        def test_k_error_0_dim_input():
            x_0d = paddle.full([], 1)
            x_0d.kthvalue(k=8)

        self.assertRaises(ValueError, test_k_error_0_dim_input)


class TestModeOpInStatic(unittest.TestCase):
    def setUp(self):
        np.random.seed(666)
        self.input_data = np.random.random((2, 20, 1, 2, 80)).astype(np.float64)
        self.k = 10

    def test_run_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input_tensor = paddle.static.data(
                name="x", shape=[2, 20, 1, 2, 80], dtype="float64"
            )
            result = paddle.kthvalue(input_tensor, self.k, axis=1)
            expect_value = cal_kthvalue(self.input_data, self.k, axis=1)[0]
            exe = paddle.static.Executor(paddle.CPUPlace())
            paddle_result = exe.run(
                feed={"x": self.input_data}, fetch_list=[result]
            )[0]
            np.testing.assert_allclose(paddle_result, expect_value, rtol=1e-05)


class TestKthvalueFP16Op(OpTest):
    def init_args(self):
        self.k = 5
        self.axis = -1
        self.keepdim = False
        self.input_data = np.random.random((2, 1, 2, 4, 10))
        self.dtype = np.float16

    def setUp(self):
        self.op_type = "kthvalue"
        self.python_api = paddle.kthvalue
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'keepdim': self.keepdim}
        output, indices = cal_kthvalue(
            self.input_data, k=self.k, axis=self.axis, keepdim=self.keepdim
        )
        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_pir=True)

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad({'X'}, 'Out', check_pir=True)


class TestKthvalueWithKeepdimFP16Op(TestKthvalueFP16Op):
    def init_args(self):
        self.k = 2
        self.axis = 1
        self.keepdim = True
        self.input_data = np.random.random((1, 3, 2, 4, 10))
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestKthvalueBF16Op(OpTest):
    def init_args(self):
        self.k = 2
        self.axis = 1

    def setUp(self):
        self.init_args()
        self.op_type = 'kthvalue'
        self.python_api = paddle.kthvalue
        self.dtype = np.uint16
        x = np.random.random((1, 3, 2, 4, 10))
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.attrs = {'k': self.k, 'axis': self.axis, 'keepdim': True}
        out, indices = cal_kthvalue(x, k=self.k, axis=self.axis, keepdim=True)
        self.outputs = {'Out': convert_float_to_uint16(out), 'Indices': indices}

    def test_check_output(self):
        paddle.enable_static()
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        paddle.enable_static()
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, {'X'}, 'Out', check_pir=True)


if __name__ == '__main__':
    unittest.main()
