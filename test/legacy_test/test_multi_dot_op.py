# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from numpy.linalg import multi_dot
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core

paddle.enable_static()


# the unittest of multi_dot
# compare the result of paddle multi_dot and numpy multi_dot
class TestMultiDotOp(OpTest):
    def setUp(self):
        self.op_type = "multi_dot"
        self.python_api = paddle.linalg.multi_dot
        self.dtype = self.get_dtype()
        self.get_inputs_and_outputs()

    def get_dtype(self):
        return "float64"

    def get_inputs_and_outputs(self):
        self.A = np.random.random((2, 8)).astype(self.dtype)
        self.B = np.random.random((8, 4)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B)]}
        self.outputs = {'Out': multi_dot([self.A, self.B])}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', check_pir=True)
        self.check_grad(['x1'], 'Out', check_pir=True)


class TestMultiDotFP16Op(TestMultiDotOp):
    def get_dtype(self):
        return "float16"


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestMultiDotBF16Op(OpTest):
    def setUp(self):
        self.op_type = "multi_dot"
        self.python_api = paddle.linalg.multi_dot
        self.dtype = self.get_dtype()
        self.get_inputs_and_outputs()
        self.place = core.CUDAPlace(0)

    def get_dtype(self):
        self.np_dtype = "float32"
        return np.uint16

    def get_inputs_and_outputs(self):
        self.A = np.random.random((2, 8)).astype(self.np_dtype)
        self.B = np.random.random((8, 4)).astype(self.np_dtype)
        self.inputs = {
            'X': [
                ('x0', convert_float_to_uint16(self.A)),
                ('x1', convert_float_to_uint16(self.B)),
            ]
        }
        self.outputs = {
            'Out': convert_float_to_uint16(multi_dot([self.A, self.B]))
        }

    def test_check_output(self):
        self.check_output_with_place(self.place, check_pir=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ['x0'], 'Out', numeric_grad_delta=0.01, check_pir=True
        )
        self.check_grad_with_place(
            self.place, ['x1'], 'Out', numeric_grad_delta=0.01, check_pir=True
        )


# (A*B)*C
class TestMultiDotOp3Mat(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((2, 10)).astype(self.dtype)
        self.B = np.random.random((10, 4)).astype(self.dtype)
        self.C = np.random.random((4, 3)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B), ('x2', self.C)]}
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', check_pir=True)
        self.check_grad(['x1'], 'Out', check_pir=True)
        self.check_grad(['x2'], 'Out', check_pir=True)


# A*(B*C)
class TestMultiDotOp3Mat2(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((3, 4)).astype(self.dtype)
        self.B = np.random.random((4, 8)).astype(self.dtype)
        self.C = np.random.random((8, 2)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B), ('x2', self.C)]}
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', check_pir=True)
        self.check_grad(['x1'], 'Out', check_pir=True)
        self.check_grad(['x2'], 'Out', check_pir=True)


class TestMultiDotOp4Mat(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((8, 6)).astype(self.dtype)
        self.B = np.random.random((6, 3)).astype(self.dtype)
        self.C = np.random.random((3, 4)).astype(self.dtype)
        self.D = np.random.random((4, 5)).astype(self.dtype)
        self.inputs = {
            'X': [
                ('x0', self.A),
                ('x1', self.B),
                ('x2', self.C),
                ('x3', self.D),
            ]
        }
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C, self.D])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', check_pir=True)
        self.check_grad(['x1'], 'Out', check_pir=True)
        self.check_grad(['x2'], 'Out', check_pir=True)
        self.check_grad(['x3'], 'Out', check_pir=True)


class TestMultiDotOpFirst1D(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random(4).astype(self.dtype)
        self.B = np.random.random((4, 3)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B)]}
        self.outputs = {'Out': multi_dot([self.A, self.B])}


class TestMultiDotOp3MatFirst1D(TestMultiDotOp3Mat):
    def get_inputs_and_outputs(self):
        self.A = np.random.random(4).astype(self.dtype)
        self.B = np.random.random((4, 3)).astype(self.dtype)
        self.C = np.random.random((3, 3)).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B), ('x2', self.C)]}
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C])}


class TestMultiDotOp4MatFirst1D(TestMultiDotOp4Mat):
    def get_inputs_and_outputs(self):
        self.A = np.random.random(4).astype(self.dtype)
        self.B = np.random.random((4, 3)).astype(self.dtype)
        self.C = np.random.random((3, 4)).astype(self.dtype)
        self.D = np.random.random((4, 5)).astype(self.dtype)
        self.inputs = {
            'X': [
                ('x0', self.A),
                ('x1', self.B),
                ('x2', self.C),
                ('x3', self.D),
            ]
        }
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C, self.D])}


class TestMultiDotOpLast1D(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((3, 6)).astype(self.dtype)
        self.B = np.random.random(6).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B)]}
        self.outputs = {'Out': multi_dot([self.A, self.B])}


class TestMultiDotOp3MatLast1D(TestMultiDotOp3Mat):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((2, 4)).astype(self.dtype)
        self.B = np.random.random((4, 3)).astype(self.dtype)
        self.C = np.random.random(3).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B), ('x2', self.C)]}
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C])}

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', check_pir=True)
        self.check_grad(['x1'], 'Out', check_pir=True)
        self.check_grad(['x2'], 'Out', check_pir=True)


class TestMultiDotOp4MatLast1D(TestMultiDotOp4Mat):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((2, 3)).astype(self.dtype)
        self.B = np.random.random((3, 2)).astype(self.dtype)
        self.C = np.random.random((2, 3)).astype(self.dtype)
        self.D = np.random.random(3).astype(self.dtype)
        self.inputs = {
            'X': [
                ('x0', self.A),
                ('x1', self.B),
                ('x2', self.C),
                ('x3', self.D),
            ]
        }
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C, self.D])}


class TestMultiDotOpFirstAndLast1D(TestMultiDotOp):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((4,)).astype(self.dtype)
        self.B = np.random.random(4).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B)]}
        self.outputs = {'Out': multi_dot([self.A, self.B])}


class TestMultiDotOp3MatFirstAndLast1D(TestMultiDotOp3Mat):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((6,)).astype(self.dtype)
        self.B = np.random.random((6, 4)).astype(self.dtype)
        self.C = np.random.random(4).astype(self.dtype)
        self.inputs = {'X': [('x0', self.A), ('x1', self.B), ('x2', self.C)]}
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C])}


class TestMultiDotOp4MatFirstAndLast1D(TestMultiDotOp4Mat):
    def get_inputs_and_outputs(self):
        self.A = np.random.random((3,)).astype(self.dtype)
        self.B = np.random.random((3, 4)).astype(self.dtype)
        self.C = np.random.random((4, 2)).astype(self.dtype)
        self.D = np.random.random(2).astype(self.dtype)
        self.inputs = {
            'X': [
                ('x0', self.A),
                ('x1', self.B),
                ('x2', self.C),
                ('x3', self.D),
            ]
        }
        self.outputs = {'Out': multi_dot([self.A, self.B, self.C, self.D])}


# python API test
class TestMultiDotOpError(unittest.TestCase):

    def test_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # The inputs type of multi_dot must be list matrix.
            input1 = 12
            self.assertRaises(
                TypeError, paddle.linalg.multi_dot, [input1, input1]
            )

            # The inputs dtype of multi_dot must be float64, float64 or float16.
            input2 = paddle.static.data(
                name='input2', shape=[10, 10], dtype="int32"
            )
            self.assertRaises(
                TypeError, paddle.linalg.multi_dot, [input2, input2]
            )

            # the number of tensor must be larger than 1
            x0 = paddle.static.data(name='x0', shape=[3, 2], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.multi_dot, [x0])

            # the first tensor must be 1D or 2D
            x1 = paddle.static.data(name='x1', shape=[3, 2, 3], dtype="float64")
            x2 = paddle.static.data(name='x2', shape=[3, 2], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.multi_dot, [x1, x2])

            # the last tensor must be 1D or 2D
            x3 = paddle.static.data(name='x3', shape=[3, 2], dtype="float64")
            x4 = paddle.static.data(name='x4', shape=[3, 2, 2], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.multi_dot, [x3, x4])

            # the tensor must be 2D, except first and last tensor
            x5 = paddle.static.data(name='x5', shape=[3, 2], dtype="float64")
            x6 = paddle.static.data(name='x6', shape=[2], dtype="float64")
            x7 = paddle.static.data(name='x7', shape=[2, 2], dtype="float64")
            self.assertRaises(ValueError, paddle.linalg.multi_dot, [x5, x6, x7])


class APITestMultiDot(unittest.TestCase):

    def test_out(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x0 = paddle.static.data(name='x0', shape=[3, 2], dtype="float64")
            x1 = paddle.static.data(name='x1', shape=[2, 3], dtype='float64')
            result = paddle.linalg.multi_dot([x0, x1])
            exe = paddle.static.Executor(paddle.CPUPlace())
            data1 = np.random.rand(3, 2).astype("float64")
            data2 = np.random.rand(2, 3).astype("float64")
            (np_res,) = exe.run(
                feed={'x0': data1, 'x1': data2}, fetch_list=[result]
            )
            expected_result = np.linalg.multi_dot([data1, data2])

        np.testing.assert_allclose(
            np_res,
            expected_result,
            rtol=1e-05,
            atol=1e-05,
            err_msg=f'two value is            {np_res}\n{expected_result}, check diff!',
        )

    def test_dygraph_without_out(self):
        paddle.disable_static()
        device = paddle.CPUPlace()
        input_array1 = np.random.rand(3, 4).astype("float64")
        input_array2 = np.random.rand(4, 3).astype("float64")
        data1 = paddle.to_tensor(input_array1)
        data2 = paddle.to_tensor(input_array2)
        out = paddle.linalg.multi_dot([data1, data2])
        expected_result = np.linalg.multi_dot([input_array1, input_array2])
        np.testing.assert_allclose(expected_result, out.numpy(), rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
