#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import unittest

import numpy as np
from op_test import (
    OpTest,
    convert_float_to_uint16,
    get_device_place,
    is_custom_device,
)

import paddle
from paddle import base
from paddle.base import Program, core, program_guard
from paddle.framework import in_pir_mode


def baddbmm_api_for_op_test(input, x, y, beta=1.0, alpha=1.0):
    return paddle.baddbmm(input, x, y, beta=beta, alpha=alpha)


class TestBaddBmmOp(OpTest):
    # test basic
    def setUp(self):
        self.op_type = "baddbmm"
        self.prim_op_type = "comp"
        self.python_api = baddbmm_api_for_op_test
        self.public_python_api = baddbmm_api_for_op_test
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((2, 10, 5)).astype(self.dtype),
            'X': np.random.random((2, 10, 10)).astype(self.dtype),
            'Y': np.random.random((2, 10, 5)).astype(self.dtype),
        }
        self.outputs = {
            'Out': self.inputs['Input']
            + np.matmul(self.inputs['X'], self.inputs['Y'])
        }

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['Input', 'X', 'Y'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_x(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_y(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_input(self):
        self.check_grad(
            ['Input'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_float16_supported(get_device_place()),
    "core is not compiled with CUDA or not support float16",
)
class TestBaddBmmFP16Op(OpTest):
    def setUp(self):
        self.op_type = "baddbmm"
        self.python_api = baddbmm_api_for_op_test
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((1, 10, 10)).astype(self.dtype),
            'X': np.random.random((1, 10, 10)).astype(self.dtype),
            'Y': np.random.random((1, 10, 10)).astype(self.dtype),
        }
        self.outputs = {
            'Out': self.inputs['Input']
            + np.matmul(self.inputs['X'], self.inputs['Y'])
        }

        self.place = get_device_place()

    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input', 'X', 'Y'], 'Out')

    def test_check_grad_x(self):
        self.check_grad_with_place(self.place, ['X'], 'Out', no_grad_set=None)

    def test_check_grad_y(self):
        self.check_grad_with_place(self.place, ['Y'], 'Out', no_grad_set=None)

    def test_check_grad_input(self):
        self.check_grad_with_place(
            self.place, ['Input'], 'Out', no_grad_set=None
        )


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestBaddBmmBF16Op(OpTest):
    def setUp(self):
        self.op_type = "baddbmm"
        self.python_api = baddbmm_api_for_op_test
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((2, 50, 1)).astype(self.dtype),
            'X': np.random.random((2, 50, 5)).astype(self.dtype),
            'Y': np.random.random((2, 5, 10)).astype(self.dtype),
        }
        self.outputs = {
            'Out': self.inputs['Input']
            + np.matmul(self.inputs['X'], self.inputs['Y'])
        }

        self.inputs['Input'] = convert_float_to_uint16(self.inputs['Input'])
        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.inputs['Y'] = convert_float_to_uint16(self.inputs['Y'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = get_device_place()

    def init_dtype_type(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input', 'X', 'Y'], 'Out')

    def test_check_grad_x(self):
        self.check_grad_with_place(self.place, ['X'], 'Out', no_grad_set=None)

    def test_check_grad_y(self):
        self.check_grad_with_place(self.place, ['Y'], 'Out', no_grad_set=None)

    def test_check_grad_input(self):
        self.check_grad_with_place(
            self.place, ['Input'], 'Out', no_grad_set=None
        )


class TestBaddBmmOpError(unittest.TestCase):
    # test error
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of baddbmm_op must be Variable.

            input = base.create_lod_tensor(
                np.array([[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]]),
                [[2]],
                base.CPUPlace(),
            )
            x1 = base.create_lod_tensor(
                np.array([[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]]),
                [[2]],
                base.CPUPlace(),
            )
            x2 = base.create_lod_tensor(
                np.array([[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]]),
                [[2]],
                base.CPUPlace(),
            )
            # After code sinking to C++, the error type changed from TypeError to ValueError
            self.assertRaises(
                (TypeError, ValueError), paddle.baddbmm, input, x1, x2
            )

            paddle.enable_static()
            # The legacy static API rejects unsupported dtypes before appending
            # the operator.
            with paddle.pir_utils.OldIrGuard():
                main = paddle.static.Program()
                startup = paddle.static.Program()
                with paddle.static.program_guard(main, startup):
                    int_input = paddle.static.data(
                        name='int_input',
                        shape=[2, 4, 4],
                        dtype="int32",
                    )
                    x3 = paddle.static.data(
                        name='x3', shape=[2, 4, 4], dtype="int32"
                    )
                    x4 = paddle.static.data(
                        name='x4', shape=[2, 4, 4], dtype="int32"
                    )
                    self.assertRaises(
                        TypeError, paddle.baddbmm, int_input, x3, x4
                    )

            # Shape errors are validated by InferMeta with otherwise valid
            # input dtypes.
            input = paddle.static.data(
                name='input',
                shape=[2, 4, 4],
                dtype="float32",
            )
            # x and y dimension mismatch
            x5 = paddle.static.data(
                name='x5',
                shape=[2, 4, 5],
                dtype="float32",
            )
            x6 = paddle.static.data(
                name='x6',
                shape=[2, 4, 4],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.baddbmm, input, x5, x6)
            # input and x are not broadcastable
            x7 = paddle.static.data(
                name='x7',
                shape=[2, 4, 4],
                dtype="float32",
            )
            x8 = paddle.static.data(
                name='x8',
                shape=[2, 4, 4],
                dtype="float32",
            )
            input1 = paddle.static.data(
                name='input1',
                shape=[2, 2, 4],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.baddbmm, input1, x7, x8)
            # input and x are not broadcastable
            x9 = paddle.static.data(
                name='x9',
                shape=[2, 4, 4],
                dtype="float32",
            )
            x10 = paddle.static.data(
                name='x10',
                shape=[2, 4, 4],
                dtype="float32",
            )
            input2 = paddle.static.data(
                name='input2',
                shape=[2, 1, 2],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.baddbmm, input2, x9, x10)
            x11 = paddle.static.data(
                name='x11',
                shape=[2, 4, 4],
                dtype="float32",
            )
            x12 = paddle.static.data(
                name='x12', shape=[2, 4, 4], dtype="float32"
            )
            input3 = paddle.static.data(
                name='input3',
                shape=[2, 4, 2],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.baddbmm, input3, x11, x12)
            x13 = paddle.static.data(
                name='x13',
                shape=[2, 4, 4],
                dtype="float32",
            )
            x14 = paddle.static.data(
                name='x14',
                shape=[2, 4, 4],
                dtype="float32",
            )
            input4 = paddle.static.data(
                name='input4',
                shape=[2, 3, 1],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.baddbmm, input4, x13, x14)


class TestBaddBmmOp2(TestBaddBmmOp):
    # test alpha and beta
    def setUp(self):
        self.op_type = "baddbmm"
        self.prim_op_type = "comp"
        self.python_api = baddbmm_api_for_op_test
        self.public_python_api = baddbmm_api_for_op_test
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((2, 10, 5)).astype(self.dtype),
            'X': np.random.random((2, 10, 10)).astype(self.dtype),
            'Y': np.random.random((2, 10, 5)).astype(self.dtype),
        }
        self.attrs = {
            'Alpha': 0.1,
            'Beta': 1.0,
        }
        self.outputs = {
            'Out': self.attrs['Beta'] * self.inputs['Input']
            + self.attrs['Alpha']
            * np.matmul(self.inputs['X'], self.inputs['Y'])
        }


class TestBaddBmmOp3(OpTest):
    def setUp(self):
        self.op_type = "baddbmm"
        self.prim_op_type = "comp"
        self.python_api = baddbmm_api_for_op_test
        self.public_python_api = baddbmm_api_for_op_test
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((2, 10, 5)).astype(self.dtype),
            'X': np.random.random((2, 10, 10)).astype(self.dtype),
            'Y': np.random.random((2, 10, 5)).astype(self.dtype),
        }
        self.attrs = {
            'Alpha': 0.5,
            'Beta': 2.0,
        }
        self.outputs = {
            'Out': self.attrs['Beta'] * self.inputs['Input']
            + self.attrs['Alpha']
            * np.matmul(self.inputs['X'], self.inputs['Y'])
        }

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['Input', 'X', 'Y'], 'Out', check_pir=True, check_prim_pir=True
        )

    def test_check_grad_x(self):
        self.check_grad(
            ['X'], 'Out', no_grad_set=None, check_pir=True, check_prim_pir=True
        )

    def test_check_grad_y(self):
        self.check_grad(
            ['Y'], 'Out', no_grad_set=None, check_pir=True, check_prim_pir=True
        )

    def test_check_grad_input(self):
        self.check_grad(
            ['Input'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )


class TestBaddBmmOp4(OpTest):
    # test broadcast
    def setUp(self):
        self.op_type = "baddbmm"
        self.prim_op_type = "comp"
        self.python_api = baddbmm_api_for_op_test
        self.public_python_api = baddbmm_api_for_op_test
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((1, 15)).astype(self.dtype),
            'X': np.random.random((1, 50, 10)).astype(self.dtype),
            'Y': np.random.random((1, 10, 15)).astype(self.dtype),
        }
        self.attrs = {
            'Alpha': 0.5,
            'Beta': 2.0,
        }

        self.outputs = {
            'Out': self.attrs['Beta']
            * np.broadcast_to(
                self.inputs['Input'][:, np.newaxis, :], (1, 50, 15)
            )
            + self.attrs['Alpha']
            * np.matmul(self.inputs['X'], self.inputs['Y'])
        }

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad_normal(self):
        self.inputs['Input'] = np.broadcast_to(
            self.inputs['Input'][:, np.newaxis, :], (1, 50, 15)
        )
        self.check_grad(
            ['Input', 'X', 'Y'], 'Out', check_pir=True, check_prim_pir=True
        )

    def test_check_grad_x(self):
        self.check_grad(['X'], 'Out', no_grad_set=None, check_pir=True)

    def test_check_grad_y(self):
        self.check_grad(['Y'], 'Out', no_grad_set=None, check_pir=True)

    def test_check_grad_input(self):
        self.inputs['Input'] = np.broadcast_to(
            self.inputs['Input'][:, np.newaxis, :], (1, 50, 15)
        )
        self.check_grad(
            ['Input'],
            'Out',
            no_grad_set=None,
            check_pir=True,
        )


class TestBaddBmmAPI(unittest.TestCase):
    def test_batch_size_mismatch(self):
        paddle.disable_static()
        try:
            input = paddle.ones([2, 3, 4], dtype=paddle.float32)
            x = paddle.ones([2, 3, 5], dtype=paddle.float32)
            y = paddle.empty([0, 5, 4], dtype=paddle.float32)
            with self.assertRaises(ValueError):
                paddle.baddbmm(input, x, y)

            input = paddle.empty([0, 3, 4], dtype=paddle.float32)
            x = paddle.empty([0, 3, 5], dtype=paddle.float32)
            y = paddle.ones([2, 5, 4], dtype=paddle.float32)
            with self.assertRaises(ValueError):
                paddle.baddbmm(input, x, y)
        finally:
            paddle.enable_static()

    def test_dtype_mismatch(self):
        paddle.disable_static()
        try:
            input = paddle.ones([2, 2, 2], dtype=paddle.float32)
            x = paddle.ones([2, 2, 2], dtype=paddle.float64)
            y = paddle.ones([2, 2, 2], dtype=paddle.float32)
            with self.assertRaises(ValueError):
                paddle.baddbmm(input, x, y)

            x = paddle.ones([2, 2, 2], dtype=paddle.float32)
            y = paddle.ones([2, 2, 2], dtype=paddle.float64)
            with self.assertRaises(ValueError):
                paddle.baddbmm(input, x, y)
        finally:
            paddle.enable_static()

    def test_static_unknown_contraction_dim(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input = paddle.static.data(
                name='input', shape=[2, 3, 4], dtype='float32'
            )
            x = paddle.static.data(name='x', shape=[2, 3, -1], dtype='float32')
            y = paddle.static.data(name='y', shape=[2, 5, 4], dtype='float32')
            out = paddle.baddbmm(input, x, y)
            self.assertEqual(out.shape, [2, 3, 4])

    def test_api_error(self):
        data_x = np.ones((2, 2, 2)).astype(np.float32)
        data_y = np.ones((2, 2, 2)).astype(np.float32)
        data_input = np.ones((2, 2, 2)).astype(np.float32)

        paddle.disable_static()

        def test_error1():
            data_x_wrong = np.ones((2, 2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x_wrong)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input)
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error1)

        def test_error2():
            data_x_wrong = np.ones((2, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x_wrong)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input)
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error2)

        def test_error3():
            data_input_wrong = np.ones((2, 2, 2, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error3)

        def test_error4():
            data_input_wrong = np.ones((2, 5)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error4)

        def test_error5():
            data_input_wrong = np.ones((3, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error5)

        def test_error6():
            data_input_wrong = np.ones((3, 2, 1)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error6)

        def test_error7():
            data_input_wrong = np.ones((1, 2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error7)

        def test_error_y1():
            data_y_wrong = np.ones((2, 2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y_wrong)
            input = paddle.to_tensor(data_input)
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error_y1)

        def test_error_y2():
            data_y_wrong = np.ones((2, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y_wrong)
            input = paddle.to_tensor(data_input)
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error_y2)

        def test_error_y3():
            data_y_wrong = np.ones((1, 2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y_wrong)
            input = paddle.to_tensor(data_input)
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error_y3)

        paddle.enable_static()

    def test_api_normal_1(self):
        data_x = np.ones((2, 2, 2)).astype(np.float32)
        data_y = np.ones((2, 2, 2)).astype(np.float32)
        data_input = np.ones((2, 2, 2)).astype(np.float32)
        data_alpha = 0.1
        data_beta = 1.0

        paddle.disable_static()

        x = paddle.to_tensor(data_x)
        y = paddle.to_tensor(data_y)
        input = paddle.to_tensor(data_input)
        paddle_output = paddle.tensor.baddbmm(
            input=input, x=x, y=y, beta=data_beta, alpha=data_alpha
        )
        numpy_output = data_beta * data_input + data_alpha * np.matmul(
            data_x, data_y
        )

        np.testing.assert_allclose(
            numpy_output, paddle_output.numpy(), rtol=1e-05
        )

        paddle.enable_static()

    def test_float64_scale_precision(self):
        paddle.disable_static()

        input = paddle.ones([1, 1, 1], dtype=paddle.float64)
        x = paddle.ones([1, 1, 1], dtype=paddle.float64)
        y = paddle.ones([1, 1, 1], dtype=paddle.float64)
        beta = 1.0000000000000002
        alpha = 1.0000000000000004

        out = paddle.baddbmm(input, x, y, beta=beta, alpha=alpha)
        expected = np.array([[[beta + alpha]]], dtype=np.float64)
        np.testing.assert_array_equal(out.numpy(), expected)

        paddle.enable_static()

    def test_legacy_static_api(self):
        paddle.enable_static()
        input_data = np.ones([2, 2, 2], dtype=np.float32)
        x_data = np.ones([2, 2, 3], dtype=np.float32)
        y_data = np.ones([2, 3, 2], dtype=np.float32)

        with paddle.pir_utils.OldIrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                input = paddle.static.data(
                    'legacy_input', [2, 2, 2], dtype='float32'
                )
                x = paddle.static.data('legacy_x', [2, 2, 3], dtype='float32')
                y = paddle.static.data('legacy_y', [2, 3, 2], dtype='float32')
                result = paddle.baddbmm(input, x, y, beta=0.5, alpha=2.0)

                out = main.global_block().create_var(
                    name='legacy_out', shape=[2, 2, 2], dtype='float32'
                )
                out.stop_gradient = True
                returned = paddle.baddbmm(
                    input, x, y, beta=0.5, alpha=2.0, out=out
                )
                self.assertIs(returned, out)

            executor = base.Executor(base.CPUPlace())
            result_value, out_value = executor.run(
                main,
                feed={
                    'legacy_input': input_data,
                    'legacy_x': x_data,
                    'legacy_y': y_data,
                },
                fetch_list=[result, out],
            )

        expected = 0.5 * input_data + 2.0 * np.matmul(x_data, y_data)
        np.testing.assert_allclose(result_value, expected)
        np.testing.assert_allclose(out_value, expected)

    def test_api_out(self):
        if in_pir_mode():
            self.skipTest("PIR not support out tensor")
        data_x = np.ones((2, 2, 2)).astype(np.float32)
        data_y = np.ones((2, 2, 2)).astype(np.float32)
        data_input = np.ones((2, 2, 2)).astype(np.float32)
        data_alpha = 0.1
        data_beta = 1.0

        paddle.disable_static()

        x = paddle.to_tensor(data_x)
        y = paddle.to_tensor(data_y)
        input = paddle.to_tensor(data_input)
        out = paddle.zeros((2, 2, 2), dtype='float32')
        paddle_output = paddle.tensor.baddbmm(
            input=input, x=x, y=y, beta=data_beta, alpha=data_alpha, out=out
        )
        numpy_output = data_beta * data_input + data_alpha * np.matmul(
            data_x, data_y
        )

        # Check that the returned tensor is the same as the out tensor
        self.assertIs(paddle_output, out)
        # Check that the values are correct
        np.testing.assert_allclose(numpy_output, out.numpy(), rtol=1e-05)

        paddle.enable_static()

    def test_api_alias(self):
        data_x = np.ones((2, 2, 2)).astype(np.float32)
        data_y = np.ones((2, 2, 2)).astype(np.float32)
        data_input = np.ones((2, 2, 2)).astype(np.float32)
        data_alpha = 0.1
        data_beta = 1.0

        paddle.disable_static()

        x = paddle.to_tensor(data_x)
        y = paddle.to_tensor(data_y)
        input = paddle.to_tensor(data_input)

        # Test using original parameter names
        paddle_output_original = paddle.tensor.baddbmm(
            input=input, x=x, y=y, beta=data_beta, alpha=data_alpha
        )

        # Test using aliases
        paddle_output_alias = paddle.tensor.baddbmm(
            input=input, batch1=x, batch2=y, beta=data_beta, alpha=data_alpha
        )

        # Check that both outputs are the same
        np.testing.assert_allclose(
            paddle_output_original.numpy(),
            paddle_output_alias.numpy(),
            rtol=1e-05,
        )

        paddle.enable_static()

    def test_normal_backward_without_out_dtype(self):
        paddle.disable_static()
        try:
            input = paddle.ones([2, 2, 2], requires_grad=True)
            x = paddle.ones([2, 2, 2], requires_grad=True)
            y = paddle.ones([2, 2, 2], requires_grad=True)

            result = paddle.baddbmm(input, x, y, beta=0.5, alpha=2.0)
            self.assertEqual(result.dtype, paddle.float32)
            self.assertFalse(result.stop_gradient)
            result.sum().backward()
            np.testing.assert_array_equal(input.grad.numpy(), 0.5)
            np.testing.assert_array_equal(x.grad.numpy(), 4.0)
            np.testing.assert_array_equal(y.grad.numpy(), 4.0)

            method_result = input.detach().baddbmm(x.detach(), y.detach())
            np.testing.assert_array_equal(method_result.numpy(), 3.0)
        finally:
            paddle.enable_static()

    def test_out_dtype_rejects_unsupported_conversion(self):
        paddle.disable_static()
        try:
            input = paddle.ones([2, 2, 2])
            x = paddle.ones([2, 2, 2])
            y = paddle.ones([2, 2, 2])
            with self.assertRaisesRegex(TypeError, "float16 or bfloat16 x"):
                paddle.baddbmm(input, x, y, out_dtype=paddle.float64)
        finally:
            paddle.enable_static()

    def test_normal_and_out_without_out_dtype(self):
        paddle.disable_static()
        try:
            input = paddle.ones([2, 2, 2])
            x = paddle.ones([2, 2, 2])
            y = paddle.ones([2, 2, 2])
            out = paddle.empty([2, 2, 2], dtype=paddle.float32)
            result = paddle.baddbmm(input, x, y, out=out)
            self.assertIs(result, out)
            np.testing.assert_array_equal(out.numpy(), 3.0)
        finally:
            paddle.enable_static()

    def test_out_rejects_autograd(self):
        paddle.disable_static()
        try:
            input = paddle.ones([2, 2, 2], requires_grad=True)
            x = paddle.ones([2, 2, 2])
            y = paddle.ones([2, 2, 2])
            out = paddle.empty([2, 2, 2])
            with self.assertRaisesRegex(RuntimeError, "don't support"):
                paddle.baddbmm(input, x, y, out=out)

            with paddle.no_grad():
                result = paddle.baddbmm(input, x, y, out=out)
            self.assertIs(result, out)
        finally:
            paddle.enable_static()

    def test_scalar_and_vector_input_backward(self):
        paddle.disable_static()
        try:
            x = paddle.ones([2, 3, 4])
            y = paddle.ones([2, 4, 5])
            for shape, expected_grad in [((), 60.0), ((5,), 12.0)]:
                input = paddle.ones(shape, requires_grad=True)
                result = paddle.baddbmm(input, x, y, beta=2.0)
                self.assertEqual(result.shape, [2, 3, 5])
                result.sum().backward()
                self.assertEqual(input.grad.shape, list(shape))
                np.testing.assert_array_equal(input.grad.numpy(), expected_grad)
        finally:
            paddle.enable_static()

    def test_inplace_requires_exact_shape_and_dtype(self):
        paddle.disable_static()
        try:
            x = paddle.ones([2, 3, 4])
            y = paddle.ones([2, 4, 5])

            input = paddle.ones([2, 3, 5])
            result = paddle.baddbmm_(input, x, y)
            self.assertIs(result, input)

            with self.assertRaises(ValueError):
                paddle.baddbmm_(paddle.ones([3, 5]), x, y)
            with self.assertRaises(ValueError):
                paddle.baddbmm_(paddle.ones([1, 3, 5]), x, y)
            with self.assertRaises((TypeError, ValueError)):
                paddle.baddbmm_(
                    paddle.ones([2, 3, 5], dtype=paddle.float64), x, y
                )
            with self.assertRaises((TypeError, ValueError)):
                paddle.baddbmm_(input, x, y, out_dtype=paddle.float32)
        finally:
            paddle.enable_static()

    def test_signature_and_tensor_method(self):
        parameters = inspect.signature(paddle.baddbmm).parameters
        self.assertEqual(
            list(parameters),
            [
                'input',
                'x',
                'y',
                'out_dtype',
                'name',
                'beta',
                'alpha',
                'out',
            ],
        )
        for parameter in ('out_dtype', 'name'):
            self.assertEqual(
                parameters[parameter].kind,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        for parameter in ('beta', 'alpha', 'out'):
            self.assertEqual(
                parameters[parameter].kind,
                inspect.Parameter.KEYWORD_ONLY,
            )

        paddle.disable_static()
        try:
            input = paddle.ones([2, 2, 2])
            x = paddle.ones([2, 2, 2])
            y = paddle.ones([2, 2, 2])
            result = input.baddbmm(batch1=x, batch2=y)
            np.testing.assert_array_equal(result.numpy(), 3.0)

            legacy_result = paddle.baddbmm(input, x, y, 0.5, 2.0)
            expected = paddle.baddbmm(input, x, y, beta=0.5, alpha=2.0)
            np.testing.assert_array_equal(
                legacy_result.numpy(), expected.numpy()
            )
        finally:
            paddle.enable_static()

    def test_legacy_positional_arg_errors(self):
        paddle.disable_static()
        try:
            input = paddle.ones([2, 2, 2])
            x = paddle.ones([2, 2, 2])
            y = paddle.ones([2, 2, 2])

            with self.assertRaisesRegex(
                TypeError, "received too many positional arguments"
            ):
                paddle.baddbmm(
                    input,
                    x,
                    y,
                    0.5,
                    2.0,
                    None,
                    'legacy_baddbmm',
                    'extra',
                )

            conflict_cases = (
                ('beta', (0.5,), 1.0),
                ('alpha', (0.5, 2.0), 1.0),
                ('out_dtype', (0.5, 2.0, paddle.float32), paddle.float32),
                (
                    'name',
                    (0.5, 2.0, None, 'legacy_baddbmm'),
                    'duplicate_name',
                ),
            )
            for name, legacy_args, keyword_value in conflict_cases:
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(
                        TypeError,
                        rf"multiple values for argument '{name}'",
                    ),
                ):
                    paddle.baddbmm(
                        input,
                        x,
                        y,
                        *legacy_args,
                        **{name: keyword_value},
                    )
        finally:
            paddle.enable_static()

    def _check_mixed_out_dtype(self, dtype):
        for batch_size in (1, 2):
            x = paddle.ones([batch_size, 4, 3], dtype=dtype).transpose(
                [0, 2, 1]
            )
            y = paddle.ones([batch_size, 5, 4], dtype=dtype).transpose(
                [0, 2, 1]
            )

            for input_dtype in (dtype, paddle.float32):
                input = paddle.ones([5], dtype=input_dtype)
                result = paddle.baddbmm(
                    input,
                    x,
                    y,
                    beta=0.5,
                    alpha=2.0,
                    out_dtype=paddle.float32,
                )
                self.assertEqual(result.dtype, paddle.float32)
                np.testing.assert_allclose(
                    result.numpy(), 8.5, rtol=1e-5, atol=1e-5
                )

        x = paddle.ones([2, 3, 4], dtype=dtype)
        y = paddle.ones([2, 4, 5], dtype=dtype)
        input = paddle.ones([5], dtype=paddle.float32)
        out = paddle.empty([2, 3, 5], dtype=paddle.float32)
        returned = paddle.baddbmm(
            input,
            x.detach(),
            y.detach(),
            beta=0.5,
            alpha=2.0,
            out_dtype=paddle.float32,
            out=out,
        )
        self.assertIs(returned, out)
        np.testing.assert_allclose(out.numpy(), 8.5, rtol=1e-5, atol=1e-5)

        positional_result = paddle.baddbmm(
            input,
            x,
            y,
            paddle.float32,
            'baddbmm_positional',
            beta=0.5,
            alpha=2.0,
        )
        legacy_result = paddle.baddbmm(
            input,
            x,
            y,
            0.5,
            2.0,
            paddle.float32,
            'legacy_baddbmm',
        )
        for result in (positional_result, legacy_result):
            self.assertEqual(result.dtype, paddle.float32)
            np.testing.assert_allclose(
                result.numpy(), 8.5, rtol=1e-5, atol=1e-5
            )

    @unittest.skipIf(
        not core.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(),
        "CUDA is required for baddbmm out_dtype",
    )
    def test_mixed_out_dtype_fp16(self):
        if not core.is_float16_supported(paddle.CUDAPlace(0)):
            self.skipTest("Float16 is not supported")
        paddle.disable_static()
        try:
            paddle.set_device('gpu')
            self._check_mixed_out_dtype(paddle.float16)
        finally:
            paddle.enable_static()

    @unittest.skipIf(
        not core.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(),
        "CUDA is required for baddbmm out_dtype",
    )
    def test_out_dtype_rejects_autograd(self):
        if not core.is_float16_supported(paddle.CUDAPlace(0)):
            self.skipTest("Float16 is not supported")
        paddle.disable_static()
        try:
            paddle.set_device('gpu')
            for grad_input in ('input', 'x', 'y'):
                tensors = {
                    'input': paddle.ones([5], dtype=paddle.float16),
                    'x': paddle.ones([2, 3, 4], dtype=paddle.float16),
                    'y': paddle.ones([2, 4, 5], dtype=paddle.float16),
                }
                tensors[grad_input].stop_gradient = False
                with (
                    self.subTest(grad_input=grad_input),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "out_dtype does not support automatic differentiation",
                    ),
                ):
                    paddle.baddbmm(
                        tensors['input'],
                        tensors['x'],
                        tensors['y'],
                        out_dtype=paddle.float32,
                    )

            input = paddle.ones([5], dtype=paddle.float16, requires_grad=True)
            x = paddle.ones([2, 3, 4], dtype=paddle.float16, requires_grad=True)
            y = paddle.ones([2, 4, 5], dtype=paddle.float16, requires_grad=True)
            with paddle.no_grad():
                result = paddle.baddbmm(input, x, y, out_dtype=paddle.float32)
            self.assertTrue(result.stop_gradient)
            np.testing.assert_array_equal(result.numpy(), 5.0)
        finally:
            paddle.enable_static()

    @unittest.skipIf(
        not core.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(),
        "CUDA is required for baddbmm out_dtype",
    )
    def test_mixed_out_dtype_bf16(self):
        if not core.is_bfloat16_supported(paddle.CUDAPlace(0)):
            self.skipTest("BFloat16 is not supported")
        paddle.disable_static()
        try:
            paddle.set_device('gpu')
            self._check_mixed_out_dtype(paddle.bfloat16)
        finally:
            paddle.enable_static()

    @unittest.skipIf(
        not core.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(),
        "CUDA is required for baddbmm out_dtype",
    )
    def test_out_dtype_infermeta_validation(self):
        if not core.is_float16_supported(paddle.CUDAPlace(0)):
            self.skipTest("Float16 is not supported")
        paddle.disable_static()
        try:
            paddle.set_device('gpu')
            input = paddle.ones([2, 3, 5], dtype=paddle.float16)
            x = paddle.ones([2, 3, 4], dtype=paddle.float16)
            y = paddle.ones([2, 4, 5], dtype=paddle.float16)

            with self.assertRaisesRegex(ValueError, "only supports float32"):
                paddle.baddbmm(input, x, y, out_dtype=paddle.float16)
            with self.assertRaisesRegex(ValueError, "must have the same dtype"):
                paddle.baddbmm(
                    input,
                    x,
                    y.astype('float32'),
                    out_dtype=paddle.float32,
                )
            with self.assertRaisesRegex(ValueError, "must match Input\\(X\\)"):
                paddle.baddbmm(
                    input.astype('float64'),
                    x,
                    y,
                    out_dtype=paddle.float32,
                )
            with self.assertRaisesRegex(ValueError, "dimension must be 3"):
                paddle.baddbmm(
                    input,
                    x,
                    paddle.ones([4, 5], dtype=paddle.float16),
                    out_dtype=paddle.float32,
                )
        finally:
            paddle.enable_static()

    @unittest.skipIf(
        not core.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(),
        "CUDA is required for baddbmm out_dtype",
    )
    def test_mixed_out_dtype_zero_k(self):
        if not core.is_float16_supported(paddle.CUDAPlace(0)):
            self.skipTest("Float16 is not supported")
        paddle.disable_static()
        try:
            paddle.set_device('gpu')
            input = paddle.full([5], float('nan'), dtype=paddle.float32)
            x = paddle.empty([2, 3, 0], dtype=paddle.float16)
            y = paddle.empty([2, 0, 5], dtype=paddle.float16)
            result = paddle.baddbmm(
                input, x, y, beta=0.0, out_dtype=paddle.float32
            )
            np.testing.assert_array_equal(result.numpy(), 0.0)
        finally:
            paddle.enable_static()

    @unittest.skipIf(
        not core.is_compiled_with_cuda() or paddle.is_compiled_with_rocm(),
        "CUDA is required for baddbmm out_dtype",
    )
    def test_mixed_out_dtype_empty_output(self):
        if not core.is_float16_supported(paddle.CUDAPlace(0)):
            self.skipTest("Float16 is not supported")
        paddle.disable_static()
        try:
            paddle.set_device('gpu')
            cases = [
                ([0, 3, 5], [0, 3, 4], [0, 4, 5]),
                ([2, 0, 5], [2, 0, 4], [2, 4, 5]),
                ([2, 3, 0], [2, 3, 4], [2, 4, 0]),
            ]
            for input_shape, x_shape, y_shape in cases:
                for input_dtype in (paddle.float16, paddle.float32):
                    with self.subTest(
                        input_shape=input_shape, input_dtype=input_dtype
                    ):
                        input = paddle.empty(input_shape, dtype=input_dtype)
                        x = paddle.empty(x_shape, dtype=paddle.float16)
                        y = paddle.empty(y_shape, dtype=paddle.float16)
                        result = paddle.baddbmm(
                            input,
                            x,
                            y,
                            out_dtype=paddle.float32,
                        )
                        self.assertEqual(result.shape, input_shape)
                        self.assertEqual(result.dtype, paddle.float32)
                        self.assertEqual(result.numel(), 0)
        finally:
            paddle.enable_static()

    def test_out_dtype_rejects_invalid_out(self):
        paddle.disable_static()
        try:
            input = paddle.ones([2, 3, 5], dtype=paddle.float16)
            x = paddle.ones([2, 3, 4], dtype=paddle.float16)
            y = paddle.ones([2, 4, 5], dtype=paddle.float16)
            out = paddle.empty([2, 3, 5], dtype=paddle.float16)
            with self.assertRaisesRegex(TypeError, "must be paddle.float32"):
                paddle.baddbmm(input, x, y, out_dtype=paddle.float32, out=out)
        finally:
            paddle.enable_static()

    def test_out_dtype_rejects_cpu(self):
        paddle.disable_static()
        try:
            place = paddle.CPUPlace()
            input = paddle.to_tensor(
                np.ones([2, 3, 5], dtype=np.float16), place=place
            )
            x = paddle.to_tensor(
                np.ones([2, 3, 4], dtype=np.float16), place=place
            )
            y = paddle.to_tensor(
                np.ones([2, 4, 5], dtype=np.float16), place=place
            )
            with self.assertRaisesRegex(
                NotImplementedError, "only supports CUDA tensors"
            ):
                paddle.baddbmm(input, x, y, out_dtype=paddle.float32)
        finally:
            paddle.enable_static()

    def test_static_out_dtype_fails_closed(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input = paddle.static.data(
                'mixed_input', [2, 3, 5], dtype='float16'
            )
            x = paddle.static.data('mixed_x', [2, 3, 4], dtype='float16')
            y = paddle.static.data('mixed_y', [2, 4, 5], dtype='float16')
            with self.assertRaises(NotImplementedError):
                paddle.baddbmm(input, x, y, out_dtype=paddle.float32)

    def test_2d_input_without_input_grad(self):
        paddle.disable_static()
        paddle.set_device('cpu')

        input = paddle.ones([1, 4], dtype=paddle.float32)
        x = paddle.ones([1, 2, 3], dtype=paddle.float32)
        y = paddle.ones([1, 3, 4], dtype=paddle.float32)
        x.stop_gradient = False
        y.stop_gradient = False

        paddle.baddbmm(input, x, y).sum().backward()

        self.assertIsNone(input.grad)
        np.testing.assert_array_equal(x.grad.numpy(), np.full([1, 2, 3], 4.0))
        np.testing.assert_array_equal(y.grad.numpy(), np.full([1, 3, 4], 2.0))

        paddle.enable_static()

    def test_2d_input_with_input_grad(self):
        paddle.disable_static()
        paddle.set_device('cpu')

        input = paddle.ones([3, 4], dtype=paddle.float32)
        x = paddle.ones([2, 3, 5], dtype=paddle.float32)
        y = paddle.ones([2, 5, 4], dtype=paddle.float32)
        input.stop_gradient = False

        out = paddle.baddbmm(input, x, y, beta=2.0)
        out.sum().backward()

        self.assertEqual(input.grad.shape, [3, 4])
        np.testing.assert_array_equal(input.grad.numpy(), np.full([3, 4], 4.0))

        paddle.enable_static()

    def test_zero_k(self):
        paddle.disable_static()
        paddle.set_device('cpu')

        input = paddle.ones([2, 3, 4], dtype=paddle.float32)
        x = paddle.empty([2, 3, 0], dtype=paddle.float32)
        y = paddle.empty([2, 0, 4], dtype=paddle.float32)
        input.stop_gradient = False
        x.stop_gradient = False
        y.stop_gradient = False

        out = paddle.baddbmm(input, x, y, beta=2.0)
        np.testing.assert_array_equal(out.numpy(), np.full([2, 3, 4], 2.0))

        out.sum().backward()
        np.testing.assert_array_equal(
            input.grad.numpy(), np.full([2, 3, 4], 2.0)
        )
        self.assertEqual(x.grad.shape, [2, 3, 0])
        self.assertEqual(y.grad.shape, [2, 0, 4])

        paddle.enable_static()

    def test_empty_output(self):
        paddle.disable_static()
        paddle.set_device('cpu')

        cases = [
            ([0, 3, 4], [0, 3, 5], [0, 5, 4]),
            ([2, 0, 4], [2, 0, 5], [2, 5, 4]),
            ([2, 3, 0], [2, 3, 5], [2, 5, 0]),
        ]
        for input_shape, x_shape, y_shape in cases:
            with self.subTest(input_shape=input_shape):
                input = paddle.empty(input_shape, dtype=paddle.float32)
                x = paddle.empty(x_shape, dtype=paddle.float32)
                y = paddle.empty(y_shape, dtype=paddle.float32)
                input.stop_gradient = False
                x.stop_gradient = False
                y.stop_gradient = False

                out = paddle.baddbmm(input, x, y)
                self.assertEqual(out.shape, input_shape)
                self.assertEqual(out.numel(), 0)

                out.sum().backward()
                self.assertEqual(input.grad.shape, input_shape)
                self.assertEqual(x.grad.shape, x_shape)
                self.assertEqual(y.grad.shape, y_shape)

        paddle.enable_static()

    def test_zero_beta_backward_special_values(self):
        if not (core.is_compiled_with_cuda() or is_custom_device()):
            self.skipTest("CUDA is not available")

        paddle.disable_static()
        paddle.set_device('gpu')

        for value in (float('nan'), float('inf')):
            input = paddle.ones([1, 2, 2], dtype=paddle.float32)
            x = paddle.ones([1, 2, 3], dtype=paddle.float32)
            y = paddle.ones([1, 3, 2], dtype=paddle.float32)
            input.stop_gradient = False

            out = paddle.baddbmm(input, x, y, beta=0.0, alpha=1.0)
            out.backward(paddle.full(out.shape, value, dtype=paddle.float32))
            self.assertTrue(bool(paddle.isnan(input.grad).all()))

        paddle.enable_static()


class TestBaddBmmBatch1Op(OpTest):
    # test basic
    def setUp(self):
        self.op_type = "baddbmm"
        self.prim_op_type = "comp"
        self.python_api = baddbmm_api_for_op_test
        self.public_python_api = baddbmm_api_for_op_test
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((1, 10, 10)).astype(self.dtype),
            'X': np.random.random((1, 10, 10)).astype(self.dtype),
            'Y': np.random.random((1, 10, 10)).astype(self.dtype),
        }
        self.outputs = {
            'Out': self.inputs['Input']
            + np.matmul(self.inputs['X'], self.inputs['Y'])
        }

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['Input', 'X', 'Y'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_x(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_y(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_input(self):
        self.check_grad(
            ['Input'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )


class TestBaddBmmBatch1FP16Op(TestBaddBmmBatch1Op):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-2)


class TestBaddBmmBatch1Op2(TestBaddBmmBatch1Op):
    # test alpha and beta
    def setUp(self):
        self.op_type = "baddbmm"
        self.prim_op_type = "comp"
        self.python_api = baddbmm_api_for_op_test
        self.public_python_api = baddbmm_api_for_op_test
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((1, 10, 10)).astype(self.dtype),
            'X': np.random.random((1, 10, 10)).astype(self.dtype),
            'Y': np.random.random((1, 10, 10)).astype(self.dtype),
        }
        self.attrs = {
            'Alpha': 0.1,
            'Beta': 1.0,
        }
        self.outputs = {
            'Out': self.attrs['Beta'] * self.inputs['Input']
            + self.attrs['Alpha']
            * np.matmul(self.inputs['X'], self.inputs['Y'])
        }


class TestBaddBmmUnderlineAPI(unittest.TestCase):
    def test_api_error(self):
        data_x = np.ones((2, 2, 2)).astype(np.float32)
        data_y = np.ones((2, 2, 2)).astype(np.float32)
        data_input = np.ones((2, 2, 2)).astype(np.float32)

        paddle.disable_static()

        def test_error1():
            data_x_wrong = np.ones((2, 2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x_wrong)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error1)

        def test_error2():
            data_x_wrong = np.ones((2, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x_wrong)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error2)

        def test_error3():
            data_input_wrong = np.ones((2, 2, 2, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error3)

        def test_error4():
            data_input_wrong = np.ones((2, 5)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error4)

        def test_error5():
            data_input_wrong = np.ones((3, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error5)

        def test_error6():
            data_input_wrong = np.ones((3, 2, 1)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error6)

        def test_error7():
            data_input_wrong = np.ones((1, 2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error7)

        def test_error8():
            data_input_wrong = np.ones((2, 3, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error8)

        def test_error9():
            data_input_wrong = np.ones((2, 1, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error9)

        def test_error_y1():
            data_y_wrong = np.ones((2, 2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y_wrong)
            input = paddle.to_tensor(data_input)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error_y1)

        def test_error_y2():
            data_y_wrong = np.ones((2, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y_wrong)
            input = paddle.to_tensor(data_input)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error_y2)

        def test_error_y3():
            data_y_wrong = np.ones((1, 2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y_wrong)
            input = paddle.to_tensor(data_input)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error_y3)

        paddle.enable_static()

    def test_api_normal_1(self):
        data_x = np.ones((2, 2, 2)).astype(np.float32)
        data_y = np.ones((2, 2, 2)).astype(np.float32)
        data_input = np.ones((2, 2, 2)).astype(np.float32)
        data_alpha = 0.1
        data_beta = 1.0

        paddle.disable_static()

        x = paddle.to_tensor(data_x)
        y = paddle.to_tensor(data_y)
        input = paddle.to_tensor(data_input)

        numpy_output = data_beta * data_input + data_alpha * np.matmul(
            data_x, data_y
        )

        paddle_output = paddle.baddbmm_(
            input=input, x=x, y=y, beta=data_beta, alpha=data_alpha
        )

        np.testing.assert_allclose(
            numpy_output, paddle_output.numpy(), rtol=1e-05
        )

        paddle.enable_static()

    def test_api_alias(self):
        data_x = np.ones((2, 2, 2)).astype(np.float32)
        data_y = np.ones((2, 2, 2)).astype(np.float32)
        data_input = np.ones((2, 2, 2)).astype(np.float32)
        data_alpha = 0.1
        data_beta = 1.0

        paddle.disable_static()

        x = paddle.to_tensor(data_x)
        y = paddle.to_tensor(data_y)
        input = paddle.to_tensor(data_input)

        # Test using original parameter names
        paddle_output_original = paddle.baddbmm_(
            input=input.clone(), x=x, y=y, beta=data_beta, alpha=data_alpha
        )

        # Test using aliases
        paddle_output_alias = paddle.baddbmm_(
            input=input.clone(),
            batch1=x,
            batch2=y,
            beta=data_beta,
            alpha=data_alpha,
        )

        # Check that both outputs are the same
        np.testing.assert_allclose(
            paddle_output_original.numpy(),
            paddle_output_alias.numpy(),
            rtol=1e-05,
        )

        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
