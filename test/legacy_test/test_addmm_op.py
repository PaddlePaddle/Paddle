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


def addmm_api_for_op_test(input, x, y, beta=1.0, alpha=1.0):
    return paddle.addmm(input, x, y, beta=beta, alpha=alpha)


class TestAddMMOp(OpTest):
    # test basic
    def setUp(self):
        self.op_type = "addmm"
        self.prim_op_type = "comp"
        self.python_api = addmm_api_for_op_test
        self.public_python_api = addmm_api_for_op_test
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((100, 1)).astype(self.dtype),
            'X': np.random.random((100, 10)).astype(self.dtype),
            'Y': np.random.random((10, 20)).astype(self.dtype),
        }
        self.outputs = {
            'Out': self.inputs['Input']
            + np.dot(self.inputs['X'], self.inputs['Y'])
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


class TestAddMMFP16Op(TestAddMMOp):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-2)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestAddMMBF16Op(OpTest):
    def setUp(self):
        self.op_type = "addmm"
        self.python_api = addmm_api_for_op_test
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((100, 1)).astype(self.np_dtype),
            'X': np.random.random((100, 10)).astype(self.np_dtype),
            'Y': np.random.random((10, 20)).astype(self.np_dtype),
        }
        self.outputs = {
            'Out': self.inputs['Input']
            + np.dot(self.inputs['X'], self.inputs['Y'])
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


class TestAddMMOpError(unittest.TestCase):
    # test error
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of addmm_op must be Variable.

            input = base.create_lod_tensor(
                np.array([[-1, -1], [-1, -1]]), [[2]], base.CPUPlace()
            )
            x1 = base.create_lod_tensor(
                np.array([[-1, -1], [-1, -1]]), [[2]], base.CPUPlace()
            )
            x2 = base.create_lod_tensor(
                np.array([[-1, -1], [-1, -1]]), [[2]], base.CPUPlace()
            )
            self.assertRaises(TypeError, paddle.addmm, input, x1, x2)

            # The input dtype of mul_op must be float32 or float64.
            input = paddle.static.data(
                name='input',
                shape=[4, 4],
                dtype="int32",
            )
            x3 = paddle.static.data(name='x3', shape=[4, 4], dtype="int32")
            x4 = paddle.static.data(name='x4', shape=[4, 4], dtype="int32")
            self.assertRaises(TypeError, paddle.addmm, input, x3, x4)
            # x and y dimension mismatch
            x5 = paddle.static.data(
                name='x5',
                shape=[4, 5],
                dtype="float32",
            )
            x6 = paddle.static.data(
                name='x6',
                shape=[4, 4],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.addmm, input, x5, x6)
            # input and x are not broadcastable
            x7 = paddle.static.data(
                name='x7',
                shape=[4, 4],
                dtype="float32",
            )
            x8 = paddle.static.data(
                name='x8',
                shape=[4, 4],
                dtype="float32",
            )
            input1 = paddle.static.data(
                name='input1',
                shape=[2, 4],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.addmm, input1, x7, x8)
            # input and x are not broadcastable
            x9 = paddle.static.data(
                name='x9',
                shape=[4, 4],
                dtype="float32",
            )
            x10 = paddle.static.data(
                name='x10',
                shape=[4, 4],
                dtype="float32",
            )
            input2 = paddle.static.data(
                name='input2',
                shape=[1, 2],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.addmm, input2, x9, x10)
            x11 = paddle.static.data(
                name='x11',
                shape=[4, 4],
                dtype="float32",
            )
            x12 = paddle.static.data(name='x12', shape=[4, 4], dtype="float32")
            input3 = paddle.static.data(
                name='input3',
                shape=[4, 2],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.addmm, input3, x11, x12)
            x13 = paddle.static.data(
                name='x13',
                shape=[4, 4],
                dtype="float32",
            )
            x14 = paddle.static.data(
                name='x14',
                shape=[4, 4],
                dtype="float32",
            )
            input4 = paddle.static.data(
                name='input4',
                shape=[3, 1],
                dtype="float32",
            )
            self.assertRaises(ValueError, paddle.addmm, input4, x13, x14)


class TestAddMMOp2(TestAddMMOp):
    # test alpha and beta
    def setUp(self):
        self.op_type = "addmm"
        self.prim_op_type = "comp"
        self.python_api = addmm_api_for_op_test
        self.public_python_api = addmm_api_for_op_test
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((20, 30)).astype(self.dtype),
            'X': np.random.random((20, 6)).astype(self.dtype),
            'Y': np.random.random((6, 30)).astype(self.dtype),
        }
        self.attrs = {
            'Alpha': 0.1,
            'Beta': 1.0,
        }
        self.outputs = {
            'Out': self.attrs['Beta'] * self.inputs['Input']
            + self.attrs['Alpha'] * np.dot(self.inputs['X'], self.inputs['Y'])
        }


class TestAddMMOp3(OpTest):
    # test broadcast
    def setUp(self):
        self.op_type = "addmm"
        self.prim_op_type = "comp"
        self.python_api = addmm_api_for_op_test
        self.public_python_api = addmm_api_for_op_test
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((1, 100)).astype(self.dtype),
            'X': np.random.random((20, 10)).astype(self.dtype),
            'Y': np.random.random((10, 100)).astype(self.dtype),
        }
        self.attrs = {
            'Alpha': 0.5,
            'Beta': 2.0,
        }
        self.outputs = {
            'Out': self.attrs['Beta'] * self.inputs['Input']
            + self.attrs['Alpha'] * np.dot(self.inputs['X'], self.inputs['Y'])
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


class TestAddMMOp4(OpTest):
    # test broadcast
    def setUp(self):
        self.op_type = "addmm"
        self.prim_op_type = "comp"
        self.python_api = addmm_api_for_op_test
        self.public_python_api = addmm_api_for_op_test
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random(100).astype(self.dtype),
            'X': np.random.random((20, 10)).astype(self.dtype),
            'Y': np.random.random((10, 100)).astype(self.dtype),
        }
        self.attrs = {
            'Alpha': 0.5,
            'Beta': 2.0,
        }
        self.outputs = {
            'Out': self.attrs['Beta'] * self.inputs['Input']
            + self.attrs['Alpha'] * np.dot(self.inputs['X'], self.inputs['Y'])
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
        self.check_grad(['X'], 'Out', no_grad_set=None, check_pir=True)

    def test_check_grad_y(self):
        self.check_grad(['Y'], 'Out', no_grad_set=None, check_pir=True)

    def test_check_grad_input(self):
        self.check_grad(
            ['Input'],
            'Out',
            no_grad_set=None,
            check_pir=True,
        )


class TestAddMMOp5(unittest.TestCase):
    def test_api_with_dygraph(self):
        np_input = np.random.random((20, 30)).astype(np.float32)
        np_x = np.random.random((20, 6)).astype(np.float32)
        np_y = np.random.random((6, 30)).astype(np.float32)

        with base.dygraph.guard():
            input = paddle.to_tensor(np_input)
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            out = paddle.tensor.addmm(input, x, y)
            np.testing.assert_allclose(
                np_input + np.dot(np_x, np_y), out.numpy(), rtol=1e-5, atol=1e-8
            )


class TestAddMMAPI(unittest.TestCase):
    def test_float64_scale_precision(self):
        paddle.disable_static()

        input = paddle.ones([1, 1], dtype=paddle.float64)
        x = paddle.ones([1, 1], dtype=paddle.float64)
        y = paddle.ones([1, 1], dtype=paddle.float64)
        input.stop_gradient = False
        x.stop_gradient = False
        y.stop_gradient = False
        beta = 1.0000000000000002
        alpha = 1.0000000000000004

        out = paddle.addmm(input, x, y, beta=beta, alpha=alpha)
        expected = np.array([[beta + alpha]], dtype=np.float64)
        np.testing.assert_array_equal(out.numpy(), expected)

        out.backward()
        np.testing.assert_array_equal(
            input.grad.numpy(), np.array([[beta]], dtype=np.float64)
        )
        np.testing.assert_array_equal(
            x.grad.numpy(), np.array([[alpha]], dtype=np.float64)
        )
        np.testing.assert_array_equal(
            y.grad.numpy(), np.array([[alpha]], dtype=np.float64)
        )

        paddle.enable_static()

    def test_api_error(self):
        data_x = np.ones((2, 2)).astype(np.float32)
        data_y = np.ones((2, 2)).astype(np.float32)
        data_input = np.ones((2, 2)).astype(np.float32)

        paddle.disable_static()

        def test_error1():
            data_x_wrong = np.ones((2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x_wrong)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input)
            out = paddle.tensor.addmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error1)

        def test_error2():
            data_x_wrong = np.ones(2).astype(np.float32)
            x = paddle.to_tensor(data_x_wrong)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input)
            out = paddle.tensor.addmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error2)

        def test_error3():
            data_input_wrong = np.ones((2, 2, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.tensor.addmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error3)

        def test_error4():
            data_input_wrong = np.ones(5).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.tensor.addmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error4)

        paddle.enable_static()

    def test_api_normal_1(self):
        data_x = np.ones((2, 2)).astype(np.float32)
        data_y = np.ones((2, 2)).astype(np.float32)
        data_input = np.ones((2, 2)).astype(np.float32)
        data_alpha = 0.1
        data_beta = 1.0

        paddle.disable_static()

        x = paddle.to_tensor(data_x)
        y = paddle.to_tensor(data_y)
        input = paddle.to_tensor(data_input)
        paddle_output = paddle.tensor.addmm(
            input=input, x=x, y=y, beta=data_beta, alpha=data_alpha
        )
        numpy_output = data_beta * data_input + data_alpha * np.dot(
            data_x, data_y
        )

        np.testing.assert_allclose(
            numpy_output, paddle_output.numpy(), rtol=1e-05
        )

        paddle.enable_static()

    def test_api_normal_2(self):
        data_x = np.ones((3, 10)).astype(np.float32)
        data_y = np.ones((10, 3)).astype(np.float32)
        data_input = np.ones(3).astype(np.float32)
        data_alpha = 0.1
        data_beta = 1.0

        paddle.disable_static()

        x = paddle.to_tensor(data_x)
        y = paddle.to_tensor(data_y)
        input = paddle.to_tensor(data_input)
        paddle_output = paddle.tensor.addmm(
            input=input, x=x, y=y, beta=data_beta, alpha=data_alpha
        )
        numpy_output = data_beta * data_input + data_alpha * np.dot(
            data_x, data_y
        )

        np.testing.assert_allclose(
            numpy_output, paddle_output.numpy(), rtol=1e-05
        )

        paddle.enable_static()

    def test_api_normal_3(self):
        data_x = np.ones((3, 10)).astype(np.float32)
        data_y = np.ones((10, 3)).astype(np.float32)
        data_input = np.ones(1).astype(np.float32)
        data_alpha = 0.1
        data_beta = 1.0

        paddle.disable_static()

        x = paddle.to_tensor(data_x)
        y = paddle.to_tensor(data_y)
        input = paddle.to_tensor(data_input)
        paddle_output = paddle.tensor.addmm(
            input=input, x=x, y=y, beta=data_beta, alpha=data_alpha
        )
        numpy_output = data_beta * data_input + data_alpha * np.dot(
            data_x, data_y
        )

        np.testing.assert_allclose(
            numpy_output, paddle_output.numpy(), rtol=1e-05
        )

        paddle.enable_static()

    def test_1d_input_without_input_grad(self):
        paddle.disable_static()
        paddle.set_device('cpu')

        input = paddle.ones([4], dtype=paddle.float32)
        x = paddle.ones([2, 3], dtype=paddle.float32)
        y = paddle.ones([3, 4], dtype=paddle.float32)
        x.stop_gradient = False
        y.stop_gradient = False

        paddle.addmm(input, x, y).sum().backward()

        self.assertIsNone(input.grad)
        np.testing.assert_array_equal(x.grad.numpy(), np.full([2, 3], 4.0))
        np.testing.assert_array_equal(y.grad.numpy(), np.full([3, 4], 2.0))

        paddle.enable_static()


class TestAddmmOutDtypeDynamicOnly(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def _skip_if_no_fp16_cuda(self):
        if not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm():
            self.skipTest("CUDA is required for addmm out_dtype")
        if paddle.device.cuda.get_device_capability() < (5, 3):
            self.skipTest(
                "FP16 addmm out_dtype requires CUDA compute capability >= 5.3"
            )

    def _skip_if_no_bf16_cuda(self):
        if not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm():
            self.skipTest("CUDA is required for addmm out_dtype")
        if paddle.device.cuda.get_device_capability()[0] < 8:
            self.skipTest(
                "BF16 addmm out_dtype requires CUDA compute capability >= 8"
            )

    def test_signature_and_legacy_positional_args(self):
        parameters = inspect.signature(paddle.addmm).parameters
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

        input = paddle.randn([3, 5], dtype='float32')
        x = paddle.randn([3, 4], dtype='float32')
        y = paddle.randn([4, 5], dtype='float32')
        result = paddle.addmm(input, x, y, 0.5, 1.5, 'legacy_addmm')
        expected = paddle.addmm(input, x, y, beta=0.5, alpha=1.5)
        np.testing.assert_allclose(
            result.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_normal_and_out_without_out_dtype(self):
        input = paddle.randn([3, 5], dtype='float32')
        x = paddle.randn([3, 4], dtype='float32')
        y = paddle.randn([4, 5], dtype='float32')
        expected = input + paddle.mm(x, y)

        result = paddle.addmm(input, x, y)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)

        out = paddle.empty([3, 5], dtype='float32')
        result = paddle.addmm(input, x, y, out=out)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-5)

    def test_fp16_to_fp32(self):
        self._skip_if_no_fp16_cuda()
        input = paddle.randn([5], dtype='float16')
        x = paddle.randn([3, 4], dtype='float16')
        y = paddle.randn([4, 5], dtype='float16')
        result = paddle.addmm(
            input, x, y, beta=0.5, alpha=1.5, out_dtype=paddle.float32
        )
        expected = paddle.addmm(
            input.astype('float32'),
            x.astype('float32'),
            y.astype('float32'),
            beta=0.5,
            alpha=1.5,
        )
        self.assertEqual(result.dtype, paddle.float32)
        np.testing.assert_allclose(
            result.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3
        )

    def test_fp16_to_fp32_positional_args(self):
        self._skip_if_no_fp16_cuda()
        input = paddle.randn([5], dtype='float16')
        x = paddle.randn([3, 4], dtype='float16')
        y = paddle.randn([4, 5], dtype='float16')
        result = paddle.addmm(
            input,
            x,
            y,
            paddle.float32,
            'addmm_positional',
            beta=0.5,
            alpha=1.5,
        )
        expected = paddle.addmm(
            input.astype('float32'),
            x.astype('float32'),
            y.astype('float32'),
            beta=0.5,
            alpha=1.5,
        )
        self.assertEqual(result.dtype, paddle.float32)
        np.testing.assert_allclose(
            result.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3
        )

    def test_fp16_to_fp32_with_float32_input_and_out(self):
        self._skip_if_no_fp16_cuda()
        input = paddle.randn([1, 5], dtype='float32')
        x = paddle.randn([3, 4], dtype='float16')
        y = paddle.randn([4, 5], dtype='float16')
        out = paddle.empty([3, 5], dtype='float32')
        result = paddle.addmm(
            input,
            x,
            y,
            beta=0.25,
            alpha=2.0,
            out_dtype=paddle.float32,
            out=out,
        )
        expected = paddle.addmm(
            input,
            x.astype('float32'),
            y.astype('float32'),
            beta=0.25,
            alpha=2.0,
        )
        np.testing.assert_allclose(
            result.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3
        )

    def test_bf16_to_fp32(self):
        self._skip_if_no_bf16_cuda()
        input = paddle.randn([3, 5], dtype='bfloat16')
        x = paddle.randn([3, 4], dtype='bfloat16')
        y = paddle.randn([4, 5], dtype='bfloat16')
        result = paddle.addmm(input, x, y, out_dtype=paddle.float32)
        expected = paddle.addmm(
            input.astype('float32'),
            x.astype('float32'),
            y.astype('float32'),
        )
        self.assertEqual(result.dtype, paddle.float32)
        np.testing.assert_allclose(
            result.numpy(), expected.numpy(), rtol=1e-2, atol=1e-2
        )

    def test_out_dtype_is_forward_only(self):
        self._skip_if_no_fp16_cuda()
        input = paddle.randn([3, 5], dtype='float16')
        x = paddle.randn([3, 4], dtype='float16')
        y = paddle.randn([4, 5], dtype='float16')
        input.stop_gradient = False
        x.stop_gradient = False
        y.stop_gradient = False
        result = paddle.addmm(input, x, y, out_dtype=paddle.float32)
        self.assertTrue(result.stop_gradient)

    def test_out_rejects_autograd(self):
        input = paddle.randn([3, 5], dtype='float32')
        x = paddle.randn([3, 4], dtype='float32')
        y = paddle.randn([4, 5], dtype='float32')
        out = paddle.empty([3, 5], dtype='float32')
        x.stop_gradient = False
        with self.assertRaises(RuntimeError):
            paddle.addmm(input, x, y, out=out)

        with paddle.no_grad():
            paddle.addmm(input, x, y, out=out)

    def test_out_dtype_rejects_unsupported_cases(self):
        input = paddle.randn([3, 5], dtype='float16')
        x = paddle.randn([3, 4], dtype='float16')
        y = paddle.randn([4, 5], dtype='float16')
        with self.assertRaises(TypeError):
            paddle.addmm(input, x, y, out_dtype=paddle.float16)
        with self.assertRaises(TypeError):
            paddle.addmm(
                input,
                x,
                y,
                out_dtype=paddle.float32,
                out=paddle.empty([3, 5], dtype='float16'),
            )
        with self.assertRaises(TypeError):
            paddle.addmm(
                input,
                x,
                y.astype('bfloat16'),
                out_dtype=paddle.float32,
            )

    def test_out_dtype_rejects_cpu(self):
        place = paddle.CPUPlace()
        input = paddle.to_tensor(np.ones([3, 5], dtype=np.float16), place=place)
        x = paddle.to_tensor(np.ones([3, 4], dtype=np.float16), place=place)
        y = paddle.to_tensor(np.ones([4, 5], dtype=np.float16), place=place)

        with self.assertRaisesRegex(
            NotImplementedError, "only supports CUDA tensors"
        ):
            paddle.addmm(input, x, y, out_dtype=paddle.float32)

    def test_invalid_dtype_is_checked_before_device(self):
        place = paddle.CPUPlace()
        input = paddle.to_tensor(np.ones([3, 5], dtype=np.float32), place=place)
        x = paddle.to_tensor(np.ones([3, 4], dtype=np.float32), place=place)
        y = paddle.to_tensor(np.ones([4, 5], dtype=np.float32), place=place)

        with self.assertRaisesRegex(TypeError, "float16 or bfloat16 x"):
            paddle.addmm(input, x, y, out_dtype=paddle.float32)

    @unittest.skipUnless(
        paddle.is_compiled_with_rocm(),
        "ROCm is required for the addmm out_dtype backend check",
    )
    def test_out_dtype_rejects_rocm(self):
        place = paddle.CUDAPlace(0)
        input = paddle.to_tensor(np.ones([3, 5], dtype=np.float16), place=place)
        x = paddle.to_tensor(np.ones([3, 4], dtype=np.float16), place=place)
        y = paddle.to_tensor(np.ones([4, 5], dtype=np.float16), place=place)

        with self.assertRaisesRegex(
            NotImplementedError, "only supports CUDA tensors"
        ):
            paddle.addmm(input, x, y, out_dtype=paddle.float32)

    def test_inplace_requires_exact_output_shape(self):
        input = paddle.ones([3, 5], dtype='float32')
        x = paddle.ones([3, 4], dtype='float32')
        y = paddle.ones([4, 5], dtype='float32')
        result = paddle.addmm_(input, x, y)
        np.testing.assert_array_equal(result.numpy(), np.full([3, 5], 5.0))

        with self.assertRaises(ValueError):
            paddle.addmm_(paddle.ones([5]), x, y)
        with self.assertRaises(ValueError):
            paddle.addmm_(input, x, y, out_dtype=paddle.float32)

    def test_static_out_dtype_fails_closed(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input = paddle.static.data('input', [3, 5], dtype='float16')
            x = paddle.static.data('x', [3, 4], dtype='float16')
            y = paddle.static.data('y', [4, 5], dtype='float16')
            with self.assertRaises(NotImplementedError):
                paddle.addmm(input, x, y, out_dtype=paddle.float32)


class TestAddmmOp_ZeroSize(OpTest):
    def setUp(self):
        self.op_type = "addmm"
        self.python_api = addmm_api_for_op_test
        self.public_python_api = addmm_api_for_op_test
        self.init_dtype_type()
        self.init_input()
        self.attrs = {
            'Alpha': 0.5,
            'Beta': 2.0,
        }
        self.outputs = {
            'Out': self.attrs['Beta'] * self.inputs['Input']
            + self.attrs['Alpha'] * np.dot(self.inputs['X'], self.inputs['Y'])
        }

    def init_input(self):
        # result shape: [20, 100]
        self.inputs = {
            'Input': np.random.random(100).astype(self.dtype),
            'X': np.random.random((20, 0)).astype(self.dtype),
            'Y': np.random.random((0, 100)).astype(self.dtype),
        }

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(['Input', 'X', 'Y'], 'Out', check_pir=True)


class TestAddmmOp_ZeroSize2(TestAddmmOp_ZeroSize):
    def init_input(self):
        # result shape: [20, 0]
        self.inputs = {
            'Input': np.random.random(0).astype(self.dtype),
            'X': np.random.random((20, 100)).astype(self.dtype),
            'Y': np.random.random((100, 0)).astype(self.dtype),
        }


class TestAddmmOp_ZeroSize3(TestAddmmOp_ZeroSize):
    def init_input(self):
        # result shape: [0, 0]
        self.inputs = {
            'Input': np.random.random(0).astype(self.dtype),
            'X': np.random.random((0, 100)).astype(self.dtype),
            'Y': np.random.random((100, 0)).astype(self.dtype),
        }


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
