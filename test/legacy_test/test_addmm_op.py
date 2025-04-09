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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import Program, core, program_guard


class TestAddMMOp(OpTest):
    # test basic
    def setUp(self):
        self.op_type = "addmm"
        self.prim_op_type = "comp"
        self.python_api = paddle.addmm
        self.public_python_api = paddle.addmm
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
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestAddMMBF16Op(OpTest):
    def setUp(self):
        self.op_type = "addmm"
        self.python_api = paddle.addmm
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
        self.place = core.CUDAPlace(0)

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
        self.python_api = paddle.addmm
        self.public_python_api = paddle.addmm
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
        self.python_api = paddle.addmm
        self.public_python_api = paddle.addmm
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
        self.python_api = paddle.addmm
        self.public_python_api = paddle.addmm
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


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
