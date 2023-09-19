# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base.framework import Program, program_guard

paddle.enable_static()


class TestStackOpBase(OpTest):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (5, 6, 7)
        self.axis = 0
        self.dtype = 'float64'

    def initParameters(self):
        pass

    def get_x_names(self):
        x_names = []
        for i in range(self.num_inputs):
            x_names.append(f'x{i}')
        return x_names

    def setUp(self):
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = 'stack'
        self.prim_op_type = "comp"
        self.python_api = paddle.stack
        self.public_python_api = paddle.stack
        self.x = []
        for i in range(self.num_inputs):
            self.x.append(
                np.random.random(size=self.input_dim).astype(self.dtype)
            )

        tmp = []
        x_names = self.get_x_names()
        for i in range(self.num_inputs):
            tmp.append((x_names[i], self.x[i]))

        self.inputs = {'X': tmp}
        self.outputs = {'Y': np.stack(self.x, axis=self.axis)}
        self.attrs = {'axis': self.axis}

    def test_check_output(self):
        self.check_output(check_prim=True)

    def test_check_grad(self):
        self.check_grad(self.get_x_names(), 'Y', check_prim=True)


class TestStackOp1(TestStackOpBase):
    def initParameters(self):
        self.num_inputs = 8


class TestStackOp2(TestStackOpBase):
    def initParameters(self):
        self.num_inputs = 10


class TestStackOp3(TestStackOpBase):
    def initParameters(self):
        self.axis = -1


class TestStackOp4(TestStackOpBase):
    def initParameters(self):
        self.axis = -4


class TestStackOp5(TestStackOpBase):
    def initParameters(self):
        self.axis = 1


class TestStackOp6(TestStackOpBase):
    def initParameters(self):
        self.axis = 3


class TestStackOp_ZeroDim(TestStackOpBase):
    def initParameters(self):
        self.input_dim = ()
        self.enable_cinn = False


class TestStackFP16Op(TestStackOpBase):
    def initParameters(self):
        self.dtype = np.float16


class TestStackFP16Op1(TestStackOpBase):
    def initParameters(self):
        self.dtype = np.float16
        self.num_inputs = 8


class TestStackFP16Op2(TestStackOpBase):
    def initParameters(self):
        self.dtype = np.float16
        self.num_inputs = 10


class TestStackFP16Op3(TestStackOpBase):
    def initParameters(self):
        self.dtype = np.float16
        self.axis = -1


class TestStackFP16Op4(TestStackOpBase):
    def initParameters(self):
        self.dtype = np.float16
        self.axis = -4


class TestStackFP16Op5(TestStackOpBase):
    def initParameters(self):
        self.dtype = np.float16
        self.axis = 1


class TestStackFP16Op6(TestStackOpBase):
    def initParameters(self):
        self.dtype = np.float16
        self.axis = 3


class TestStackBF16Op(OpTest):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (5, 6, 7)
        self.axis = 0
        self.dtype = np.uint16

    def initParameters(self):
        pass

    def get_x_names(self):
        x_names = []
        for i in range(self.num_inputs):
            x_names.append(f'x{i}')
        return x_names

    def setUp(self):
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = 'stack'
        self.prim_op_type = "comp"
        self.python_api = paddle.stack
        self.public_python_api = paddle.stack
        self.x = []
        for i in range(self.num_inputs):
            self.x.append(
                np.random.random(size=self.input_dim).astype(np.float32)
            )

        out = np.stack(self.x, axis=self.axis)

        tmp = []
        x_names = self.get_x_names()
        for i in range(self.num_inputs):
            tmp.append((x_names[i], convert_float_to_uint16(self.x[i])))

        self.inputs = {'X': tmp}
        self.outputs = {'Y': convert_float_to_uint16(out)}
        self.attrs = {'axis': self.axis}

    def test_check_output(self):
        self.check_output(check_prim=True)

    def test_check_grad(self):
        self.check_grad(self.get_x_names(), 'Y', check_prim=True)


class TestStackAPIWithLoDTensorArray(unittest.TestCase):
    """
    Test stack api when the input(x) is a LoDTensorArray.
    """

    def setUp(self):
        self.axis = 1
        self.iter_num = 3
        self.input_shape = [2, 3]
        self.x = np.random.random(self.input_shape).astype("float32")
        self.place = (
            base.CUDAPlace(0)
            if base.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        self.set_program()

    def set_program(self):
        self.program = base.Program()
        with base.program_guard(self.program):
            input = paddle.assign(self.x)
            tensor_array = paddle.tensor.create_array(dtype='float32')
            zero = paddle.tensor.fill_constant(
                shape=[1], value=0, dtype="int64"
            )

            for i in range(self.iter_num):
                paddle.tensor.array_write(input, zero + i, tensor_array)

            self.out_var = paddle.stack(tensor_array, axis=self.axis)

    def test_case(self):
        self.assertTrue(self.out_var.shape[self.axis] == -1)
        exe = base.Executor(self.place)
        res = exe.run(self.program, fetch_list=self.out_var)
        np.testing.assert_array_equal(
            res[0], np.stack([self.x] * self.iter_num, axis=self.axis)
        )


class TestTensorStackAPIWithLoDTensorArray(unittest.TestCase):
    """
    Test stack api when the input(x) is a LoDTensorArray.
    """

    def setUp(self):
        self.axis = 1
        self.iter_num = 3
        self.input_shape = [2, 3]
        self.x = np.random.random(self.input_shape).astype("float32")
        self.place = (
            base.CUDAPlace(0)
            if base.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        self.set_program()

    def set_program(self):
        self.program = base.Program()
        with base.program_guard(self.program):
            input = paddle.assign(self.x)
            tensor_array = paddle.tensor.create_array(dtype='float32')
            zero = paddle.tensor.fill_constant(
                shape=[1], value=0, dtype="int64"
            )

            for i in range(self.iter_num):
                paddle.tensor.array_write(input, zero + i, tensor_array)

            self.out_var = paddle.stack(tensor_array, axis=self.axis)

    def test_case(self):
        self.assertTrue(self.out_var.shape[self.axis] == -1)
        exe = base.Executor(self.place)
        res = exe.run(self.program, fetch_list=self.out_var)
        np.testing.assert_array_equal(
            res[0], np.stack([self.x] * self.iter_num, axis=self.axis)
        )


class API_test(unittest.TestCase):
    def test_out(self):
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data('data1', shape=[1, 2], dtype='float64')
            data2 = paddle.static.data('data2', shape=[1, 2], dtype='float64')
            data3 = paddle.static.data('data3', shape=[1, 2], dtype='float64')
            result_stack = paddle.stack([data1, data2, data3], axis=0)
            place = base.CPUPlace()
            exe = base.Executor(place)
            input1 = np.random.random([1, 2]).astype('float64')
            input2 = np.random.random([1, 2]).astype('float64')
            input3 = np.random.random([1, 2]).astype('float64')
            (result,) = exe.run(
                feed={"data1": input1, "data2": input2, "data3": input3},
                fetch_list=[result_stack],
            )
            expected_result = np.stack([input1, input2, input3], axis=0)
            np.testing.assert_allclose(expected_result, result, rtol=1e-05)

    def test_single_tensor_error(self):
        with base.program_guard(base.Program(), base.Program()):
            x = paddle.rand([2, 3])
            self.assertRaises(TypeError, paddle.stack, x)


class API_DygraphTest(unittest.TestCase):
    def test_out(self):
        data1 = np.array([[1.0, 2.0]])
        data2 = np.array([[3.0, 4.0]])
        data3 = np.array([[5.0, 6.0]])
        with base.dygraph.guard():
            x1 = base.dygraph.to_variable(data1)
            x2 = base.dygraph.to_variable(data2)
            x3 = base.dygraph.to_variable(data3)
            result = paddle.stack([x1, x2, x3])
            result_np = result.numpy()
        expected_result = np.stack([data1, data2, data3])
        np.testing.assert_allclose(expected_result, result_np, rtol=1e-05)

        with base.dygraph.guard():
            y1 = base.dygraph.to_variable(data1)
            result = paddle.stack([y1], axis=0)
            result_np_2 = result.numpy()
        expected_result_2 = np.stack([data1], axis=0)
        np.testing.assert_allclose(expected_result_2, result_np_2, rtol=1e-05)

    def test_single_tensor_error(self):
        with base.dygraph.guard():
            x = paddle.to_tensor([1, 2, 3])
            self.assertRaises(Exception, paddle.stack, x)


class TestStackOpWithNegativeShape(unittest.TestCase):
    def test_out(self):
        main_prg, startup_prg = Program(), Program()
        with program_guard(main_prg, startup_prg):
            b = paddle.static.data(name='b', shape=[-1], dtype='int64')
            e = paddle.static.data(name='e', shape=[3], dtype='int64')
            k = paddle.stack([b, e], axis=0)
            exe = paddle.static.Executor()
            exe.run(startup_prg)
            out = exe.run(
                main_prg,
                feed={
                    'b': np.ones(
                        [
                            3,
                        ]
                    ).astype("int64"),
                    'e': np.zeros(
                        [
                            3,
                        ]
                    ).astype("int64"),
                },
                fetch_list=[k],
            )
        np.testing.assert_allclose(
            out[0], np.array([[1, 1, 1], [0, 0, 0]]), rtol=1e-05
        )


class TestStackAPI_ZeroDim(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()

        x1 = paddle.rand([])
        x2 = paddle.rand([])
        x1.stop_gradient = False
        x2.stop_gradient = False
        out = paddle.stack([x1, x2])
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [2])
        self.assertEqual(x1.grad.shape, [])
        self.assertEqual(x2.grad.shape, [])
        self.assertEqual(out.grad.shape, [2])

        paddle.enable_static()


class TestStackListOfSingleTensor(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        paddle.seed(2022)
        self.x = [paddle.randn((4, 2, 6), dtype="float32")]

    def test_list_single_tensor(self):
        expect = paddle.stack(self.x)
        paddle.base.core._set_prim_all_enabled(True)
        st_model = paddle.jit.to_static(paddle.stack)
        actual = st_model(self.x)
        np.testing.assert_allclose(expect, actual)
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
