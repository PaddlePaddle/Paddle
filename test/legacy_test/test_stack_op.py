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
        self.check_output(check_prim=True, check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(
            self.get_x_names(),
            'Y',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


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
        self.check_output(check_prim=True, check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(
            self.get_x_names(),
            'Y',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestStackAPIWithDenseTensorArray(unittest.TestCase):
    """
    Test stack api when the input(x) is a DenseTensorArray.
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

    def test_case(self):
        self.program = paddle.static.Program()
        with paddle.static.program_guard(self.program):
            input = paddle.assign(self.x)
            tensor_array = paddle.tensor.create_array(dtype='float32')
            zero = paddle.tensor.fill_constant(
                shape=[1], value=0, dtype="int64"
            )

            for i in range(self.iter_num):
                paddle.tensor.array_write(input, zero + i, tensor_array)

            self.out_var = paddle.stack(tensor_array, axis=self.axis)
        self.assertTrue(self.out_var.shape[self.axis] == -1)
        exe = base.Executor(self.place)
        res = exe.run(self.program, fetch_list=self.out_var)
        np.testing.assert_array_equal(
            res[0], np.stack([self.x] * self.iter_num, axis=self.axis)
        )


class TestTensorStackAPIWithDenseTensorArray(unittest.TestCase):
    """
    Test stack api when the input(x) is a DenseTensorArray.
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

    def test_case(self):
        self.program = paddle.static.Program()
        with paddle.static.program_guard(self.program):
            input = paddle.assign(self.x)
            tensor_array = paddle.tensor.create_array(dtype='float32')
            zero = paddle.tensor.fill_constant(
                shape=[1], value=0, dtype="int64"
            )

            for i in range(self.iter_num):
                paddle.tensor.array_write(input, zero + i, tensor_array)

            self.out_var = paddle.stack(tensor_array, axis=self.axis)
        self.assertTrue(self.out_var.shape[self.axis] == -1)
        exe = base.Executor(self.place)
        res = exe.run(self.program, fetch_list=self.out_var)
        np.testing.assert_array_equal(
            res[0], np.stack([self.x] * self.iter_num, axis=self.axis)
        )


class API_test(unittest.TestCase):

    def test_out(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
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
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.rand([2, 3])
            self.assertRaises(TypeError, paddle.stack, x)


class API_DygraphTest(unittest.TestCase):
    def test_out(self):
        data1 = np.array([[1.0, 2.0]])
        data2 = np.array([[3.0, 4.0]])
        data3 = np.array([[5.0, 6.0]])
        with base.dygraph.guard():
            x1 = paddle.to_tensor(data1)
            x2 = paddle.to_tensor(data2)
            x3 = paddle.to_tensor(data3)
            result = paddle.stack([x1, x2, x3])
            result_np = result.numpy()
        expected_result = np.stack([data1, data2, data3])
        np.testing.assert_allclose(expected_result, result_np, rtol=1e-05)

        with base.dygraph.guard():
            y1 = paddle.to_tensor(data1)
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
        main_prg, startup_prg = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(main_prg, startup_prg):
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
        self.x[0].stop_gradient = False

    def test_list_single_tensor(self):
        expect = paddle.stack(self.x)
        paddle.base.core._set_prim_all_enabled(True)
        st_model = paddle.jit.to_static(paddle.stack, full_graph=True)
        actual = st_model(self.x)
        np.testing.assert_allclose(expect, actual)
        paddle.enable_static()


class TestPrimStackGrad(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        paddle.seed(2022)
        self.x = [paddle.randn((4, 2, 6), dtype="float32") for _ in range(3)]
        for i in range(len(self.x)):
            self.x[i].stop_gradient = False

    def test_stack_double_grad(self):
        paddle.base.core.set_prim_eager_enabled(True)
        z = paddle.stack(self.x)
        z = paddle.tanh(z)
        grads_out = paddle.grad(z, self.x[1], create_graph=True)
        ggrads_out = paddle.grad(grads_out, self.x[1], create_graph=True)[0]

        zz = paddle.tanh(self.x[1])
        grads_expected = paddle.grad(zz, self.x[1], create_graph=True)
        ggrads_expected = paddle.grad(
            grads_expected, self.x[1], create_graph=False
        )[0]

        np.testing.assert_allclose(ggrads_out, ggrads_expected)
        paddle.enable_static()
        paddle.base.core.set_prim_eager_enabled(False)

    def test_stack_triple_grad(self):
        paddle.base.core.set_prim_eager_enabled(True)
        z = paddle.stack(self.x)
        z = paddle.tanh(z)
        grads_out = paddle.grad(z, self.x[1], create_graph=True)
        ggrads_out = paddle.grad(grads_out, self.x[1], create_graph=True)
        gggrads_out = paddle.grad(ggrads_out, self.x[1], create_graph=False)[0]

        zz = paddle.tanh(self.x[1])
        grads_expected = paddle.grad(zz, self.x[1], create_graph=True)
        ggrads_expected = paddle.grad(
            grads_expected, self.x[1], create_graph=True
        )
        gggrads_expected = paddle.grad(
            ggrads_expected, self.x[1], create_graph=True
        )[0]

        np.testing.assert_allclose(gggrads_out, gggrads_expected)
        paddle.enable_static()
        paddle.base.core.set_prim_eager_enabled(False)


class TestStackAPI_ZeroSizedTensor(unittest.TestCase):
    def test_dygraph_cpu(self):
        place = base.CPUPlace()
        paddle.disable_static(place)

        x1 = paddle.ones([1, 0])
        x2 = paddle.ones([1, 0])
        x1.stop_gradient = False
        x2.stop_gradient = False
        out = paddle.stack([x1, x2])
        out.retain_grads()
        out.backward()

        np.testing.assert_equal(out.shape, [2, 1, 0])
        np.testing.assert_equal(x1.grad, None)
        np.testing.assert_equal(x2.grad, None)
        np.testing.assert_equal(out, np.ones([2, 1, 0]))

        paddle.enable_static()

    def test_dygraph_gpu(self):
        if base.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            paddle.disable_static(place)

            x1 = paddle.ones([1, 0])
            x2 = paddle.ones([1, 0])
            x1.stop_gradient = False
            x2.stop_gradient = False
            out = paddle.stack([x1, x2])
            out.retain_grads()
            out.backward()

            np.testing.assert_equal(out.shape, [2, 1, 0])
            np.testing.assert_equal(x1.grad, None)
            np.testing.assert_equal(x2.grad, None)
            np.testing.assert_equal(out, np.ones([2, 1, 0]))

            paddle.enable_static()

    def test_static_cpu(self):
        paddle.enable_static()
        place = base.CPUPlace()
        exe = base.Executor(place)
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data1 = paddle.static.data('data1', shape=[0, 2], dtype='float64')
            data2 = paddle.static.data('data2', shape=[0, 2], dtype='float64')
            data3 = paddle.static.data('data3', shape=[0, 2], dtype='float64')
            result_stack = paddle.stack([data1, data2, data3], axis=0)
            input1 = np.ones([0, 2]).astype('float64')
            input2 = np.ones([0, 2]).astype('float64')
            input3 = np.ones([0, 2]).astype('float64')
            (result,) = exe.run(
                feed={"data1": input1, "data2": input2, "data3": input3},
                fetch_list=[result_stack],
            )
            expected_result = np.stack([input1, input2, input3], axis=0)
            np.testing.assert_equal(expected_result, result)

    def test_static_gpu(self):
        if base.is_compiled_with_cuda():
            paddle.enable_static()
            place = base.CUDAPlace(0)
            exe = base.Executor(place)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                data1 = paddle.static.data(
                    'data1', shape=[0, 2], dtype='float64'
                )
                data2 = paddle.static.data(
                    'data2', shape=[0, 2], dtype='float64'
                )
                data3 = paddle.static.data(
                    'data3', shape=[0, 2], dtype='float64'
                )
                result_stack = paddle.stack([data1, data2, data3], axis=0)
                input1 = np.ones([0, 2]).astype('float64')
                input2 = np.ones([0, 2]).astype('float64')
                input3 = np.ones([0, 2]).astype('float64')
                (result,) = exe.run(
                    feed={"data1": input1, "data2": input2, "data3": input3},
                    fetch_list=[result_stack],
                )
                expected_result = np.stack([input1, input2, input3], axis=0)
                np.testing.assert_equal(expected_result, result)


if __name__ == '__main__':
    unittest.main()
