#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
import unittest
from paddle.fluid.op import Operator
from paddle.fluid import compiler, Program, program_guard


# cast output to complex for numpy.linalg.eig
def cast_to_complex(input, output):
    if (input.dtype == np.float32):
        output = output.astype(np.complex64)
    elif (input.dtype == np.float64):
        output = output.astype(np.complex128)
    return output


# define eig backward function for a single square matrix
def eig_backward(w, v, grad_w, grad_v):
    v_tran = np.transpose(v)
    v_tran = np.conjugate(v_tran)
    w_conj = np.conjugate(w)
    w_conj_l = w_conj.reshape(1, w.size)
    w_conj_r = w_conj.reshape(w.size, 1)
    w_conj_2d = w_conj_l - w_conj_r

    vhgv = np.matmul(v_tran, grad_v)
    real_vhgv = np.real(vhgv)
    diag_real = real_vhgv.diagonal()

    diag_2d = diag_real.reshape(1, w.size)
    rhs = v * diag_2d
    mid = np.matmul(v_tran, rhs)
    result = vhgv - mid

    res = np.divide(result, w_conj_2d)
    row, col = np.diag_indices_from(res)
    res[row, col] = 1.0

    tmp = np.matmul(res, v_tran)
    dx = np.linalg.solve(v_tran, tmp)
    return dx


class TestEigOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        paddle.device.set_device("cpu")
        self.op_type = "eig"
        self.__class__.op_type = self.op_type
        self.init_input()
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(self.x)}
        self.outputs = {'Eigenvalues': self.out[0], 'Eigenvectors': self.out[1]}

    def init_input(self):
        self.set_dtype()
        self.set_dims()
        self.x = np.random.random(self.shape).astype(self.dtype)
        self.out = np.linalg.eig(self.x)
        self.out = (cast_to_complex(self.x, self.out[0]),
                    cast_to_complex(self.x, self.out[1]))

    # for the real input, a customized checker is needed
    def checker(self, outs):
        actual_out_w = outs[0].flatten()
        expect_out_w = self.out[0].flatten()
        actual_out_v = outs[1].flatten()
        expect_out_v = self.out[1].flatten()

        length_w = len(expect_out_w)
        act_w_real = np.sort(
            np.array([np.abs(actual_out_w[i].real) for i in range(length_w)]))
        act_w_imag = np.sort(
            np.array([np.abs(actual_out_w[i].imag) for i in range(length_w)]))
        exp_w_real = np.sort(
            np.array([np.abs(expect_out_w[i].real) for i in range(length_w)]))
        exp_w_imag = np.sort(
            np.array([np.abs(expect_out_w[i].imag) for i in range(length_w)]))

        for i in range(length_w):
            self.assertTrue(
                np.allclose(act_w_real[i], exp_w_real[i], 1e-6, 1e-5),
                "The eigenvalues real part have diff: \nExpected " +
                str(act_w_real[i]) + "\n" + "But got: " + str(exp_w_real[i]))
            self.assertTrue(
                np.allclose(act_w_imag[i], exp_w_imag[i], 1e-6, 1e-5),
                "The eigenvalues image part have diff: \nExpected " +
                str(act_w_imag[i]) + "\n" + "But got: " + str(exp_w_imag[i]))

        length_v = len(expect_out_v)
        act_v_real = np.sort(
            np.array([np.abs(actual_out_v[i].real) for i in range(length_v)]))
        act_v_imag = np.sort(
            np.array([np.abs(actual_out_v[i].imag) for i in range(length_v)]))
        exp_v_real = np.sort(
            np.array([np.abs(expect_out_v[i].real) for i in range(length_v)]))
        exp_v_imag = np.sort(
            np.array([np.abs(expect_out_v[i].imag) for i in range(length_v)]))

        for i in range(length_v):
            self.assertTrue(
                np.allclose(act_v_real[i], exp_v_real[i], 1e-6, 1e-5),
                "The eigenvectors real part have diff: \nExpected " +
                str(act_v_real[i]) + "\n" + "But got: " + str(exp_v_real[i]))
            self.assertTrue(
                np.allclose(act_v_imag[i], exp_v_imag[i], 1e-6, 1e-5),
                "The eigenvectors image part have diff: \nExpected " +
                str(act_v_imag[i]) + "\n" + "But got: " + str(exp_v_imag[i]))

    def set_dtype(self):
        self.dtype = np.complex64

    def set_dims(self):
        self.shape = (10, 10)

    def init_grad(self):
        # grad_w, grad_v complex dtype
        gtype = self.dtype
        if self.dtype == np.float32:
            gtype = np.complex64
        elif self.dtype == np.float64:
            gtype = np.complex128
        self.grad_w = np.ones(self.out[0].shape, gtype)
        self.grad_v = np.ones(self.out[1].shape, gtype)
        self.grad_x = eig_backward(self.out[0], self.out[1], self.grad_w,
                                   self.grad_v)

    def test_check_output(self):
        self.check_output_with_place_customized(
            checker=self.checker, place=core.CPUPlace())

    def test_check_grad(self):
        self.init_grad()
        self.check_grad(
            ['X'], ['Eigenvalues', 'Eigenvectors'],
            user_defined_grads=[self.grad_x],
            user_defined_grad_outputs=[self.grad_w, self.grad_v])


class TestComplex128(TestEigOp):
    def set_dtype(self):
        self.dtype = np.complex128


@skip_check_grad_ci(
    reason="For float dtype, numpy.linalg.eig forward outputs real or complex when input is real, therefore the grad computation may be not the same with paddle.linalg.eig"
)
class TestDouble(TestEigOp):
    def set_dtype(self):
        self.dtype = np.float64

    def test_check_grad(self):
        pass


@skip_check_grad_ci(
    reason="For float dtype, numpy.linalg.eig forward outputs real or complex when input is real, therefore the grad computation may be not the same with paddle.linalg.eig"
)
class TestEigBatchMarices(TestEigOp):
    def set_dtype(self):
        self.dtype = np.float64

    def set_dims(self):
        self.shape = (3, 10, 10)

    def test_check_grad(self):
        pass


@skip_check_grad_ci(
    reason="For float dtype, numpy.linalg.eig forward outputs real or complex when input is real, therefore the grad computation may be not the same with paddle.linalg.eig"
)
class TestFloat(TestEigOp):
    def set_dtype(self):
        self.dtype = np.float32

    def test_check_grad(self):
        pass


class TestEigStatic(TestEigOp):
    def test_check_output_with_place(self):
        paddle.enable_static()
        place = core.CPUPlace()
        input_np = np.random.random([3, 3]).astype('complex')
        expect_val, expect_vec = np.linalg.eig(input_np)
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[3, 3], dtype='complex')
            act_val, act_vec = paddle.linalg.eig(input)

            exe = fluid.Executor(place)
            fetch_val, fetch_vec = exe.run(fluid.default_main_program(),
                                           feed={"input": input_np},
                                           fetch_list=[act_val, act_vec])
        self.assertTrue(
            np.allclose(expect_val, fetch_val, 1e-6, 1e-6),
            "The eigen values have diff: \nExpected " + str(expect_val) + "\n" +
            "But got: " + str(fetch_val))
        self.assertTrue(
            np.allclose(np.abs(expect_vec), np.abs(fetch_vec), 1e-6, 1e-6),
            "The eigen vectors have diff: \nExpected " +
            str(np.abs(expect_vec)) + "\n" + "But got: " +
            str(np.abs(fetch_vec)))


class TestEigWrongDimsError(unittest.TestCase):
    def test_error(self):
        paddle.device.set_device("cpu")
        paddle.disable_static()
        a = np.random.random((3)).astype('float32')
        x = paddle.to_tensor(a)
        self.assertRaises(ValueError, paddle.linalg.eig, x)


class TestEigNotSquareError(unittest.TestCase):
    def test_error(self):
        paddle.device.set_device("cpu")
        paddle.disable_static()
        a = np.random.random((1, 2, 3)).astype('float32')
        x = paddle.to_tensor(a)
        self.assertRaises(ValueError, paddle.linalg.eig, x)


class TestEigUnsupportedDtypeError(unittest.TestCase):
    def test_error(self):
        paddle.device.set_device("cpu")
        paddle.disable_static()
        a = (np.random.random((3, 3)) * 10).astype('int64')
        x = paddle.to_tensor(a)
        self.assertRaises(ValueError, paddle.linalg.eig, x)


if __name__ == "__main__":
    unittest.main()
