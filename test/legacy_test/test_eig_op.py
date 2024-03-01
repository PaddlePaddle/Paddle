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

import unittest

import numpy as np
from op_test import OpTest, skip_check_grad_ci

import paddle
from paddle import base
from paddle.base import core


# cast output to complex for numpy.linalg.eig
def cast_to_complex(input, output):
    if input.dtype == np.float32:
        output = output.astype(np.complex64)
    elif input.dtype == np.float64:
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
        self.python_api = paddle.linalg.eig
        self.__class__.op_type = self.op_type
        self.init_input()
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x)}
        self.outputs = {'Eigenvalues': self.out[0], 'Eigenvectors': self.out[1]}

    def init_input(self):
        self.set_dtype()
        self.set_dims()
        self.x = np.random.random(self.shape).astype(self.dtype)
        self.out = np.linalg.eig(self.x)
        self.out = (
            cast_to_complex(self.x, self.out[0]),
            cast_to_complex(self.x, self.out[1]),
        )

    # for the real input, a customized checker is needed
    def checker(self, outs):
        actual_out_w = outs[0].flatten()
        expect_out_w = self.out[0].flatten()
        actual_out_v = outs[1].flatten()
        expect_out_v = self.out[1].flatten()

        length_w = len(expect_out_w)
        act_w_real = np.sort(
            np.array([np.abs(actual_out_w[i].real) for i in range(length_w)])
        )
        act_w_imag = np.sort(
            np.array([np.abs(actual_out_w[i].imag) for i in range(length_w)])
        )
        exp_w_real = np.sort(
            np.array([np.abs(expect_out_w[i].real) for i in range(length_w)])
        )
        exp_w_imag = np.sort(
            np.array([np.abs(expect_out_w[i].imag) for i in range(length_w)])
        )

        for i in range(length_w):
            np.testing.assert_allclose(
                act_w_real[i],
                exp_w_real[i],
                rtol=1e-06,
                atol=1e-05,
                err_msg='The eigenvalues real part have diff: \nExpected '
                + str(act_w_real[i])
                + '\n'
                + 'But got: '
                + str(exp_w_real[i]),
            )
            np.testing.assert_allclose(
                act_w_imag[i],
                exp_w_imag[i],
                rtol=1e-06,
                atol=1e-05,
                err_msg='The eigenvalues image part have diff: \nExpected '
                + str(act_w_imag[i])
                + '\n'
                + 'But got: '
                + str(exp_w_imag[i]),
            )

        length_v = len(expect_out_v)
        act_v_real = np.sort(
            np.array([np.abs(actual_out_v[i].real) for i in range(length_v)])
        )
        act_v_imag = np.sort(
            np.array([np.abs(actual_out_v[i].imag) for i in range(length_v)])
        )
        exp_v_real = np.sort(
            np.array([np.abs(expect_out_v[i].real) for i in range(length_v)])
        )
        exp_v_imag = np.sort(
            np.array([np.abs(expect_out_v[i].imag) for i in range(length_v)])
        )

        for i in range(length_v):
            np.testing.assert_allclose(
                act_v_real[i],
                exp_v_real[i],
                rtol=1e-06,
                atol=1e-05,
                err_msg='The eigenvectors real part have diff: \nExpected '
                + str(act_v_real[i])
                + '\n'
                + 'But got: '
                + str(exp_v_real[i]),
            )
            np.testing.assert_allclose(
                act_v_imag[i],
                exp_v_imag[i],
                rtol=1e-06,
                atol=1e-05,
                err_msg='The eigenvectors image part have diff: \nExpected '
                + str(act_v_imag[i])
                + '\n'
                + 'But got: '
                + str(exp_v_imag[i]),
            )

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
        self.grad_x = eig_backward(
            self.out[0], self.out[1], self.grad_w, self.grad_v
        )

    def test_check_output(self):
        self.check_output_with_place_customized(
            checker=self.checker, place=core.CPUPlace(), check_pir=True
        )

    def test_check_grad(self):
        self.init_grad()
        self.check_grad(
            ['X'],
            ['Eigenvalues', 'Eigenvectors'],
            user_defined_grads=[self.grad_x],
            user_defined_grad_outputs=[self.grad_w, self.grad_v],
            check_pir=True,
        )


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
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(
                name="input", shape=[3, 3], dtype='complex'
            )
            act_val, act_vec = paddle.linalg.eig(input)

            exe = base.Executor(place)
            fetch_val, fetch_vec = exe.run(
                base.default_main_program(),
                feed={"input": input_np},
                fetch_list=[act_val, act_vec],
            )
        np.testing.assert_allclose(
            expect_val,
            fetch_val,
            rtol=1e-06,
            atol=1e-06,
            err_msg='The eigen values have diff: \nExpected '
            + str(expect_val)
            + '\n'
            + 'But got: '
            + str(fetch_val),
        )
        np.testing.assert_allclose(
            np.abs(expect_vec),
            np.abs(fetch_vec),
            rtol=1e-06,
            atol=1e-06,
            err_msg='The eigen vectors have diff: \nExpected '
            + str(np.abs(expect_vec))
            + '\n'
            + 'But got: '
            + str(np.abs(fetch_vec)),
        )


class TestEigDyGraph(unittest.TestCase):
    def test_check_output_with_place(self):
        np.random.seed(1024)
        input_np = np.random.random([3, 3]).astype('complex')
        expect_val, expect_vec = np.linalg.eig(input_np)

        paddle.set_device("cpu")
        paddle.disable_static()

        input_tensor = paddle.to_tensor(input_np)
        fetch_val, fetch_vec = paddle.linalg.eig(input_tensor)

        np.testing.assert_allclose(
            expect_val,
            fetch_val.numpy(),
            rtol=1e-06,
            atol=1e-06,
            err_msg='The eigen values have diff: \nExpected '
            + str(expect_val)
            + '\n'
            + 'But got: '
            + str(fetch_val),
        )
        np.testing.assert_allclose(
            np.abs(expect_vec),
            np.abs(fetch_vec.numpy()),
            rtol=1e-06,
            atol=1e-06,
            err_msg='The eigen vectors have diff: \nExpected '
            + str(np.abs(expect_vec))
            + '\n'
            + 'But got: '
            + str(np.abs(fetch_vec.numpy())),
        )

    def test_check_grad(self):
        test_shape = [3, 3]
        test_type = 'float64'
        paddle.set_device("cpu")

        np.random.seed(1024)
        input_np = np.random.random(test_shape).astype(test_type)
        real_w, real_v = np.linalg.eig(input_np)

        grad_w = np.ones(real_w.shape, test_type)
        grad_v = np.ones(real_v.shape, test_type)
        grad_x = eig_backward(real_w, real_v, grad_w, grad_v)

        with base.dygraph.guard():
            x = paddle.to_tensor(input_np)
            x.stop_gradient = False
            w, v = paddle.linalg.eig(x)
            (w.sum() + v.sum()).backward()

        np.testing.assert_allclose(
            np.abs(x.grad.numpy()),
            np.abs(grad_x),
            rtol=1e-05,
            atol=1e-05,
            err_msg='The grad x have diff: \nExpected '
            + str(np.abs(grad_x))
            + '\n'
            + 'But got: '
            + str(np.abs(x.grad.numpy())),
        )


class TestEigWrongDimsError(unittest.TestCase):
    def test_error(self):
        paddle.device.set_device("cpu")
        paddle.disable_static()
        a = np.random.random(3).astype('float32')
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
        self.assertRaises(RuntimeError, paddle.linalg.eig, x)


if __name__ == "__main__":
    unittest.main()
