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

<<<<<<< HEAD
import unittest

import numpy as np
from op_test import OpTest

import paddle
=======
from __future__ import print_function

import unittest
import numpy as np
import paddle
from op_test import OpTest
from gradient_checker import grad_check
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def valid_eigh_result(A, eigh_value, eigh_vector, uplo):
    assert A.ndim == 2 or A.ndim == 3

    if A.ndim == 2:
        valid_single_eigh_result(A, eigh_value, eigh_vector, uplo)
        return

    for batch_A, batch_w, batch_v in zip(A, eigh_value, eigh_vector):
        valid_single_eigh_result(batch_A, batch_w, batch_v, uplo)


def valid_single_eigh_result(A, eigh_value, eigh_vector, uplo):
    FP32_MAX_RELATIVE_ERR = 5e-5
    FP64_MAX_RELATIVE_ERR = 1e-14

    if A.dtype == np.single or A.dtype == np.csingle:
        rtol = FP32_MAX_RELATIVE_ERR
    else:
        rtol = FP64_MAX_RELATIVE_ERR

    M, N = A.shape

    triangular_func = np.tril if uplo == 'L' else np.triu

    if not np.iscomplexobj(A):
        # Reconstruct A by filling triangular part
        A = triangular_func(A) + triangular_func(A, -1).T
    else:
        # Reconstruct A to Hermitian matrix
        A = triangular_func(A) + np.matrix(triangular_func(A, -1)).H

    # Diagonal matrix of eigen value
    T = np.diag(eigh_value)

    # A = Q*T*Q'
    residual = A - (eigh_vector @ T @ np.linalg.inv(eigh_vector))

    # ||A - Q*T*Q'|| / (N*||A||) < rtol
    np.testing.assert_array_less(
<<<<<<< HEAD
        np.linalg.norm(residual, np.inf) / (N * np.linalg.norm(A, np.inf)), rtol
    )
=======
        np.linalg.norm(residual, np.inf) / (N * np.linalg.norm(A, np.inf)),
        rtol)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # ||I - Q*Q'|| / M < rtol
    residual = np.eye(M) - eigh_vector @ np.linalg.inv(eigh_vector)
    np.testing.assert_array_less(np.linalg.norm(residual, np.inf) / M, rtol)


class TestEighOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        paddle.enable_static()
        self.op_type = "eigh"
        self.init_input()
        self.init_config()
        np.random.seed(123)
        out_w, out_v = np.linalg.eigh(self.x_np, self.UPLO)
        self.inputs = {"X": self.x_np}
        self.attrs = {"UPLO": self.UPLO}
        self.outputs = {'Eigenvalues': out_w, "Eigenvectors": out_v}

    def init_config(self):
        self.UPLO = 'L'

    def init_input(self):
        self.x_shape = (10, 10)
        self.x_type = np.float64
        self.x_np = np.random.random(self.x_shape).astype(self.x_type)

    def test_check_output(self):
        self.check_output(no_check_set=['Eigenvectors'])

    def test_grad(self):
        self.check_grad(["X"], ["Eigenvalues"])


class TestEighUPLOCase(TestEighOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_config(self):
        self.UPLO = 'U'


class TestEighGPUCase(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.x_shape = [32, 32]
        self.dtype = "float32"
        self.UPLO = "L"
        np.random.seed(123)
        self.x_np = np.random.random(self.x_shape).astype(self.dtype)

    def test_check_output_gpu(self):
        if paddle.is_compiled_with_cuda():
            paddle.disable_static(place=paddle.CUDAPlace(0))
            input_real_data = paddle.to_tensor(self.x_np)
            actual_w, actual_v = paddle.linalg.eigh(input_real_data, self.UPLO)
<<<<<<< HEAD
            valid_eigh_result(
                self.x_np, actual_w.numpy(), actual_v.numpy(), self.UPLO
            )


class TestEighAPI(unittest.TestCase):
=======
            valid_eigh_result(self.x_np, actual_w.numpy(), actual_v.numpy(),
                              self.UPLO)


class TestEighAPI(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.init_input_data()
        self.UPLO = 'L'
        self.rtol = 1e-5  # for test_eigh_grad
        self.atol = 1e-5  # for test_eigh_grad
<<<<<<< HEAD
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
=======
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        np.random.seed(123)

    def init_input_shape(self):
        self.x_shape = [5, 5]

    def init_input_data(self):
        self.init_input_shape()
        self.dtype = "float32"
        self.real_data = np.random.random(self.x_shape).astype(self.dtype)
        complex_data = np.random.random(self.x_shape).astype(
<<<<<<< HEAD
            self.dtype
        ) + 1j * np.random.random(self.x_shape).astype(self.dtype)
        self.trans_dims = list(range(len(self.x_shape) - 2)) + [
            len(self.x_shape) - 1,
            len(self.x_shape) - 2,
        ]
        # build a random conjugate matrix
        self.complex_symm = np.divide(
            complex_data + np.conj(complex_data.transpose(self.trans_dims)), 2
        )
=======
            self.dtype) + 1J * np.random.random(self.x_shape).astype(self.dtype)
        self.trans_dims = list(range(len(self.x_shape) - 2)) + [
            len(self.x_shape) - 1, len(self.x_shape) - 2
        ]
        #build a random conjugate matrix
        self.complex_symm = np.divide(
            complex_data + np.conj(complex_data.transpose(self.trans_dims)), 2)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def check_static_float_result(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
<<<<<<< HEAD
            input_x = paddle.static.data(
                'input_x', shape=self.x_shape, dtype=self.dtype
            )
            output_w, output_v = paddle.linalg.eigh(input_x)
            exe = paddle.static.Executor(self.place)
            actual_w, actual_v = exe.run(
                main_prog,
                feed={"input_x": self.real_data},
                fetch_list=[output_w, output_v],
            )
=======
            input_x = paddle.static.data('input_x',
                                         shape=self.x_shape,
                                         dtype=self.dtype)
            output_w, output_v = paddle.linalg.eigh(input_x)
            exe = paddle.static.Executor(self.place)
            actual_w, actual_v = exe.run(main_prog,
                                         feed={"input_x": self.real_data},
                                         fetch_list=[output_w, output_v])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            valid_eigh_result(self.real_data, actual_w, actual_v, self.UPLO)

    def check_static_complex_result(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            x_dtype = np.complex64 if self.dtype == "float32" else np.complex128
<<<<<<< HEAD
            input_x = paddle.static.data(
                'input_x', shape=self.x_shape, dtype=x_dtype
            )
            output_w, output_v = paddle.linalg.eigh(input_x)
            exe = paddle.static.Executor(self.place)
            actual_w, actual_v = exe.run(
                main_prog,
                feed={"input_x": self.complex_symm},
                fetch_list=[output_w, output_v],
            )
=======
            input_x = paddle.static.data('input_x',
                                         shape=self.x_shape,
                                         dtype=x_dtype)
            output_w, output_v = paddle.linalg.eigh(input_x)
            exe = paddle.static.Executor(self.place)
            actual_w, actual_v = exe.run(main_prog,
                                         feed={"input_x": self.complex_symm},
                                         fetch_list=[output_w, output_v])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            valid_eigh_result(self.complex_symm, actual_w, actual_v, self.UPLO)

    def test_in_static_mode(self):
        paddle.enable_static()
        self.check_static_float_result()
        self.check_static_complex_result()

    def test_in_dynamic_mode(self):
        paddle.disable_static()
        input_real_data = paddle.to_tensor(self.real_data)
        actual_w, actual_v = paddle.linalg.eigh(input_real_data)
<<<<<<< HEAD
        valid_eigh_result(
            self.real_data, actual_w.numpy(), actual_v.numpy(), self.UPLO
        )

        input_complex_data = paddle.to_tensor(self.complex_symm)
        actual_w, actual_v = paddle.linalg.eigh(input_complex_data)
        valid_eigh_result(
            self.complex_symm, actual_w.numpy(), actual_v.numpy(), self.UPLO
        )
=======
        valid_eigh_result(self.real_data, actual_w.numpy(), actual_v.numpy(),
                          self.UPLO)

        input_complex_data = paddle.to_tensor(self.complex_symm)
        actual_w, actual_v = paddle.linalg.eigh(input_complex_data)
        valid_eigh_result(self.complex_symm, actual_w.numpy(), actual_v.numpy(),
                          self.UPLO)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_eigh_grad(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.complex_symm, stop_gradient=False)
        w, v = paddle.linalg.eigh(x)
        (w.sum() + paddle.abs(v).sum()).backward()
<<<<<<< HEAD
        np.testing.assert_allclose(
            abs(x.grad.numpy()),
            abs(x.grad.numpy().conj().transpose(self.trans_dims)),
            rtol=self.rtol,
            atol=self.atol,
        )


class TestEighBatchAPI(TestEighAPI):
=======
        np.testing.assert_allclose(abs(x.grad.numpy()),
                                   abs(x.grad.numpy().conj().transpose(
                                       self.trans_dims)),
                                   rtol=self.rtol,
                                   atol=self.atol)


class TestEighBatchAPI(TestEighAPI):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_input_shape(self):
        self.x_shape = [2, 5, 5]


class TestEighAPIError(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_error(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
<<<<<<< HEAD
            # input maxtrix must greater than 2 dimensions
            input_x = paddle.static.data(
                name='x_1', shape=[12], dtype='float32'
            )
            self.assertRaises(ValueError, paddle.linalg.eigh, input_x)

            # input matrix must be square matrix
            input_x = paddle.static.data(
                name='x_2', shape=[12, 32], dtype='float32'
            )
            self.assertRaises(ValueError, paddle.linalg.eigh, input_x)

            # uplo must be in 'L' or 'U'
            input_x = paddle.static.data(
                name='x_3', shape=[4, 4], dtype="float32"
            )
            uplo = 'R'
            self.assertRaises(ValueError, paddle.linalg.eigh, input_x, uplo)

            # x_data cannot be integer
            input_x = paddle.static.data(
                name='x_4', shape=[4, 4], dtype="int32"
            )
=======
            #input maxtrix must greater than 2 dimensions
            input_x = paddle.static.data(name='x_1',
                                         shape=[12],
                                         dtype='float32')
            self.assertRaises(ValueError, paddle.linalg.eigh, input_x)

            #input matrix must be square matrix
            input_x = paddle.static.data(name='x_2',
                                         shape=[12, 32],
                                         dtype='float32')
            self.assertRaises(ValueError, paddle.linalg.eigh, input_x)

            #uplo must be in 'L' or 'U'
            input_x = paddle.static.data(name='x_3',
                                         shape=[4, 4],
                                         dtype="float32")
            uplo = 'R'
            self.assertRaises(ValueError, paddle.linalg.eigh, input_x, uplo)

            #x_data cannot be integer
            input_x = paddle.static.data(name='x_4',
                                         shape=[4, 4],
                                         dtype="int32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertRaises(TypeError, paddle.linalg.eigh, input_x)


if __name__ == "__main__":
    unittest.main()
