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

from __future__ import print_function

import unittest
import numpy as np
import paddle
from op_test import OpTest
from gradient_checker import grad_check


class TestEigvalshOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.op_type = "eigvalsh"
        self.init_input()
        self.init_config()
        np.random.seed(123)
        out_w, out_v = np.linalg.eigh(self.x_np, self.UPLO)
        self.inputs = {"X": self.x_np}
        self.attrs = {"UPLO": self.UPLO, "is_test": False}
        self.outputs = {'Eigenvalues': out_w, 'Eigenvectors': out_v}

    def init_config(self):
        self.UPLO = 'L'

    def init_input(self):
        self.x_shape = (10, 10)
        self.x_type = np.float64
        self.x_np = np.random.random(self.x_shape).astype(self.x_type)

    def test_check_output(self):
        # Vectors in posetive or negative is equivalent
        self.check_output(no_check_set=['Eigenvectors'])

    def test_grad(self):
        self.check_grad(["X"], ["Eigenvalues"])


class TestEigvalshUPLOCase(TestEigvalshOp):
    def init_config(self):
        self.UPLO = 'U'


class TestEigvalshGPUCase(unittest.TestCase):
    def setUp(self):
        self.x_shape = [32, 32]
        self.dtype = "float32"
        np.random.seed(123)
        self.x_np = np.random.random(self.x_shape).astype(self.dtype)
        if (paddle.version.cuda() >= "11.6"):
            self.rtol = 5e-6
            self.atol = 6e-5
        else:
            self.rtol = 1e-5
            self.atol = 1e-5

    def test_check_output_gpu(self):
        if paddle.is_compiled_with_cuda():
            paddle.disable_static(place=paddle.CUDAPlace(0))
            input_real_data = paddle.to_tensor(self.x_np)
            expected_w = np.linalg.eigvalsh(self.x_np)
            actual_w = paddle.linalg.eigvalsh(input_real_data)
            np.testing.assert_allclose(
                actual_w, expected_w, rtol=self.rtol, atol=self.atol)


class TestEigvalshAPI(unittest.TestCase):
    def setUp(self):
        self.x_shape = [5, 5]
        self.dtype = "float32"
        self.UPLO = 'L'
        if (paddle.version.cuda() >= "11.6"):
            self.rtol = 5e-6
            self.atol = 6e-5
        else:
            self.rtol = 1e-5
            self.atol = 1e-5
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()
        np.random.seed(123)
        self.init_input_data()

    def init_input_data(self):
        self.real_data = np.random.random(self.x_shape).astype(self.dtype)
        complex_data = np.random.random(self.x_shape).astype(
            self.dtype) + 1J * np.random.random(self.x_shape).astype(self.dtype)
        self.trans_dims = list(range(len(self.x_shape) - 2)) + [
            len(self.x_shape) - 1, len(self.x_shape) - 2
        ]
        self.complex_symm = np.divide(
            complex_data + np.conj(complex_data.transpose(self.trans_dims)), 2)

    def compare_result(self, actual_w, expected_w):
        np.testing.assert_allclose(
            actual_w, expected_w, rtol=self.rtol, atol=self.atol)

    def check_static_float_result(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            input_x = paddle.static.data(
                'input_x', shape=self.x_shape, dtype=self.dtype)
            output_w = paddle.linalg.eigvalsh(input_x)
            exe = paddle.static.Executor(self.place)
            expected_w = exe.run(main_prog,
                                 feed={"input_x": self.real_data},
                                 fetch_list=[output_w])

            actual_w = np.linalg.eigvalsh(self.real_data)
            self.compare_result(actual_w, expected_w[0])

    def check_static_complex_result(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            x_dtype = np.complex64 if self.dtype == "float32" else np.complex128
            input_x = paddle.static.data(
                'input_x', shape=self.x_shape, dtype=x_dtype)
            output_w = paddle.linalg.eigvalsh(input_x)
            exe = paddle.static.Executor(self.place)
            expected_w = exe.run(main_prog,
                                 feed={"input_x": self.complex_symm},
                                 fetch_list=[output_w])
            actual_w = np.linalg.eigvalsh(self.complex_symm)
            self.compare_result(actual_w, expected_w[0])

    def test_in_static_mode(self):
        paddle.enable_static()
        self.check_static_float_result()
        self.check_static_complex_result()

    def test_in_dynamic_mode(self):
        paddle.disable_static(self.place)
        input_real_data = paddle.to_tensor(self.real_data)
        expected_w = np.linalg.eigvalsh(self.real_data)
        actual_w = paddle.linalg.eigvalsh(input_real_data)
        self.compare_result(actual_w, expected_w)

        input_complex_symm = paddle.to_tensor(self.complex_symm)
        expected_w = np.linalg.eigvalsh(self.complex_symm)
        actual_w = paddle.linalg.eigvalsh(input_complex_symm)
        self.compare_result(actual_w, expected_w)

    def test_eigvalsh_grad(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.complex_symm, stop_gradient=False)
        w = paddle.linalg.eigvalsh(x)
        (w.sum()).backward()
        np.testing.assert_allclose(
            abs(x.grad.numpy()),
            abs(x.grad.numpy().conj().transpose(self.trans_dims)),
            rtol=self.rtol,
            atol=self.atol)


class TestEigvalshBatchAPI(TestEigvalshAPI):
    def init_input_shape(self):
        self.x_shape = [2, 5, 5]


class TestEigvalshAPIError(unittest.TestCase):
    def test_error(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            #input maxtrix must greater than 2 dimensions
            input_x = paddle.static.data(
                name='x_1', shape=[12], dtype='float32')
            self.assertRaises(ValueError, paddle.linalg.eigvalsh, input_x)

            #input matrix must be square matrix
            input_x = paddle.static.data(
                name='x_2', shape=[12, 32], dtype='float32')
            self.assertRaises(ValueError, paddle.linalg.eigvalsh, input_x)

            #uplo must be in 'L' or 'U'
            input_x = paddle.static.data(
                name='x_3', shape=[4, 4], dtype="float32")
            uplo = 'R'
            self.assertRaises(ValueError, paddle.linalg.eigvalsh, input_x, uplo)

            #x_data cannot be integer
            input_x = paddle.static.data(
                name='x_4', shape=[4, 4], dtype="int32")
            self.assertRaises(TypeError, paddle.linalg.eigvalsh, input_x)


if __name__ == "__main__":
    unittest.main()
