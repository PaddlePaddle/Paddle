#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core

RTOL = {'float32': 1e-7, 'float64': 1e-11}
ATOL = {'float32': 1e-7, 'float64': 1e-11}


def cholesky_inverse_numpy(M, upper=False):
    if upper:
        M = M.T @ M
    else:
        M = M @ M.T
    return np.linalg.inv(M)


class TestCholeskyInverse(unittest.TestCase):
    def setUp(self):
        self.init_dtype()
        self.generate_input()
        self.generate_output()
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def generate_input(self):
        self._shape = (3, 3)
        self._upper = False
        self._input = np.array(
            [[3.0, 0.0, 0.0], [2.0, 3.0, 0.0], [-1.0, 1.0, -2.0]]
        ).astype(self._dtype)

    def generate_output(self):
        self._output = cholesky_inverse_numpy(self._input, self._upper)

    def init_dtype(self):
        self._dtype = 'float64'

    def test_dygraph(self):
        for place in self.places:
            paddle.disable_static(place)
            x = paddle.to_tensor(self._input, dtype=self._dtype, place=place)
            out = paddle.linalg.cholesky_inverse(x, self._upper).numpy()

            np.testing.assert_allclose(
                out,
                self._output,
                atol=ATOL.get(self._dtype),
                rtol=RTOL.get(self._dtype),
            )

            # test `Tensor.xxx`
            out = x.cholesky_inverse(self._upper).numpy()
            np.testing.assert_allclose(
                out,
                self._output,
                atol=ATOL.get(self._dtype),
                rtol=RTOL.get(self._dtype),
            )

    def test_static(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))

        for place in places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name="input",
                    shape=self._shape,
                    dtype=self._dtype,
                )

                out = paddle.linalg.cholesky_inverse(x, self._upper)
                exe = paddle.static.Executor(place)

                res = exe.run(
                    feed={"input": self._input},
                    fetch_list=[out],
                )[0]

            np.testing.assert_allclose(
                res,
                self._output,
                rtol=RTOL.get(self._dtype),
                atol=ATOL.get(self._dtype),
            )

    def test_grad(self):
        for place in self.places:
            x = paddle.to_tensor(
                self._input, dtype=self._dtype, place=place, stop_gradient=False
            )
            out = paddle.linalg.cholesky_inverse(x, self._upper)
            out.backward()
            x_grad = x.grad

            self.assertEqual(list(x_grad.shape), list(x.shape))
            self.assertEqual(x_grad.dtype, x.dtype)


class TestFloat32(TestCholeskyInverse):
    def init_dtype(self):
        self._dtype = 'float32'


class TestUpperTriangularMatrix(TestCholeskyInverse):
    def generate_input(self):
        self._shape = (4, 4)
        self._upper = True
        self._input = np.array(
            [
                [-1.0, 1.0, -2.0, 3.0],
                [0.0, 2.0, 3.0, -5.0],
                [0.0, 0.0, 3.0, -4.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).astype(self._dtype)


class TestCholeskyDecompositionData(TestCholeskyInverse):
    """
    Make Cholesky decomposition data to get CholeskyInverse
    compare with paddle.linalg.inv result
    """

    def generate_input(self):
        np.random.seed(123)
        self._upper = False
        self._shape = (5, 5)
        A = np.random.rand(5, 5).astype(self._dtype)
        A = A @ A.T + np.eye(5) * 1e-3

        self._input = np.linalg.cholesky(A)
        self._cholesky = A

    def generate_output(self):
        self._output = paddle.linalg.inv(
            paddle.to_tensor(self._cholesky, dtype=self._dtype)
        ).numpy()


class TestErrorDimension(unittest.TestCase):
    def test_0d(self):
        x = paddle.to_tensor(123, dtype='float32')
        with self.assertRaises(ValueError):
            paddle.linalg.cholesky_inverse(x)

    def test_1d(self):
        x = paddle.rand((3,), dtype='float32')
        with self.assertRaises(ValueError):
            paddle.linalg.cholesky_inverse(x)

    def test_3d(self):
        x = paddle.rand((3, 4, 5), dtype='float32')
        with self.assertRaises(ValueError):
            paddle.linalg.cholesky_inverse(x)

    def test_asymmetric_matrix(self):
        x = paddle.rand((3, 4), dtype='float32')
        with self.assertRaises(ValueError):
            paddle.linalg.cholesky_inverse(x)


class TestErrorDtype(unittest.TestCase):
    def test_float16(self):
        if core.is_compiled_with_cuda():
            x = paddle.rand((3, 3), dtype='float16')
            with self.assertRaises((RuntimeError, ValueError, TypeError)):
                paddle.linalg.cholesky_inverse(x)

    def test_bfloat16(self):
        if core.is_compiled_with_cuda():
            x = paddle.rand((3, 3), dtype='bfloat16')
            with self.assertRaises((RuntimeError, ValueError, TypeError)):
                paddle.linalg.cholesky_inverse(x)


if __name__ == '__main__':
    unittest.main()
