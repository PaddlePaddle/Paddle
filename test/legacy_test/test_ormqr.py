# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


def geqrf(x):
    def _geqrf(x):
        m, n = x.shape
        tau = np.zeros([n, 1], dtype=x.dtype)
        for i in range(min(n, m)):
            alpha = x[i, i]
            normx = np.linalg.norm(x[min(i + 1, m) :, i])
            beta = np.linalg.norm(x[i:, i])
            if x.dtype in [np.complex64, np.complex128]:
                s = 1 if alpha < 0 else -1
            else:
                alphar = x[i, i].real
                s = 1 if alphar < 0 else -1
            u1 = alpha - s * beta
            w = x[i:, i] / u1
            w[0] = 1
            x[i + 1 :, i] = w[1 : m - i + 1]
            if normx == 0:
                tau[i] = 0
            else:
                tau[i] = -s * u1 / beta
                x[i, i] = s * beta
            w = w.reshape([-1, 1])
            if x.dtype in [np.complex64, np.complex128]:
                x[i:, i + 1 :] = x[i:, i + 1 :] - (tau[i] * w) @ (
                    np.conj(w).T @ x[i:, i + 1 :]
                )
            else:
                x[i:, i + 1 :] = x[i:, i + 1 :] - (tau[i] * w) @ (
                    w.T @ x[i:, i + 1 :]
                )
        return x, tau[: min(m, n)].reshape(-1)

    if len(x.shape) == 2:
        return _geqrf(x)
    m, n = x.shape[-2:]
    org_x_shape = x.shape
    x = x.reshape((-1, x.shape[-2], x.shape[-1]))
    n_batch = x.shape[0]
    out = np.zeros([n_batch, m, n], dtype=x.dtype)
    taus = np.zeros([n_batch, min(m, n)], dtype=x.dtype)
    org_taus_shape = [*org_x_shape[:-2], min(m, n)]
    for i in range(n_batch):
        out[i], t = _geqrf(x[i])
        taus[i, :] = t.reshape(-1)
    return out.reshape(org_x_shape), taus.reshape(org_taus_shape)


def ref_ormqr(input, tau, y, left=True, transpose=False):
    m, n = input.shape[-2:]

    def _ref_ormqr(input_matrix, tau_vector):
        k = tau_vector.shape[-1]
        Q = np.eye(m, dtype=input_matrix.dtype)
        for i in range(min(k, n)):
            w = input_matrix[i:, i]
            w[0] = 1
            w = w.reshape(-1, 1)
            if np.iscomplexobj(input_matrix):
                Q[:, i:] = Q[:, i:] - (
                    Q[:, i:] @ w @ np.conj(w).T * tau_vector[i]
                )
            else:
                Q[:, i:] = Q[:, i:] - (Q[:, i:] @ w @ w.T * tau_vector[i])
        return Q[:, :n]

    if input.ndim == 2:
        Q = _ref_ormqr(input, tau)
        Q = Q.T if transpose else Q
    else:
        org_input_shape = input.shape
        org_tau_shape = tau.shape

        input_reshaped = input.reshape(
            (-1, org_input_shape[-2], org_input_shape[-1])
        )
        tau_reshaped = tau.reshape((-1, org_tau_shape[-1]))

        n_batch = input_reshaped.shape[0]
        Q = np.zeros((n_batch, m, n), dtype=input.dtype)

        for i in range(n_batch):
            Q[i] = _ref_ormqr(input_reshaped[i], tau_reshaped[i])

        Q = (
            np.transpose(Q.reshape(org_input_shape), (0, 2, 1))
            if transpose
            else Q.reshape(org_input_shape)
        )
    result = np.matmul(Q, y) if left else np.matmul(y, Q)

    return result


class TestOrmqrAPI(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.init_input()

    def init_input(self):
        self.x = np.array(
            [
                [1, 2, 4],
                [0, 0, 5],
                [0, 3, 6],
            ],
            dtype=np.float32,
        )
        self.y = np.array(
            [
                [1, 2, 4],
                [0, 0, 5],
                [0, 3, 6],
            ],
            dtype=np.float32,
        )

    def test_static_api(self):
        m, n = self.x.shape[-2:]
        self.geqrf_x, self.tau = geqrf(self.x)
        self._x = self.geqrf_x.copy()
        self._tau = self.tau.copy()
        self._y = self.y.copy()
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(
                'x', self.geqrf_x.shape, dtype=self.geqrf_x.dtype
            )
            tau = paddle.static.data(
                'tau', self.tau.shape, dtype=self.tau.dtype
            )
            y = paddle.static.data('y', self.y.shape, dtype=self.y.dtype)
            out = paddle.linalg.ormqr(x, tau, y)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'x': self.geqrf_x, 'tau': self.tau, 'y': self.y},
                fetch_list=[out],
            )
            out_ref = ref_ormqr(self._x, self._tau, self._y)
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-3, atol=1e-3)

    def test_dygraph_api(self):
        m, n = self.x.shape[-2:]
        self.geqrf_x, self.tau = geqrf(self.x)
        self._x = self.geqrf_x.copy()
        self._tau = self.tau.copy()
        self._y = self.y.copy()
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.geqrf_x)
        tau = paddle.to_tensor(self.tau)
        y = paddle.to_tensor(self.y)
        out = paddle.linalg.ormqr(x, tau, y)
        out_ref = ref_ormqr(self._x, self._tau, self._y)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-3, atol=1e-3)
        paddle.enable_static()

    def test_error(self):
        pass


class TestOrmqrAPICase1(TestOrmqrAPI):
    def init_input(self):
        self.x = np.random.randn(4, 3).astype('float32')
        self.y = np.random.randn(3, 4).astype('float32')


class TestOrmqrAPICase2(TestOrmqrAPI):
    def init_input(self):
        self.x = np.random.randn(4, 3).astype('float64')
        self.y = np.random.randn(3, 4).astype('float64')


class TestOrmqrAPICase3(TestOrmqrAPI):
    def init_input(self):
        self.x = np.random.randn(5, 4, 3).astype('float32')
        self.y = np.random.randn(5, 3, 4).astype('float32')


# complex dtype
class TestOrmqrAPICase4(TestOrmqrAPI):
    def init_input(self):
        self.x = np.random.randn(4, 3).astype('complex64')
        self.y = np.random.randn(3, 4).astype('complex64')


class TestOrmqrAPICase5(TestOrmqrAPI):
    def init_input(self):
        self.x = np.random.randn(4, 3).astype('complex128')
        self.y = np.random.randn(3, 4).astype('complex128')


class TestOrmqrAPICase6(TestOrmqrAPI):
    def init_input(self):
        if paddle.is_compiled_with_cuda():
            self.x = np.random.randn(4, 3).astype('float16')
            self.y = np.random.randn(3, 4).astype('float16')
        else:
            self.x = np.random.randn(4, 3).astype('float32')
            self.y = np.random.randn(3, 4).astype('float32')


class TestOrmqrAPI_type_error(TestOrmqrAPI):
    def test_error(self):
        with self.assertRaises(AssertionError):
            x = paddle.randn([4, 3], dtype='float64')
            tau = paddle.randn([3], dtype='float32')
            y = paddle.randn([3, 4], dtype='float64')
            out = paddle.linalg.ormqr(x, tau, y)


class TestOrmqrAPI_shape_error(TestOrmqrAPI):
    def test_error(self):
        with self.assertRaises(AssertionError):
            x = paddle.randn([3], dtype='float32')
            tau = paddle.randn([], dtype='float32')
            y = paddle.randn([3], dtype='float32')
            out = paddle.linalg.ormqr(x, tau, y)


class TestOrmqrAPI_dim_error(TestOrmqrAPI):
    def test_error(self):
        with self.assertRaises(AssertionError):
            x = paddle.randn([3, 4], dtype='float32')
            tau = paddle.randn([3, 4], dtype='float32')
            y = paddle.randn([4, 3], dtype='float32')
            out = paddle.linalg.ormqr(x, tau, y)


class TestOrmqrAPI_householder_error(TestOrmqrAPI):
    def test_error(self):
        with self.assertRaises(AssertionError):
            x = paddle.randn([4, 3], dtype='float32')
            tau = paddle.randn([4], dtype='float32')
            y = paddle.randn([3, 4], dtype='float32')
            out = paddle.linalg.ormqr(x, tau, y)


class TestOrmqrAPI_y_error(TestOrmqrAPI):
    def test_error(self):
        with self.assertRaises(AssertionError):
            x = paddle.randn([4, 3], dtype='float32')
            tau = paddle.randn([3], dtype='float32')
            y = paddle.randn([3, 4], dtype='float32')
            out = paddle.linalg.ormqr(x, tau, y, left=True, transpose=True)

        with self.assertRaises(AssertionError):
            x = paddle.randn([4, 3], dtype='float32')
            tau = paddle.randn([3], dtype='float32')
            y = paddle.randn([4, 3], dtype='float32')
            out = paddle.linalg.ormqr(x, tau, y, left=True, transpose=False)

        with self.assertRaises(AssertionError):
            x = paddle.randn([4, 3], dtype='float32')
            tau = paddle.randn([3], dtype='float32')
            y = paddle.randn([3, 4], dtype='float32')
            out = paddle.linalg.ormqr(x, tau, y, left=False, transpose=False)

        with self.assertRaises(AssertionError):
            x = paddle.randn([4, 3], dtype='float32')
            tau = paddle.randn([3], dtype='float32')
            y = paddle.randn([4, 3], dtype='float32')
            out = paddle.linalg.ormqr(x, tau, y, left=False, transpose=True)


class TestOrmqrAPI_batch_error(TestOrmqrAPI):
    def test_error(self):
        with self.assertRaises(AssertionError):
            x = paddle.randn([5, 3, 4], dtype='float32')
            tau = paddle.randn([5, 4], dtype='float32')
            y = paddle.randn([4, 4, 3], dtype='float32')
            out = paddle.linalg.ormqr(x, tau, y)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
