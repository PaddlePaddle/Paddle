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


def geqrf(A):
    def _geqrf(A):
        m, n = A.shape
        tau = np.zeros([n, 1], dtype=A.dtype)
        for i in range(min(n, m)):
            alpha = A[i, i]
            normx = np.linalg.norm(A[min(i + 1, m) :, i])
            beta = np.linalg.norm(A[i:, i])
            s = 1 if alpha < 0 else -1
            u1 = alpha - s * beta
            w = A[i:, i] / u1
            w[0] = 1
            A[i + 1 :, i] = w[1 : m - i + 1]
            if normx == 0:
                tau[i] = 0
            else:
                tau[i] = -s * u1 / beta
                A[i, i] = s * beta
            w = w.reshape([-1, 1])
            A[i:, i + 1 :] = A[i:, i + 1 :] - (tau[i] * w) @ (
                w.T @ A[i:, i + 1 :]
            )
        return A, tau[: min(m, n)].reshape(-1)

    if len(A.shape) == 2:
        return _geqrf(A)
    m, n = A.shape[-2:]
    org_A_shape = A.shape
    A = A.reshape((-1, A.shape[-2], A.shape[-1]))
    n_batch = A.shape[0]
    out = np.zeros([n_batch, m, n])
    taus = np.zeros([n_batch, min(m, n)])
    org_taus_shape = list(org_A_shape[:-2]) + [min(m, n)]
    for i in range(n_batch):
        out[i], t = _geqrf(A[i])
        taus[i, :] = t.reshape(-1)
    return out.reshape(org_A_shape), taus.reshape(org_taus_shape)


def ref_qr(A):
    def _ref_qr(A):
        q, _ = np.linalg.qr(A)
        return q

    if len(A.shape) == 2:
        return _ref_qr(A)
    m, n = A.shape[-2:]
    org_shape = A.shape
    A = A.reshape((-1, A.shape[-2], A.shape[-1]))
    n_batch = A.shape[0]
    out = np.zeros([n_batch, m, n])
    for i in range(n_batch):
        out[i] = _ref_qr(A[i])
    return out.reshape(org_shape)


class TestHouseholderProductAPI(unittest.TestCase):
    def setUp(self):
        self.init_input()
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def init_input(self):
        self.x = np.array(
            [
                [1, 2, 4],
                [0, 0, 5],
                [0, 3, 6],
            ],
            dtype=np.float32,
        )

    def test_static_api(self):
        m, n = self.x.shape[-2:]
        self._x = self.x.copy()
        self.geqrf_x, self.tau = geqrf(self.x)
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(
                'x', self.geqrf_x.shape, dtype=self.geqrf_x.dtype
            )
            tau = paddle.static.data(
                'tau', self.tau.shape, dtype=self.tau.dtype
            )
            out = paddle.linalg.householder_product(x, tau)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'x': self.geqrf_x, 'tau': self.tau}, fetch_list=[out]
            )
            out_ref = ref_qr(self._x)
            np.testing.assert_allclose(out_ref, res[0], atol=1e-3)

    def test_dygraph_api(self):
        m, n = self.x.shape[-2:]
        self._x = self.x.copy()
        self.geqrf_x, self.tau = geqrf(self.x)
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.geqrf_x)
        tau = paddle.to_tensor(self.tau)
        out = paddle.linalg.householder_product(x, tau)
        out_ref = ref_qr(self._x)
        np.testing.assert_allclose(out_ref, out.numpy(), atol=1e-3)
        paddle.enable_static()

    def test_error(self):
        pass


class TestHouseholderProductAPICase1(TestHouseholderProductAPI):
    def init_input(self):
        self.x = np.random.randn(4, 3).astype('float32')


class TestHouseholderProductAPICase2(TestHouseholderProductAPI):
    def init_input(self):
        self.x = np.random.randn(4, 3).astype('float64')


class TestHouseholderProductAPICase3(TestHouseholderProductAPI):
    def init_input(self):
        self.x = np.random.randn(10, 2).astype('float32')


class TestHouseholderProductAPICase4(TestHouseholderProductAPI):
    def init_input(self):
        self.x = np.random.randn(5, 4, 3).astype('float32')


class TestHouseholderProductAPICase5(TestHouseholderProductAPI):
    def init_input(self):
        self.x = np.random.randn(4, 3).astype('float32')


class TestHouseholderProductAPI_batch_error(TestHouseholderProductAPI):
    # shape "*" in x.shape:[*, m, n] and tau.shape:[*, k] must be the same, eg. * == [2, 2] in x, but * == [2, 3] in tau
    def test_error(self):
        with self.assertRaises(AssertionError):
            x = paddle.randn([2, 2, 5, 4])
            tau = paddle.randn([2, 3, 4])
            out = paddle.linalg.householder_product(x, tau)


class TestHouseholderProductAPI_dim_error(TestHouseholderProductAPI):
    # len(x.shape) must be greater(equal) than 2, len(tau.shape) must be greater(equal) than 1
    def test_error(self):
        with self.assertRaises(AssertionError):
            x = paddle.to_tensor(
                [
                    3,
                ],
                dtype=paddle.float32,
            )
            tau = paddle.to_tensor([], dtype=paddle.float32)
            out = paddle.linalg.householder_product(x, tau)


class TestHouseholderProductAPI_type_error(TestHouseholderProductAPI):
    # type of x and tau must be float32 or float64
    def test_error(self):
        with self.assertRaises(TypeError):
            x = paddle.randn([3, 2, 1], dtype=paddle.int32)
            tau = paddle.randn([3, 4], dtype=paddle.int32)
            out = paddle.linalg.householder_product(x, tau)


class TestHouseholderProductAPI_shape_dismatch_error(TestHouseholderProductAPI):
    # len(x.shape) and len(tau.shape) + 1 must be equal
    def test_error(self):
        with self.assertRaises(AssertionError):
            x = paddle.randn([3, 2, 1], dtype=paddle.float32)
            tau = paddle.randn([6, 2, 4], dtype=paddle.float32)
            out = paddle.linalg.householder_product(x, tau)


class TestHouseholderProductAPI_col_row_error(TestHouseholderProductAPI):
    # row must be bigger than col in x
    def test_error(self):
        with self.assertRaises(AssertionError):
            x = paddle.randn([3, 6], dtype=paddle.float32)
            tau = paddle.randn([6], dtype=paddle.float32)
            out = paddle.linalg.householder_product(x, tau)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
