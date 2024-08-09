# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import scipy.sparse as sp

import paddle
import paddle.nn.functional as F

np.random.seed(2022)
devices = ['cpu']
if paddle.device.get_device() != "cpu":
    devices.append(paddle.device.get_device())


class TestCsrSoftmax(unittest.TestCase):
    def test_softmax2d(self):
        mask = np.random.rand(16, 128) < 0.5
        np_x = np.random.rand(16, 128) * mask
        np_csr = sp.csr_matrix(np_x)

        row_number = np_csr.shape[0]
        np_out = np.array([])
        for i in range(row_number):
            start = np_csr.indptr[i]
            end = np_csr.indptr[i + 1]
            if start == end:
                continue
            x = np_csr.data[start:end]
            x_max = np.max(x, keepdims=True)
            x_exp = np.exp(x - x_max)
            x_exp_sum = np.sum(x_exp, keepdims=True)
            np_out = np.concatenate([np_out, x_exp / x_exp_sum])

        csr = paddle.to_tensor(np_x, stop_gradient=False).to_sparse_csr()
        m = paddle.sparse.nn.Softmax()
        out = m(csr)
        np.testing.assert_allclose(
            out.crows().numpy(), np_csr.indptr, rtol=1e-05
        )
        np.testing.assert_allclose(
            out.cols().numpy(), np_csr.indices, rtol=1e-05
        )
        np.testing.assert_allclose(out.values().numpy(), np_out, rtol=1e-05)

        # dx = (dout - sum(dout * out)) * out, dout=rand_x
        out.backward(csr.detach())
        dx = np.array([])
        for i in range(row_number):
            start = np_csr.indptr[i]
            end = np_csr.indptr[i + 1]
            if start == end:
                continue
            out = np_out[start:end]
            dout = np_csr.data[start:end]
            sum = np.sum(dout * out, keepdims=True)
            dx = np.concatenate([dx, (dout - sum) * out])

        np.testing.assert_allclose(
            csr.grad.crows().numpy(), np_csr.indptr, rtol=1e-05
        )
        np.testing.assert_allclose(
            csr.grad.cols().numpy(), np_csr.indices, rtol=1e-05
        )
        np.testing.assert_allclose(csr.grad.values().numpy(), dx, rtol=1e-05)

    def test_softmax3d(self):
        batchNum = 16
        mask = np.random.rand(batchNum, 16, 128) < 0.5
        np_x = np.random.rand(batchNum, 16, 128) * mask

        np_out_list = []
        np_out = np.array([])
        for i in range(batchNum):
            np_csr = sp.csr_matrix(np_x[i, :, :])
            row_number = np_csr.shape[0]
            for j in range(
                row_number,
            ):
                start = np_csr.indptr[j]
                end = np_csr.indptr[j + 1]
                if start == end:
                    continue
                x = np_csr.data[start:end]
                x_max = np.max(x, keepdims=True)
                x_exp = np.exp(x - x_max)
                x_exp_sum = np.sum(x_exp, keepdims=True)
                np_out_list.append(x_exp / x_exp_sum)
                np_out = np.concatenate([np_out, x_exp / x_exp_sum])

        csr = paddle.to_tensor(np_x, stop_gradient=False).to_sparse_csr()
        m = paddle.sparse.nn.Softmax()
        out = m(csr)
        np.testing.assert_allclose(out.values().numpy(), np_out, rtol=1e-05)

        # dx = (dout - sum(dout * out)) * out, dout=rand_x
        out.backward(csr.detach())
        dx = np.array([])
        batch_offset = 0
        for i in range(batchNum):
            np_csr = sp.csr_matrix(np_x[i, :, :])
            row_number = np_csr.shape[0]
            for j in range(row_number):
                start = np_csr.indptr[j]
                end = np_csr.indptr[j + 1]
                if start == end:
                    continue
                dout = np_csr.data[start:end]
                out = np_out[batch_offset + start : batch_offset + end]
                sum = np.sum(dout * out, keepdims=True)
                dx = np.concatenate([dx, (dout - sum) * out])

            batch_offset += np_csr.nnz

        np.testing.assert_allclose(csr.grad.values().numpy(), dx, rtol=1e-05)


class TestCooSoftmax(unittest.TestCase):
    def sparse_softmax(self, sparse, dense_shape, sparse_dim, dim):
        """
        sparse softmax algorithm in Python.
        """
        inf = float('inf')
        indices = sparse.indices()
        values = sparse.values()
        size = sparse.shape
        dense_size = tuple(size[sparse_dim:])
        dense_dim = len(dense_size)

        if dim < sparse_dim:
            nnz = sparse.nnz()

            # compute pool indices
            strides = np.ones((sparse_dim, 1))
            for i in reversed(range(sparse_dim - 1)):
                strides[i, 0] = strides[i + 1, 0] * size[i + 1]
            strides[dim, 0] = 0
            strides = paddle.to_tensor(strides, dtype=indices.dtype)

            pool = paddle.sum((indices * strides), axis=0).numpy()
            i2p = {}
            for i in range(nnz):
                c = int(pool[i])
                if c not in i2p:
                    i2p[c] = len(i2p)
                pool[i] = i2p[c]

            mx = paddle.empty((pool.max() + 1, *dense_size)).numpy()
            mx[:] = -inf
            np_values = values.numpy()
            for n in range(nnz):
                p = pool[n]
                mx[p] = np.where(mx[p] > np_values[n], mx[p], np_values[n])

            # apply exp to (v - mx) and sum the results
            exp_values = paddle.empty_like(values).numpy()
            exp_sums = np.zeros_like(mx)
            for n in range(nnz):
                p = pool[n]
                v = exp_values[n] = np.exp(np_values[n] - mx[p])
                exp_sums[p] = exp_sums[p] + v

            # normalize with the sum of exponents
            for n in range(nnz):
                p = pool[n]
                exp_values[n] = exp_values[n] / exp_sums[p]
            return paddle.sparse.sparse_coo_tensor(
                indices, exp_values, dense_shape
            )

        elif dim < sparse_dim + dense_dim:
            return paddle.sparse.sparse_coo_tensor(
                indices, F.softmax(values, dim - sparse_dim + 1), size
            )
        else:
            print(
                f"`dim(={dim})` must be smaller than `sparse_dim(={sparse_dim}) + dense_dim(={dense_dim})`"
            )

    def check_run(self, dense_shape):
        mask = np.random.rand(*dense_shape) < 0.5
        np_x = np.random.rand(*dense_shape) * mask
        for device in devices:
            paddle.device.set_device(device)
            for sparse_dim in range(1, len(dense_shape)):
                coo = (
                    paddle.to_tensor(np_x, stop_gradient=False)
                    .detach()
                    .to_sparse_coo(sparse_dim)
                )

                size = coo.shape
                dense_size = tuple(size[sparse_dim:])
                dense_dim = len(dense_size)

                for axis in range(sparse_dim + dense_dim):
                    coo = (
                        paddle.to_tensor(np_x, stop_gradient=False)
                        .detach()
                        .to_sparse_coo(sparse_dim)
                    )
                    coo.stop_gradient = False

                    py_out = self.sparse_softmax(
                        coo, dense_shape, sparse_dim, axis
                    )
                    m = paddle.sparse.nn.Softmax(axis=axis)
                    out = m(coo)

                    np.testing.assert_allclose(
                        py_out.indices().numpy(),
                        out.indices().numpy(),
                        rtol=1e-05,
                    )
                    np.testing.assert_allclose(
                        py_out.values().numpy(),
                        out.values().numpy(),
                        rtol=1e-05,
                    )

                    out.backward(coo.detach())
                    dense_tensor = paddle.to_tensor(np_x, stop_gradient=False)
                    model_dense = paddle.nn.Softmax(axis=axis)
                    dense_out = model_dense(dense_tensor)
                    dense_out.backward(dense_tensor.detach())
                    dg_npy = dense_tensor.grad.numpy()
                    np.testing.assert_allclose(
                        coo.grad.to_dense().numpy(), dg_npy, rtol=1e-05
                    )

    def test_softmax2d(self):
        self.check_run((16, 128))

    def test_softmax3d(self):
        self.check_run((16, 16, 128))

    def test_softmax2d_static(self):
        for device in devices:
            paddle.device.set_device(device)
            np_x = np.array([[11, 0, 0, 14, 15], [0, 22, 0, 24, 0]]).astype(
                'float32'
            )
            coo = (
                paddle.to_tensor(np_x, stop_gradient=False)
                .detach()
                .to_sparse_coo(2)
            )
            m = paddle.sparse.nn.Softmax()
            dy_out = m(coo)
            dy_out_dense = dy_out.to_dense().numpy()

            paddle.enable_static()
            indices = paddle.static.data(
                name='indices', shape=[2, 5], dtype='int32'
            )
            values = paddle.static.data(
                name='values', shape=[5, 1], dtype='float32'
            )

            dense_shape = [2, 5]
            sp_x = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
            sparse_softmax = paddle.sparse.nn.Softmax()
            sp_y = sparse_softmax(sp_x)
            out = sp_y.to_dense()

            exe = paddle.static.Executor()
            indices_data = [[0, 0, 0, 1, 1], [0, 3, 4, 1, 3]]
            values_data = np.array([11, 14, 15, 22, 24]).astype('float32')

            fetch = exe.run(
                feed={'indices': indices_data, 'values': values_data},
                fetch_list=[out],
                return_numpy=True,
            )

            np.testing.assert_allclose(dy_out_dense, fetch[0], rtol=1e-5)
            paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
