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

np.random.seed(2022)


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


if __name__ == "__main__":
    unittest.main()
