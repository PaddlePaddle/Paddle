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

import paddle
from paddle.fluid.framework import _test_eager_guard

import numpy as np
import scipy
import scipy.sparse as sp
import unittest
import os
import re

np.random.seed(2022)


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
    "paddle is not compiled with CUDA and cuda version need to >= 11.0")
class TestCsrDenseMatmul2D(unittest.TestCase):
    # x: csr, y: dense, out: dense
    def test_matmul(self):
        with _test_eager_guard():
            mask = np.random.rand(10, 12) < 0.2
            np_x = np.random.rand(10, 12) * mask

            np_csr = sp.csr_matrix(np_x)
            np_dense = np.random.rand(12, 6)
            np_out = np_csr @ np_dense

            np_out_grad = np.ones([10, 6])

            # dx(csr) = dout(dense) * y'(dense) * mask
            np_csr_grad = sp.csr_matrix(
                np.matmul(np_out_grad, np_dense.transpose(1, 0)) * mask)
            # dy(dense) = x'(csr) * dout(dense)
            np_dense_grad = np_csr.transpose() @ np_out_grad

            csr = paddle.to_tensor(np_x, stop_gradient=False).to_sparse_csr()
            dense = paddle.to_tensor(np_dense, stop_gradient=False)
            out = paddle.incubate.sparse.matmul(csr, dense)

            self.assertTrue(np.allclose(np_out, out.numpy()))

            if get_cuda_version() >= 11030:
                out.backward()
                self.assertTrue(
                    np.allclose(np_csr_grad.indptr,
                                csr.grad.crows().numpy()))
                self.assertTrue(
                    np.allclose(np_csr_grad.indices,
                                csr.grad.cols().numpy()))
                self.assertTrue(
                    np.allclose(np_csr_grad.data,
                                csr.grad.values().numpy()))

                self.assertTrue(np.allclose(np_dense_grad, dense.grad.numpy()))


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or get_cuda_version() < 11030,
    "paddle is not compiled with CUDA and cuda version need to >= 11.3")
class TestCsrMaskedMatmul2D(unittest.TestCase):
    # x: dense, y: dense, out: csr
    def test_matmul(self):
        with _test_eager_guard():
            np_mask = np.random.rand(10, 6) < 0.2

            np_x = np.random.rand(10, 12)
            np_y = np.random.rand(12, 6)
            np_out = sp.csr_matrix(np.matmul(np_x, np_y) * np_mask)

            np_out_grad = sp.csr_matrix(np.ones([10, 6]) * np_mask)
            # dx(dense) = dout(csr) * y'(dense)
            np_x_grad = np_out_grad @ np_y.transpose(1, 0)
            # dy(dense) = x'(dense) * dout(csr) -> dy'(dense) = dout'(csr) * x(dense)
            np_y_grad = (np_out_grad.transpose() @ np_x).transpose(1, 0)

            x = paddle.to_tensor(np_x, stop_gradient=False)
            y = paddle.to_tensor(np_y, stop_gradient=False)
            mask = paddle.to_tensor(np.ones([10, 6]) * np_mask).to_sparse_csr()
            out = paddle.incubate.sparse.masked_matmul(x, y, mask)

            self.assertTrue(np.allclose(np_out.indptr, out.crows().numpy()))
            self.assertTrue(np.allclose(np_out.indices, out.cols().numpy()))
            self.assertTrue(np.allclose(np_out.data, out.values().numpy()))

            out.backward()
            self.assertTrue(np.allclose(out.is_sparse_csr(), True))
            self.assertTrue(np.allclose(np_x_grad, x.grad.numpy()))
            self.assertTrue(np.allclose(np_y_grad, y.grad.numpy()))


#TODO(zhouwei25): support unit test of batch 'paddle.sparse.mm/masked_mm'

if __name__ == "__main__":
    unittest.main()
