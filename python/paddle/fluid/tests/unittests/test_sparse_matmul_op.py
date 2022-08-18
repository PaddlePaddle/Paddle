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
import numpy as np
import scipy
import scipy.sparse as sp
import unittest
import os
import re

paddle.set_default_dtype('float64')


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


class TestMatmul(unittest.TestCase):
    # x: sparse, y: dense, out: dense
    def check_result(self, x_shape, y_shape, format):
        if len(x_shape) == 3:
            mask = paddle.randint(0, 2, [x_shape[-2], x_shape[-1]])
        else:
            mask = paddle.randint(0, 2, x_shape)
        origin_x = paddle.rand(x_shape) * mask
        origin_y = paddle.rand(y_shape)

        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_y = origin_y.detach()
        dense_y.stop_gradient = False
        dense_out = paddle.matmul(dense_x, dense_y)

        if format == "coo":
            sp_x = origin_x.detach().to_sparse_coo(len(x_shape))
        else:
            sp_x = origin_x.detach().to_sparse_csr()
        sp_x.stop_gradient = False
        sp_y = origin_y.detach()
        sp_y.stop_gradient = False
        sp_out = paddle.incubate.sparse.matmul(sp_x, sp_y)

        np.testing.assert_allclose(sp_out.numpy(),
                                   dense_out.numpy(),
                                   rtol=1e-05)
        if get_cuda_version() >= 11030:
            dense_out.backward()
            sp_out.backward()
            np.testing.assert_allclose(sp_x.grad.to_dense().numpy(),
                                       (dense_x.grad * mask).numpy(),
                                       rtol=1e-05)
            np.testing.assert_allclose(sp_y.grad.numpy(),
                                       dense_y.grad.numpy(),
                                       rtol=1e-05)

    @unittest.skipIf(not paddle.is_compiled_with_cuda()
                     or get_cuda_version() < 11000, "only support cuda>=11.0")
    def test_matmul_2d(self):
        self.check_result([16, 12], [12, 10], 'coo')
        self.check_result([16, 12], [12, 10], 'csr')

    @unittest.skipIf(not paddle.is_compiled_with_cuda()
                     or get_cuda_version() < 11070, "only support cuda>=11.7")
    def test_matmul_3d(self):
        self.check_result([8, 16, 12], [8, 12, 10], 'coo')
        self.check_result([8, 16, 12], [8, 12, 10], 'csr')


class TestMaskedMatmul(unittest.TestCase):
    # x: dense, y: dense, out: sparse_`csr
    @unittest.skipIf(not paddle.is_compiled_with_cuda()
                     or get_cuda_version() < 11030,
                     "only support on cuda>=11.3")
    def test_masked_matmul_2d(self):
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

        np.testing.assert_allclose(np_out.indptr,
                                   out.crows().numpy(),
                                   rtol=1e-05)
        np.testing.assert_allclose(np_out.indices,
                                   out.cols().numpy(),
                                   rtol=1e-05)
        np.testing.assert_allclose(np_out.data,
                                   out.values().numpy(),
                                   rtol=1e-05)

        out.backward()
        np.testing.assert_allclose(out.is_sparse_csr(), True, rtol=1e-05)
        np.testing.assert_allclose(np_x_grad, x.grad.numpy(), rtol=1e-05)
        np.testing.assert_allclose(np_y_grad, y.grad.numpy(), rtol=1e-05)

    @unittest.skipIf(not paddle.is_compiled_with_cuda()
                     or get_cuda_version() < 11070,
                     "only support on cuda>=11.7")
    def test_masked_matmul_3d(self):
        paddle.set_default_dtype('float32')
        origin_x = paddle.rand([16, 16, 12])
        mask = paddle.randint(0, 2, [16, 12])
        origin_x = origin_x * mask
        origin_y = paddle.rand([16, 12, 10])

        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_y = origin_y.detach()
        dense_y.stop_gradient = False
        dense_out = paddle.matmul(dense_x, dense_y)
        dense_out.backward()

        sp_x = origin_x.detach().to_sparse_csr()
        sp_x.stop_gradient = False
        sp_y = origin_y.detach()
        sp_y.stop_gradient = False
        sp_out = paddle.incubate.sparse.matmul(sp_x, sp_y)
        sp_out.backward()

        np.testing.assert_allclose(sp_out.numpy(),
                                   dense_out.numpy(),
                                   rtol=1e-05)
        np.testing.assert_allclose(sp_x.grad.to_dense().numpy(),
                                   (dense_x.grad * mask).numpy(),
                                   rtol=1e-05)
        np.testing.assert_allclose(sp_y.grad.numpy(),
                                   dense_y.grad.numpy(),
                                   rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
