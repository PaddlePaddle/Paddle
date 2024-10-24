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

import os
import re
import unittest

import numpy as np

import paddle
from paddle.base.framework import in_pir_mode

paddle.seed(100)


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
    "paddle is not compiled with CUDA and cuda version need to >= 11.0",
)
class TestCsrMv(unittest.TestCase):
    # x: csr-matrix, y: dense-vec, out: dense-vec
    def test_mv(self):
        paddle.set_default_dtype('float64')
        origin_x = paddle.rand([64, 32])
        mask = paddle.randint(0, 2, [64, 32])
        origin_x = origin_x * mask.astype('float64')
        origin_vec = paddle.rand([32])

        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_vec = origin_vec.detach()
        dense_vec.stop_gradient = False
        dense_out = paddle.mv(dense_x, dense_vec)
        dense_out.backward()

        sp_x = origin_x.detach().to_sparse_csr()
        sp_x.stop_gradient = False
        sp_vec = origin_vec.detach()
        sp_vec.stop_gradient = False
        sp_out = paddle.sparse.mv(sp_x, sp_vec)
        sp_out.backward()

        np.testing.assert_allclose(
            sp_out.numpy(), dense_out.numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            sp_x.grad.to_dense().numpy(),
            (dense_x.grad * mask.astype('float64')).numpy(),
            rtol=1e-05,
        )
        np.testing.assert_allclose(
            sp_vec.grad.numpy(), dense_vec.grad.numpy(), rtol=1e-05
        )


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
    "paddle is not compiled with CUDA and cuda version need to >= 11.0",
)
class TestCooMv(unittest.TestCase):
    # x: csr-matrix, y: dense-vec, out: dense-vec
    def test_mv(self):
        paddle.set_default_dtype('float64')
        origin_x = paddle.rand([64, 32])
        mask = paddle.randint(0, 2, [64, 32])
        origin_x = origin_x * mask.astype('float64')
        origin_vec = paddle.rand([32])

        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_vec = origin_vec.detach()
        dense_vec.stop_gradient = False
        dense_out = paddle.mv(dense_x, dense_vec)
        dense_out.backward()

        sp_x = origin_x.detach().to_sparse_coo(sparse_dim=2)
        sp_x.stop_gradient = False
        sp_vec = origin_vec.detach()
        sp_vec.stop_gradient = False
        sp_out = paddle.sparse.mv(sp_x, sp_vec)
        sp_out.backward()

        np.testing.assert_allclose(
            sp_out.numpy(), dense_out.numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            sp_x.grad.to_dense().numpy(),
            (dense_x.grad * mask.astype('float64')).numpy(),
            rtol=1e-05,
        )
        np.testing.assert_allclose(
            sp_vec.grad.numpy(), dense_vec.grad.numpy(), rtol=1e-05
        )


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
    "paddle is not compiled with CUDA and cuda version need to >= 11.0",
)
class TestCooMvStatic(unittest.TestCase):
    # x: csr-matrix, y: dense-vec, out: dense-vec
    def test_mv(self):
        if in_pir_mode():
            paddle.set_default_dtype('float64')
            origin_x = paddle.rand([64, 32])
            mask = paddle.randint(0, 2, [64, 32])
            origin_x = origin_x * mask.astype('float64')
            origin_vec = paddle.rand([32])

            dense_x = origin_x.detach()

            dense_vec = origin_vec.detach()

            dense_out = paddle.mv(dense_x, dense_vec)
            indices_data, values_data = (
                origin_x.detach().to_sparse_coo(sparse_dim=2).indices,
                origin_x.detach().to_sparse_coo(sparse_dim=2).values,
            )
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                indices = paddle.static.data(
                    name='indices',
                    shape=indices_data.shape,
                    dtype=indices_data.dtype,
                )
                values = paddle.static.data(
                    name='values',
                    shape=values_data.shape,
                    dtype=values_data.dtype,
                )
                sp_x = paddle.sparse.sparse_coo_tensor(
                    indices,
                    values,
                    shape=origin_x.shape,
                    dtype=origin_x.dtype,
                )
                sp_vec = paddle.static.data(
                    name='vec',
                    shape=origin_vec.shape,
                    dtype=origin_vec.dtype,
                )
                sp_out = paddle.sparse.mv(sp_x, sp_vec)
                exe = paddle.static.Executor()
                fetch = exe.run(
                    feed={
                        'indices': indices_data.numpy(),
                        'values': values_data.numpy(),
                        'vec': origin_vec.detach().numpy(),
                    },
                    fetch_list=[sp_out],
                    return_numpy=False,
                )
                sp_out = fetch[0]
                np.testing.assert_allclose(
                    sp_out.numpy(), dense_out.numpy(), rtol=1e-05
                )
                paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
