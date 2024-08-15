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


class TestAddmm(unittest.TestCase):
    # input: dense, x: sparse, y: dense, out: dense
    def check_result(self, input_shape, x_shape, y_shape, format):
        if len(x_shape) == 3:
            mask = paddle.randint(0, 2, [x_shape[-2], x_shape[-1]])
        else:
            mask = paddle.randint(0, 2, x_shape)

        origin_input = paddle.rand(input_shape)
        origin_x = paddle.rand(x_shape) * mask.astype(
            paddle.get_default_dtype()
        )
        origin_y = paddle.rand(y_shape)

        dense_input = origin_input.detach()
        dense_input.stop_gradient = False
        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_y = origin_y.detach()
        dense_y.stop_gradient = False
        dense_out = 2.0 * paddle.matmul(dense_x, dense_y) + 3.0 * dense_input

        sp_input = dense_input.detach()
        sp_input.stop_gradient = False
        if format == "coo":
            sp_x = origin_x.detach().to_sparse_coo(len(x_shape))
        else:
            sp_x = origin_x.detach().to_sparse_csr()
        sp_x.stop_gradient = False
        sp_y = origin_y.detach()
        sp_y.stop_gradient = False
        sp_out = paddle.sparse.addmm(sp_input, sp_x, sp_y, 3.0, 2.0)

        np.testing.assert_allclose(
            sp_out.numpy(), dense_out.numpy(), rtol=1e-05
        )
        if get_cuda_version() >= 11030:
            dense_out.backward()
            sp_out.backward()
            np.testing.assert_allclose(
                sp_input.grad.numpy(), dense_input.grad.numpy(), rtol=1e-05
            )
            np.testing.assert_allclose(
                sp_x.grad.to_dense().numpy(),
                (
                    dense_x.grad * mask.astype(paddle.get_default_dtype())
                ).numpy(),
                rtol=1e-05,
            )
            np.testing.assert_allclose(
                sp_y.grad.numpy(), dense_y.grad.numpy(), rtol=1e-05
            )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
        "only support cuda>=11.0",
    )
    def test_addmm_2d(self):
        self.check_result([16, 10], [16, 12], [12, 10], 'coo')
        self.check_result([16, 10], [16, 12], [12, 10], 'csr')

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11080,
        "only support cuda>=11.8",
    )
    def test_addmm_3d(self):
        self.check_result([8, 16, 10], [8, 16, 12], [8, 12, 10], 'coo')
        self.check_result([8, 16, 10], [8, 16, 12], [8, 12, 10], 'csr')


class TestAddmmStatic(unittest.TestCase):

    def check_result(self, input_shape, x_shape, y_shape):
        '''Only support sparse_coo_tensor in static graph'''
        if len(x_shape) == 3:
            mask = paddle.randint(0, 2, [x_shape[-2], x_shape[-1]])
        else:
            mask = paddle.randint(0, 2, x_shape)

        origin_input = paddle.rand(input_shape)
        origin_x = paddle.rand(x_shape) * mask.astype(
            paddle.get_default_dtype()
        )
        origin_y = paddle.rand(y_shape)

        dense_input = origin_input.detach()
        dense_x = origin_x.detach()
        dense_y = origin_y.detach()
        dense_out = 2.0 * paddle.matmul(dense_x, dense_y) + 3.0 * dense_input

        indices_data, values_data = (
            origin_x.detach().to_sparse_coo(sparse_dim=len(x_shape)).indices(),
            origin_x.detach().to_sparse_coo(sparse_dim=len(x_shape)).values(),
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
                shape=dense_x.shape,
                dtype=dense_x.dtype,
            )
            sp_y = paddle.static.data(
                name='sp_y',
                shape=dense_y.shape,
                dtype=dense_y.dtype,
            )
            sp_input = paddle.static.data(
                name='sp_input',
                shape=dense_input.shape,
                dtype=dense_input.dtype,
            )
            sp_out = paddle.sparse.addmm(sp_input, sp_x, sp_y, 3.0, 2.0)
            sp_dense_out = sp_out.to_dense()

            sparse_exe = paddle.static.Executor()
            sparse_fetch = sparse_exe.run(
                feed={
                    'indices': indices_data.numpy(),
                    "values": values_data.numpy(),
                    'sp_y': origin_y.numpy(),
                    'sp_input': origin_input.numpy(),
                },
                fetch_list=[sp_dense_out],
                return_numpy=True,
            )

            np.testing.assert_allclose(
                dense_out.numpy(), sparse_fetch[0], rtol=1e-5
            )
            paddle.disable_static()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
        "only support cuda>=11.0",
    )
    def test_addmm_2d(self):
        if in_pir_mode():
            self.check_result([16, 10], [16, 12], [12, 10])

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11080,
        "only support cuda>=11.8",
    )
    def test_addmm_3d(self):
        if in_pir_mode():
            self.check_result([8, 16, 10], [8, 16, 12], [8, 12, 10])


if __name__ == "__main__":
    unittest.main()
