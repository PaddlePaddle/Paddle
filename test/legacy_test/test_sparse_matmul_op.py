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
import scipy.sparse as sp

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


class TestMatmulSparseDense(unittest.TestCase):
    # x: sparse, y: dense, out: dense
    def check_result(self, x_shape, y_shape, format):
        if len(x_shape) == 3:
            mask = paddle.randint(0, 2, [x_shape[-2], x_shape[-1]])
        else:
            mask = paddle.randint(0, 2, x_shape)
        origin_x = paddle.rand(x_shape) * mask.astype(
            paddle.get_default_dtype()
        )
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
        sp_out = paddle.sparse.matmul(sp_x, sp_y)

        np.testing.assert_allclose(
            sp_out.numpy(), dense_out.numpy(), rtol=1e-05
        )
        if get_cuda_version() >= 11030:
            dense_out.backward()
            sp_out.backward()
            np.testing.assert_allclose(
                sp_x.grad.to_dense().numpy(),
                (dense_x.grad * mask.astype(dense_x.dtype)).numpy(),
                rtol=1e-05,
            )
            np.testing.assert_allclose(
                sp_y.grad.numpy(), dense_y.grad.numpy(), rtol=1e-05
            )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
        "only support cuda>=11.0",
    )
    def test_matmul_2d(self):
        self.check_result([16, 12], [12, 10], 'coo')
        self.check_result([16, 12], [12, 10], 'csr')

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11080,
        "only support cuda>=11.8",
    )
    def test_matmul_3d(self):
        self.check_result([8, 16, 12], [8, 12, 10], 'coo')
        self.check_result([8, 16, 12], [8, 12, 10], 'csr')


class TestMatmulSparseSparseInt64Index(unittest.TestCase):
    # x: sparse, y: sparse, out: sparse
    def check_result(self, x_shape, y_shape, format):
        origin_x = paddle.rand(x_shape)
        origin_y = paddle.rand(y_shape)

        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_y = origin_y.detach()
        dense_y.stop_gradient = False
        dense_out = paddle.matmul(dense_x, dense_y)

        if format == "coo":
            sp_x = origin_x.detach().to_sparse_coo(len(x_shape))
            sp_y = origin_y.detach().to_sparse_coo(len(y_shape))
        else:
            sp_x = origin_x.detach().to_sparse_csr()
            sp_y = origin_y.detach().to_sparse_csr()

        sp_x.stop_gradient = False
        sp_y.stop_gradient = False

        sp_out = paddle.sparse.matmul(sp_x, sp_y)

        np.testing.assert_allclose(
            sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05
        )

        dense_out.backward()
        sp_out.backward()
        np.testing.assert_allclose(
            sp_x.grad.to_dense().numpy(),
            dense_x.grad.numpy(),
            rtol=1e-05,
        )
        np.testing.assert_allclose(
            sp_y.grad.to_dense().numpy(), dense_y.grad.numpy(), rtol=1e-05
        )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
        "only support cuda>=11.0",
    )
    def test_matmul_2d(self):
        self.check_result([16, 12], [12, 10], 'coo')
        self.check_result([16, 12], [12, 10], 'csr')

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
        "only support cuda>=11.0",
    )
    def test_matmul_3d(self):
        self.check_result([8, 16, 12], [8, 12, 10], 'coo')
        self.check_result([8, 16, 12], [8, 12, 10], 'csr')


class TestMatmulSparseSparseInt32Index(unittest.TestCase):
    # x: sparse, y: sparse, out: sparse
    def check_result(self, x_shape, y_shape, format):
        origin_x = paddle.rand(x_shape)
        origin_y = paddle.rand(y_shape)

        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_y = origin_y.detach()
        dense_y.stop_gradient = False
        dense_out = paddle.matmul(dense_x, dense_y)

        if format == "coo":
            sp_x = origin_x.detach().to_sparse_coo(len(x_shape))
            # cast to 32-bit index.
            sp_x_indices = paddle.cast(sp_x.indices(), "int32")
            sp_x = paddle.sparse.sparse_coo_tensor(
                sp_x_indices, sp_x.values(), sp_x.shape
            )

            sp_y = origin_y.detach().to_sparse_coo(len(y_shape))
            # cast to 32-bit index.
            sp_y_indices = paddle.cast(sp_y.indices(), "int32")
            sp_y = paddle.sparse.sparse_coo_tensor(
                sp_y_indices, sp_y.values(), sp_y.shape
            )
        else:
            sp_x = origin_x.detach().to_sparse_csr()
            # cast to 32-bit index.
            sp_x_crows = paddle.cast(sp_x.crows(), "int32")
            sp_x_cols = paddle.cast(sp_x.cols(), "int32")
            sp_x = paddle.sparse.sparse_csr_tensor(
                sp_x_crows, sp_x_cols, sp_x.values(), sp_x.shape
            )

            sp_y = origin_y.detach().to_sparse_csr()
            # cast to 32-bit index.
            sp_y_crows = paddle.cast(sp_y.crows(), "int32")
            sp_y_cols = paddle.cast(sp_y.cols(), "int32")
            sp_y = paddle.sparse.sparse_csr_tensor(
                sp_y_crows, sp_y_cols, sp_y.values(), sp_y.shape
            )

        sp_x.stop_gradient = False
        sp_y.stop_gradient = False
        sp_out = paddle.sparse.matmul(sp_x, sp_y)

        np.testing.assert_allclose(
            sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05
        )

        dense_out.backward()
        sp_out.backward()
        np.testing.assert_allclose(
            sp_x.grad.to_dense().numpy(),
            dense_x.grad.numpy(),
            rtol=1e-05,
        )
        np.testing.assert_allclose(
            sp_y.grad.to_dense().numpy(), dense_y.grad.numpy(), rtol=1e-05
        )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
        "only support cuda>=11.0",
    )
    def test_matmul_2d(self):
        self.check_result([16, 12], [12, 10], 'coo')
        self.check_result([16, 12], [12, 10], 'csr')

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
        "only support cuda>=11.0",
    )
    def test_matmul_3d(self):
        self.check_result([8, 16, 12], [8, 12, 10], 'coo')
        self.check_result([8, 16, 12], [8, 12, 10], 'csr')


class TestMaskedMatmul(unittest.TestCase):
    # x: dense, y: dense, out: sparse_`csr
    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11030,
        "only support on cuda>=11.3",
    )
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
        out = paddle.sparse.masked_matmul(x, y, mask)

        np.testing.assert_allclose(
            np_out.indptr, out.crows().numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            np_out.indices, out.cols().numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            np_out.data, out.values().numpy(), rtol=1e-05
        )

        out.backward()
        np.testing.assert_allclose(out.is_sparse_csr(), True, rtol=1e-05)
        np.testing.assert_allclose(np_x_grad, x.grad.numpy(), rtol=1e-05)
        np.testing.assert_allclose(np_y_grad, y.grad.numpy(), rtol=1e-05)

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11080,
        "only support on cuda>=11.8",
    )
    def test_masked_matmul_3d(self):
        paddle.set_default_dtype('float32')
        origin_x = paddle.rand([16, 16, 12])
        mask = paddle.randint(0, 2, [16, 12])
        origin_x = origin_x * mask.astype('float32')
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
        sp_out = paddle.sparse.matmul(sp_x, sp_y)
        sp_out.backward()

        np.testing.assert_allclose(
            sp_out.numpy(), dense_out.numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            sp_x.grad.to_dense().numpy(),
            (dense_x.grad * mask.astype('float32')).numpy(),
            rtol=1e-05,
        )
        np.testing.assert_allclose(
            sp_y.grad.numpy(), dense_y.grad.numpy(), rtol=1e-05
        )


class TestMatmulSparseDenseStatic(unittest.TestCase):
    # x: sparse, y: dense, out: dense
    def check_result(self, x_shape, y_shape):
        # only support sparse_coo_tensor in static graph
        if len(x_shape) == 3:
            mask = paddle.randint(0, 2, [x_shape[-2], x_shape[-1]])
        else:
            mask = paddle.randint(0, 2, x_shape)
        origin_x = paddle.rand(x_shape) * mask.astype(
            paddle.get_default_dtype()
        )
        origin_y = paddle.rand(y_shape)

        dense_x = origin_x.detach()
        dense_y = origin_y.detach()
        dense_out = paddle.matmul(dense_x, dense_y)

        indices_data, values_data = (
            origin_x.detach().to_sparse_coo(len(x_shape)).indices(),
            origin_x.detach().to_sparse_coo(len(x_shape)).values(),
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
            sp_y = paddle.static.data(
                name='sp_y',
                shape=origin_y.shape,
                dtype=origin_y.dtype,
            )
            sp_out = paddle.sparse.matmul(sp_x, sp_y)
            exe = paddle.static.Executor()
            fetch = exe.run(
                feed={
                    'indices': indices_data.numpy(),
                    'values': values_data.numpy(),
                    'sp_y': origin_y.detach().numpy(),
                },
                fetch_list=[sp_out],
                return_numpy=False,
            )
            sp_out = fetch[0]
            np.testing.assert_allclose(
                sp_out.numpy(), dense_out.numpy(), rtol=1e-05
            )
            paddle.disable_static()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
        "only support cuda>=11.0",
    )
    def test_matmul_2d(self):
        if in_pir_mode():
            self.check_result([16, 12], [12, 10])

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11080,
        "only support cuda>=11.8",
    )
    def test_matmul_3d(self):
        if in_pir_mode():
            self.check_result([8, 16, 12], [8, 12, 10])


class TestMatmulSparseSparseStatic(unittest.TestCase):
    '''
    only support sparse_coo_tensor in static graph
    '''

    # x: sparse, y: sparse, out: sparse
    def check_result(self, x_shape, y_shape):
        origin_x = paddle.rand(x_shape)
        origin_y = paddle.rand(y_shape)

        dense_x = origin_x.detach()
        dense_y = origin_y.detach()
        dense_out = paddle.matmul(dense_x, dense_y)

        x_indices_data, x_values_data = (
            origin_x.detach().to_sparse_coo(len(x_shape)).indices(),
            origin_x.detach().to_sparse_coo(len(x_shape)).values(),
        )
        y_indices_data, y_values_data = (
            origin_y.detach().to_sparse_coo(len(y_shape)).indices(),
            origin_y.detach().to_sparse_coo(len(y_shape)).values(),
        )
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x_indices = paddle.static.data(
                name='x_indices',
                shape=x_indices_data.shape,
                dtype=x_indices_data.dtype,
            )
            x_values = paddle.static.data(
                name='x_values',
                shape=x_values_data.shape,
                dtype=x_values_data.dtype,
            )
            sp_x = paddle.sparse.sparse_coo_tensor(
                x_indices,
                x_values,
                shape=origin_x.shape,
                dtype=origin_x.dtype,
            )
            y_indices = paddle.static.data(
                name='y_indices',
                shape=y_indices_data.shape,
                dtype=y_indices_data.dtype,
            )
            y_values = paddle.static.data(
                name='y_values',
                shape=y_values_data.shape,
                dtype=y_values_data.dtype,
            )
            sp_y = paddle.sparse.sparse_coo_tensor(
                y_indices,
                y_values,
                shape=origin_y.shape,
                dtype=origin_y.dtype,
            )
            sp_out = paddle.sparse.matmul(sp_x, sp_y)
            exe = paddle.static.Executor()
            fetch = exe.run(
                feed={
                    'x_indices': x_indices_data.numpy(),
                    'x_values': x_values_data.numpy(),
                    'y_indices': y_indices_data.numpy(),
                    'y_values': y_values_data.numpy(),
                },
                fetch_list=[sp_out],
                return_numpy=False,
            )
            sp_out = fetch[0]
            np.testing.assert_allclose(
                sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05
            )
            paddle.disable_static()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
        "only support cuda>=11.0",
    )
    def test_matmul_2d(self):
        if in_pir_mode():
            self.check_result([16, 12], [12, 10])

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11000,
        "only support cuda>=11.0",
    )
    def test_matmul_3d(self):
        if in_pir_mode():
            self.check_result([8, 16, 12], [8, 12, 10])


class TestMaskedMatmulStatic(unittest.TestCase):
    '''
    only support sparse_csr_tensor in static graph
    '''

    # x: dense, y: dense, out: sparse_csr
    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11030,
        "only support on cuda>=11.3",
    )
    def test_masked_matmul_2d(self):
        if in_pir_mode():
            np_mask = np.random.rand(10, 6) < 0.2

            np_x = np.random.rand(10, 12)
            np_y = np.random.rand(12, 6)

            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            mask = paddle.to_tensor(np.ones([10, 6]) * np_mask).to_sparse_coo(
                len(np_mask.shape)
            )
            out = paddle.sparse.masked_matmul(x, y, mask)

            indices_data, values_data = (
                mask.indices(),
                mask.values(),
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
                sp_mask = paddle.sparse.sparse_coo_tensor(
                    indices,
                    values,
                    shape=mask.shape,
                    dtype=mask.dtype,
                )
                sp_x = paddle.static.data(
                    name='x',
                    shape=x.shape,
                    dtype=x.dtype,
                )
                sp_y = paddle.static.data(
                    name='y',
                    shape=y.shape,
                    dtype=y.dtype,
                )
                out = paddle.sparse.masked_matmul(sp_x, sp_y, sp_mask)
                exe = paddle.static.Executor()
                fetch = exe.run(
                    feed={
                        'indices': indices_data.numpy(),
                        'values': values_data.numpy(),
                        'x': x.numpy(),
                        'y': y.numpy(),
                    },
                    fetch_list=[out],
                    return_numpy=False,
                )
                sp_out = fetch[0]
                np.testing.assert_allclose(
                    sp_out.to_dense().numpy(),
                    out.to_dense().numpy(),
                    rtol=1e-05,
                )
                paddle.disable_static()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda() or get_cuda_version() < 11080,
        "only support on cuda>=11.8",
    )
    def test_masked_matmul_3d(self):
        if in_pir_mode():
            paddle.set_default_dtype('float32')
            origin_x = paddle.rand([16, 16, 12])
            mask = paddle.randint(0, 2, [16, 12])
            origin_x = origin_x * mask.astype('float32')
            origin_y = paddle.rand([16, 12, 10])
            x = origin_x.detach()
            y = origin_y.detach()

            mask = paddle.to_tensor(np.ones([16, 12]) * mask).to_sparse_coo(
                len(mask.shape)
            )
            out = paddle.sparse.masked_matmul(x, y, mask)

            indices_data, values_data = (
                mask.indices(),
                mask.values(),
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
                sp_mask = paddle.sparse.sparse_coo_tensor(
                    indices,
                    values,
                    shape=mask.shape,
                    dtype=mask.dtype,
                )
                sp_x = paddle.static.data(
                    name='x',
                    shape=origin_x.shape,
                    dtype=origin_x.dtype,
                )
                sp_y = paddle.static.data(
                    name='y',
                    shape=origin_y.shape,
                    dtype=origin_y.dtype,
                )
                out = paddle.sparse.masked_matmul(sp_x, sp_y, sp_mask)
                exe = paddle.static.Executor()
                fetch = exe.run(
                    feed={
                        'indices': indices_data.numpy(),
                        'values': values_data.numpy(),
                        'x': origin_x.numpy(),
                        'y': origin_y.numpy(),
                    },
                    fetch_list=[out],
                    return_numpy=False,
                )
                sp_out = fetch[0]
                np.testing.assert_allclose(
                    sp_out.to_dense().numpy(),
                    out.to_dense().numpy(),
                    rtol=1e-05,
                )
                paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
