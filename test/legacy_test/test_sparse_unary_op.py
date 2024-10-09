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

import paddle
from paddle.base.framework import convert_np_dtype_to_dtype_, in_pir_mode

devices = ['cpu', 'gpu']


class TestSparseUnary(unittest.TestCase):
    def to_sparse(self, x, format):
        if format == 'coo':
            return x.detach().to_sparse_coo(sparse_dim=x.ndim)
        elif format == 'csr':
            return x.detach().to_sparse_csr()

    def check_result(
        self,
        dense_func,
        sparse_func,
        format,
        device='cpu',
        dtype='float32',
        *args,
    ):
        if dtype == 'complex64':
            origin_x_real = paddle.rand([8, 16, 32], 'float32')
            origin_x_com = paddle.rand([8, 16, 32], 'float32')
            origin_x = (origin_x_real + 1j * origin_x_com).astype('complex64')
            mask = paddle.randint(0, 2, [8, 16, 32]).astype("float32")
            while paddle.sum(mask) == 0:
                mask = paddle.randint(0, 2, [8, 16, 32]).astype("float32")
        elif dtype == 'complex128':
            origin_x_real = paddle.rand([8, 16, 32], 'float64')
            origin_x_com = paddle.rand([8, 16, 32], 'float64')
            origin_x = (origin_x_real + 1j * origin_x_com).astype('complex128')
            mask = paddle.randint(0, 2, [8, 16, 32]).astype("float64")
            while paddle.sum(mask) == 0:
                mask = paddle.randint(0, 2, [8, 16, 32]).astype("float64")
        else:
            origin_x = paddle.rand([8, 16, 32], dtype)
            mask = paddle.randint(0, 2, [8, 16, 32]).astype(dtype)
            while paddle.sum(mask) == 0:
                mask = paddle.randint(0, 2, [8, 16, 32]).astype(dtype)

        # --- check sparse coo with dense --- #
        dense_x = origin_x * mask
        dense_x.to(device)
        sp_x = self.to_sparse(dense_x, format)
        sp_x.stop_gradient = False
        if len(args) == 0:
            sp_out = sparse_func(sp_x)
        elif len(args) == 1:
            sp_out = sparse_func(sp_x, args[0])
        elif len(args) == 2:
            sp_out = sparse_func(sp_x, args[0], args[1])
        sp_out.backward()

        dense_x.stop_gradient = False
        if len(args) == 0:
            dense_out = dense_func(dense_x)
        elif len(args) == 1:
            dense_out = dense_func(dense_x, args[0])
        elif len(args) == 2:
            if dense_func == paddle.cast:
                dense_out = dense_func(dense_x, args[1])

                int_dtype = convert_np_dtype_to_dtype_(args[0])
                if sp_out.is_sparse_csr():
                    self.assertEqual(sp_out.crows().dtype, int_dtype)
                    self.assertEqual(sp_out.cols().dtype, int_dtype)
                elif sp_out.is_sparse_coo():
                    self.assertEqual(sp_out.indices().dtype, int_dtype)
            else:
                dense_out = dense_func(dense_x, args[0], args[1])
        dense_out.backward()

        # compare forward
        np.testing.assert_allclose(
            sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05
        )

        # compare backward
        if dense_func == paddle.sqrt:
            expect_grad = np.nan_to_num(dense_x.grad.numpy(), 0.0, 0.0, 0.0)
        else:
            expect_grad = (dense_x.grad * mask).numpy()
        np.testing.assert_allclose(
            sp_x.grad.to_dense().numpy(), expect_grad, rtol=1e-05
        )

    def compare_with_dense(self, dense_func, sparse_func, dtype='float32'):
        for device in devices:
            # The sparse unary op is only compatible with float16 on the CUDA.
            if (device == 'cpu' and dtype != 'float16') or (
                device == 'gpu' and paddle.is_compiled_with_cuda()
            ):
                self.check_result(dense_func, sparse_func, 'coo', device, dtype)
                self.check_result(dense_func, sparse_func, 'csr', device, dtype)

    def compare_with_dense_one_attr(self, dense_func, sparse_func, attr1):
        for device in devices:
            if device == 'cpu' or (
                device == 'gpu' and paddle.is_compiled_with_cuda()
            ):
                self.check_result(
                    dense_func, sparse_func, 'coo', device, 'float32', attr1
                )
                self.check_result(
                    dense_func, sparse_func, 'csr', device, 'float32', attr1
                )

    def compare_with_dense_two_attr(
        self, dense_func, sparse_func, attr1, attr2
    ):
        for device in devices:
            if device == 'cpu' or (
                device == 'gpu' and paddle.is_compiled_with_cuda()
            ):
                self.check_result(
                    dense_func,
                    sparse_func,
                    'coo',
                    device,
                    'float32',
                    attr1,
                    attr2,
                )
                self.check_result(
                    dense_func,
                    sparse_func,
                    'csr',
                    device,
                    'float32',
                    attr1,
                    attr2,
                )

    def test_sparse_abs(self):
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'float16')
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'float32')
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'float64')
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'complex64')
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'complex128')

    def test_sparse_sin(self):
        self.compare_with_dense(paddle.sin, paddle.sparse.sin, 'float16')
        self.compare_with_dense(paddle.sin, paddle.sparse.sin, 'float32')
        self.compare_with_dense(paddle.sin, paddle.sparse.sin, 'float64')
        self.compare_with_dense(paddle.sin, paddle.sparse.sin, 'complex64')
        self.compare_with_dense(paddle.sin, paddle.sparse.sin, 'complex128')

    def test_sparse_tan(self):
        self.compare_with_dense(paddle.tan, paddle.sparse.tan, 'float16')
        self.compare_with_dense(paddle.tan, paddle.sparse.tan, 'float32')
        self.compare_with_dense(paddle.tan, paddle.sparse.tan, 'float64')
        self.compare_with_dense(paddle.tan, paddle.sparse.tan, 'complex64')
        self.compare_with_dense(paddle.tan, paddle.sparse.tan, 'complex128')

    def test_sparse_asin(self):
        self.compare_with_dense(paddle.asin, paddle.sparse.asin, 'float16')
        self.compare_with_dense(paddle.asin, paddle.sparse.asin, 'float32')
        self.compare_with_dense(paddle.asin, paddle.sparse.asin, 'float64')
        self.compare_with_dense(paddle.asin, paddle.sparse.asin, 'complex64')
        self.compare_with_dense(paddle.asin, paddle.sparse.asin, 'complex128')

    def test_sparse_atan(self):
        self.compare_with_dense(paddle.atan, paddle.sparse.atan, 'float16')
        self.compare_with_dense(paddle.atan, paddle.sparse.atan, 'float32')
        self.compare_with_dense(paddle.atan, paddle.sparse.atan, 'float64')
        self.compare_with_dense(paddle.atan, paddle.sparse.atan, 'complex64')
        self.compare_with_dense(paddle.atan, paddle.sparse.atan, 'complex128')

    def test_sparse_tanh(self):
        self.compare_with_dense(paddle.tanh, paddle.sparse.tanh, 'float16')
        self.compare_with_dense(paddle.tanh, paddle.sparse.tanh, 'float32')
        self.compare_with_dense(paddle.tanh, paddle.sparse.tanh, 'float64')
        self.compare_with_dense(paddle.tanh, paddle.sparse.tanh, 'complex64')
        self.compare_with_dense(paddle.tanh, paddle.sparse.tanh, 'complex128')

    def test_sparse_asinh(self):
        self.compare_with_dense(paddle.asinh, paddle.sparse.asinh, 'float16')
        self.compare_with_dense(paddle.asinh, paddle.sparse.asinh, 'float32')
        self.compare_with_dense(paddle.asinh, paddle.sparse.asinh, 'float64')
        self.compare_with_dense(paddle.asinh, paddle.sparse.asinh, 'complex64')
        self.compare_with_dense(paddle.asinh, paddle.sparse.asinh, 'complex128')

    def test_sparse_atanh(self):
        self.compare_with_dense(paddle.atanh, paddle.sparse.atanh, 'float16')
        self.compare_with_dense(paddle.atanh, paddle.sparse.atanh, 'float32')
        self.compare_with_dense(paddle.atanh, paddle.sparse.atanh, 'float64')
        self.compare_with_dense(paddle.atanh, paddle.sparse.atanh, 'complex64')
        self.compare_with_dense(paddle.atanh, paddle.sparse.atanh, 'complex128')

    def test_sparse_sqrt(self):
        self.compare_with_dense(paddle.sqrt, paddle.sparse.sqrt)

    def test_sparse_square(self):
        self.compare_with_dense(paddle.square, paddle.sparse.square, 'float16')
        self.compare_with_dense(paddle.square, paddle.sparse.square, 'float32')
        self.compare_with_dense(paddle.square, paddle.sparse.square, 'float64')
        self.compare_with_dense(
            paddle.square, paddle.sparse.square, 'complex64'
        )
        self.compare_with_dense(
            paddle.square, paddle.sparse.square, 'complex128'
        )

    def test_sparse_log1p(self):
        self.compare_with_dense(paddle.log1p, paddle.sparse.log1p, 'float16')
        self.compare_with_dense(paddle.log1p, paddle.sparse.log1p, 'float32')
        self.compare_with_dense(paddle.log1p, paddle.sparse.log1p, 'float64')
        self.compare_with_dense(paddle.log1p, paddle.sparse.log1p, 'complex64')
        self.compare_with_dense(paddle.log1p, paddle.sparse.log1p, 'complex128')

    def test_sparse_relu(self):
        self.compare_with_dense(paddle.nn.ReLU(), paddle.sparse.nn.ReLU())

    def test_sparse_relu6(self):
        self.compare_with_dense(paddle.nn.ReLU6(), paddle.sparse.nn.ReLU6())

    def test_sparse_leaky_relu(self):
        self.compare_with_dense(
            paddle.nn.LeakyReLU(0.1), paddle.sparse.nn.LeakyReLU(0.1)
        )

    def test_sparse_sinh(self):
        self.compare_with_dense(paddle.sinh, paddle.sparse.sinh, 'float16')
        self.compare_with_dense(paddle.sinh, paddle.sparse.sinh, 'float32')
        self.compare_with_dense(paddle.sinh, paddle.sparse.sinh, 'float64')
        self.compare_with_dense(paddle.sinh, paddle.sparse.sinh, 'complex64')
        self.compare_with_dense(paddle.sinh, paddle.sparse.sinh, 'complex128')

    def test_sparse_expm1(self):
        self.compare_with_dense(paddle.expm1, paddle.sparse.expm1, 'float16')
        self.compare_with_dense(paddle.expm1, paddle.sparse.expm1, 'float32')
        self.compare_with_dense(paddle.expm1, paddle.sparse.expm1, 'float64')
        self.compare_with_dense(paddle.expm1, paddle.sparse.expm1, 'complex64')
        self.compare_with_dense(paddle.expm1, paddle.sparse.expm1, 'complex128')

    def test_sparse_deg2rad(self):
        self.compare_with_dense(paddle.deg2rad, paddle.sparse.deg2rad)

    def test_sparse_rad2deg(self):
        self.compare_with_dense(paddle.rad2deg, paddle.sparse.rad2deg)

    def test_sparse_neg(self):
        self.compare_with_dense(paddle.neg, paddle.sparse.neg)

    def test_sparse_pow(self):
        self.compare_with_dense_one_attr(paddle.pow, paddle.sparse.pow, 3)

    def test_sparse_mul_scalar(self):
        self.compare_with_dense_one_attr(
            paddle.Tensor.__mul__, paddle.sparse.multiply, 3
        )

    def test_sparse_div_scalar(self):
        self.compare_with_dense_one_attr(
            paddle.Tensor.__div__, paddle.sparse.divide, 2
        )

    def test_sparse_cast(self):
        self.compare_with_dense_two_attr(
            paddle.cast, paddle.sparse.cast, 'int32', 'float32'
        )
        self.compare_with_dense_two_attr(
            paddle.cast, paddle.sparse.cast, 'int32', 'float64'
        )


class TestSparseUnaryStatic(unittest.TestCase):
    '''
    test sparse unary op with static graph in pir mode
    static graph only support sparse coo format
    '''

    def check_result_coo(
        self, dense_func, sparse_func, device='cpu', dtype='float32', *args
    ):
        paddle.set_device(device)
        if dtype == 'complex64':
            origin_x_real = paddle.rand([8, 16, 32], 'float32')
            origin_x_com = paddle.rand([8, 16, 32], 'float32')
            origin_x = (origin_x_real + 1j * origin_x_com).astype('complex64')
            mask = paddle.randint(0, 2, [8, 16, 32]).astype("float32")
            n = 0
            while paddle.sum(mask) == 0:
                mask = paddle.randint(0, 2, [8, 16, 32]).astype("float32")
                n += 1
                if n > 1000:
                    mask[0] = 1
                    break
        elif dtype == 'complex128':
            origin_x_real = paddle.rand([8, 16, 32], 'float64')
            origin_x_com = paddle.rand([8, 16, 32], 'float64')
            origin_x = (origin_x_real + 1j * origin_x_com).astype('complex128')
            mask = paddle.randint(0, 2, [8, 16, 32]).astype("float64")
            n = 0
            while paddle.sum(mask) == 0:
                mask = paddle.randint(0, 2, [8, 16, 32]).astype("float64")
                n += 1
                if n > 1000:
                    mask[0] = 1
                    break
        else:
            origin_x = paddle.rand([8, 16, 32], dtype)
            mask = paddle.randint(0, 2, [8, 16, 32]).astype(dtype)
            n = 0
            while paddle.sum(mask) == 0:
                mask = paddle.randint(0, 2, [8, 16, 32]).astype(dtype)
                n += 1
                if n > 1000:
                    mask[0] = 1
                    break

        # --- check sparse coo with dense --- #
        dense_x = origin_x * mask
        indices_data, values_data = (
            dense_x.detach().to_sparse_coo(sparse_dim=dense_x.ndim).indices(),
            dense_x.detach().to_sparse_coo(sparse_dim=dense_x.ndim).values(),
        )
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x_indices = paddle.static.data(
                name="x_indices",
                shape=indices_data.shape,
                dtype=indices_data.dtype,
            )
            x_values = paddle.static.data(
                name="x_values",
                shape=values_data.shape,
                dtype=values_data.dtype,
            )
            sparse_x = paddle.sparse.sparse_coo_tensor(
                x_indices,
                x_values,
                shape=dense_x.shape,
                dtype=dense_x.dtype,
            )
            if len(args) == 0:
                sparse_out = sparse_func(sparse_x)
            elif len(args) == 1:
                sparse_out = sparse_func(sparse_x, args[0])
            elif len(args) == 2:
                sparse_out = sparse_func(sparse_x, args[0], args[1])
            exe = paddle.static.Executor()
            sp_fetch = exe.run(
                feed={
                    "x_indices": x_indices.numpy(),
                    "x_values": x_values.numpy(),
                },
                fetch_list=[sparse_out],
                return_numpy=False,
            )
            sp_out = sp_fetch[0]

        dense_x.stop_gradient = False
        if len(args) == 0:
            dense_out = dense_func(dense_x)
        elif len(args) == 1:
            dense_out = dense_func(dense_x, args[0])
        elif len(args) == 2:
            if dense_func == paddle.cast:
                dense_out = dense_func(dense_x, args[1])

                int_dtype = convert_np_dtype_to_dtype_(args[0])
                # only support coo format
                self.assertEqual(sp_out.indices().dtype, int_dtype)
            else:
                dense_out = dense_func(dense_x, args[0], args[1])
        np.testing.assert_allclose(
            sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05
        )
        paddle.disable_static()

    def compare_with_dense(self, dense_func, sparse_func, dtype='float32'):
        if in_pir_mode():
            for device in devices:
                # The sparse unary op is only compatible with float16 on the CUDA.
                if (device == 'cpu' and dtype != 'float16') or (
                    device == 'gpu' and paddle.is_compiled_with_cuda()
                ):
                    self.check_result_coo(
                        dense_func, sparse_func, device, dtype
                    )

    def compare_with_dense_one_attr(self, dense_func, sparse_func, attr1):
        if in_pir_mode():
            for device in devices:
                if device == 'cpu' or (
                    device == 'gpu' and paddle.is_compiled_with_cuda()
                ):
                    self.check_result_coo(
                        dense_func, sparse_func, device, 'float32', attr1
                    )

    def compare_with_dense_two_attr(
        self, dense_func, sparse_func, attr1, attr2
    ):
        if in_pir_mode():
            for device in devices:
                if device == 'cpu' or (
                    device == 'gpu' and paddle.is_compiled_with_cuda()
                ):
                    self.check_result_coo(
                        dense_func,
                        sparse_func,
                        device,
                        'float32',
                        attr1,
                        attr2,
                    )

    def test_sparse_abs(self):
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'float16')
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'float32')
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'float64')
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'complex64')
        self.compare_with_dense(paddle.abs, paddle.sparse.abs, 'complex128')

    def test_sparse_sin(self):
        self.compare_with_dense(paddle.sin, paddle.sparse.sin, 'float16')
        self.compare_with_dense(paddle.sin, paddle.sparse.sin, 'float32')
        self.compare_with_dense(paddle.sin, paddle.sparse.sin, 'float64')
        self.compare_with_dense(paddle.sin, paddle.sparse.sin, 'complex64')
        self.compare_with_dense(paddle.sin, paddle.sparse.sin, 'complex128')

    def test_sparse_tan(self):
        self.compare_with_dense(paddle.tan, paddle.sparse.tan, 'float16')
        self.compare_with_dense(paddle.tan, paddle.sparse.tan, 'float32')
        self.compare_with_dense(paddle.tan, paddle.sparse.tan, 'float64')
        self.compare_with_dense(paddle.tan, paddle.sparse.tan, 'complex64')
        self.compare_with_dense(paddle.tan, paddle.sparse.tan, 'complex128')

    def test_sparse_asin(self):
        self.compare_with_dense(paddle.asin, paddle.sparse.asin, 'float16')
        self.compare_with_dense(paddle.asin, paddle.sparse.asin, 'float32')
        self.compare_with_dense(paddle.asin, paddle.sparse.asin, 'float64')
        self.compare_with_dense(paddle.asin, paddle.sparse.asin, 'complex64')
        self.compare_with_dense(paddle.asin, paddle.sparse.asin, 'complex128')

    def test_sparse_atan(self):
        self.compare_with_dense(paddle.atan, paddle.sparse.atan, 'float16')
        self.compare_with_dense(paddle.atan, paddle.sparse.atan, 'float32')
        self.compare_with_dense(paddle.atan, paddle.sparse.atan, 'float64')
        self.compare_with_dense(paddle.atan, paddle.sparse.atan, 'complex64')
        self.compare_with_dense(paddle.atan, paddle.sparse.atan, 'complex128')

    def test_sparse_tanh(self):
        self.compare_with_dense(paddle.tanh, paddle.sparse.tanh, 'float16')
        self.compare_with_dense(paddle.tanh, paddle.sparse.tanh, 'float32')
        self.compare_with_dense(paddle.tanh, paddle.sparse.tanh, 'float64')
        self.compare_with_dense(paddle.tanh, paddle.sparse.tanh, 'complex64')
        self.compare_with_dense(paddle.tanh, paddle.sparse.tanh, 'complex128')

    def test_sparse_asinh(self):
        self.compare_with_dense(paddle.asinh, paddle.sparse.asinh, 'float16')
        self.compare_with_dense(paddle.asinh, paddle.sparse.asinh, 'float32')
        self.compare_with_dense(paddle.asinh, paddle.sparse.asinh, 'float64')
        self.compare_with_dense(paddle.asinh, paddle.sparse.asinh, 'complex64')
        self.compare_with_dense(paddle.asinh, paddle.sparse.asinh, 'complex128')

    def test_sparse_atanh(self):
        self.compare_with_dense(paddle.atanh, paddle.sparse.atanh, 'float16')
        self.compare_with_dense(paddle.atanh, paddle.sparse.atanh, 'float32')
        self.compare_with_dense(paddle.atanh, paddle.sparse.atanh, 'float64')
        self.compare_with_dense(paddle.atanh, paddle.sparse.atanh, 'complex64')
        self.compare_with_dense(paddle.atanh, paddle.sparse.atanh, 'complex128')

    def test_sparse_sqrt(self):
        self.compare_with_dense(paddle.sqrt, paddle.sparse.sqrt)

    def test_sparse_square(self):
        self.compare_with_dense(paddle.square, paddle.sparse.square, 'float16')
        self.compare_with_dense(paddle.square, paddle.sparse.square, 'float32')
        self.compare_with_dense(paddle.square, paddle.sparse.square, 'float64')
        self.compare_with_dense(
            paddle.square, paddle.sparse.square, 'complex64'
        )
        self.compare_with_dense(
            paddle.square, paddle.sparse.square, 'complex128'
        )

    def test_sparse_log1p(self):
        self.compare_with_dense(paddle.log1p, paddle.sparse.log1p, 'float16')
        self.compare_with_dense(paddle.log1p, paddle.sparse.log1p, 'float32')
        self.compare_with_dense(paddle.log1p, paddle.sparse.log1p, 'float64')
        self.compare_with_dense(paddle.log1p, paddle.sparse.log1p, 'complex64')
        self.compare_with_dense(paddle.log1p, paddle.sparse.log1p, 'complex128')

    def test_sparse_relu(self):
        self.compare_with_dense(paddle.nn.ReLU(), paddle.sparse.nn.ReLU())

    def test_sparse_relu6(self):
        self.compare_with_dense(paddle.nn.ReLU6(), paddle.sparse.nn.ReLU6())

    def test_sparse_leaky_relu(self):
        self.compare_with_dense(
            paddle.nn.LeakyReLU(0.1), paddle.sparse.nn.LeakyReLU(0.1)
        )

    def test_sparse_sinh(self):
        self.compare_with_dense(paddle.sinh, paddle.sparse.sinh, 'float16')
        self.compare_with_dense(paddle.sinh, paddle.sparse.sinh, 'float32')
        self.compare_with_dense(paddle.sinh, paddle.sparse.sinh, 'float64')
        self.compare_with_dense(paddle.sinh, paddle.sparse.sinh, 'complex64')
        self.compare_with_dense(paddle.sinh, paddle.sparse.sinh, 'complex128')

    def test_sparse_expm1(self):
        self.compare_with_dense(paddle.expm1, paddle.sparse.expm1, 'float16')
        self.compare_with_dense(paddle.expm1, paddle.sparse.expm1, 'float32')
        self.compare_with_dense(paddle.expm1, paddle.sparse.expm1, 'float64')
        self.compare_with_dense(paddle.expm1, paddle.sparse.expm1, 'complex64')
        self.compare_with_dense(paddle.expm1, paddle.sparse.expm1, 'complex128')

    def test_sparse_deg2rad(self):
        self.compare_with_dense(paddle.deg2rad, paddle.sparse.deg2rad)

    def test_sparse_rad2deg(self):
        self.compare_with_dense(paddle.rad2deg, paddle.sparse.rad2deg)

    def test_sparse_neg(self):
        self.compare_with_dense(paddle.neg, paddle.sparse.neg)

    def test_sparse_pow(self):
        self.compare_with_dense_one_attr(paddle.pow, paddle.sparse.pow, 3)

    def test_sparse_mul_scalar(self):
        self.compare_with_dense_one_attr(
            paddle.Tensor.__mul__, paddle.sparse.multiply, 3
        )

    def test_sparse_div_scalar(self):
        self.compare_with_dense_one_attr(
            paddle.Tensor.__div__, paddle.sparse.divide, 2
        )

    def test_sparse_cast(self):
        self.compare_with_dense_two_attr(
            paddle.cast, paddle.sparse.cast, 'int32', 'float32'
        )
        self.compare_with_dense_two_attr(
            paddle.cast, paddle.sparse.cast, 'int32', 'float64'
        )


if __name__ == "__main__":
    unittest.main()
