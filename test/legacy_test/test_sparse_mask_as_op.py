# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


def generate_data(shape, dtype):
    """
    Generate `data` and `mask` with the same shape and dtype.
    """
    _mask = np.random.randint(0, 2, shape)
    if np.sum(_mask) == 0:
        _mask.flat[0] = 1
    mask = (np.random.randint(-100, 100, shape) * _mask).astype(dtype)
    data = np.random.randint(-100, 100, shape).astype(dtype)
    return data, mask


class TestMaskAs(unittest.TestCase):
    def setUp(self):
        self.init_format()
        self.places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def init_format(self):
        self.format = None

    def check(self, shape, dtype, place, check_grad=True):
        paddle.disable_static()
        data_np, mask_np = generate_data(shape, dtype)

        data = paddle.to_tensor(data_np, dtype=dtype, place=place)
        data.stop_gradient = False

        if self.format == 'coo':
            mask = paddle.to_tensor(
                mask_np, dtype=dtype, place=place
            ).to_sparse_coo(len(shape))
        else:
            mask = paddle.to_tensor(
                mask_np, dtype=dtype, place=place
            ).to_sparse_csr()

        sparse_out = paddle.sparse.mask_as(data, mask)
        np_sparse = data_np * (mask_np != 0)

        np.testing.assert_allclose(
            sparse_out.to_dense().numpy(), np_sparse, rtol=1e-05
        )

        if check_grad:
            sparse_out.backward()
            sparse_grad = data.grad

            self.assertEqual(list(sparse_grad.shape), list(data.shape))
            self.assertEqual(sparse_grad.dtype, data.dtype)

            # make a dense tensor to compare the grad from sparse_out
            dense_tensor = paddle.to_tensor(data_np)
            dense_tensor.stop_gradient = False
            dense_out = (
                dense_tensor
                * paddle.to_tensor(mask_np != 0).astype(dense_tensor.dtype)
            ).astype(dtype)
            dense_out.backward()

            np.testing.assert_allclose(
                data.grad.numpy(),
                dense_tensor.grad.numpy(),
            )

    def check_with_dtypes(self, shape):
        for place in self.places:
            self.check(shape, 'float32', place)
            self.check(shape, 'float64', place)
            self.check(shape, 'int32', place)
            self.check(shape, 'int64', place)
            self.check(shape, 'complex64', place)
            self.check(shape, 'complex128', place)

            # `int8`` not registered in `FullLikeCooKernel`, so skip check_grad
            self.check(shape, 'int8', place, check_grad=False)

            # `int16` not registered in `multiply`, so skip check_grad
            self.check(shape, 'int16', place, check_grad=False)

        if paddle.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            self.check(shape, 'float16', place)


class TestMaskAsCoo(TestMaskAs):
    def init_format(self):
        self.format = 'coo'

    def test_1d(self):
        self.check_with_dtypes((5,))

    def test_2d(self):
        self.check_with_dtypes((5, 3))

    def test_3d(self):
        self.check_with_dtypes((5, 3, 4))

    def test_4d(self):
        self.check_with_dtypes((5, 3, 4, 2))


class TestMaskAsCsr(TestMaskAs):
    def init_format(self):
        self.format = 'csr'

    def test_2d(self):
        self.check_with_dtypes((5, 3))

    def test_3d(self):
        self.check_with_dtypes((5, 3, 4))

    def test_error_dimension(self):
        # error 1d
        with self.assertRaises(ValueError):
            self.check_with_dtypes((5,))

        # error 4d
        with self.assertRaises(ValueError):
            self.check_with_dtypes((5, 3, 4, 2))


if __name__ == "__main__":
    unittest.main()
