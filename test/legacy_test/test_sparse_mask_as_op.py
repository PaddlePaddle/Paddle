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

import os
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
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def init_format(self):
        self.format = None

    def check(self, shape, dtype, place, check_grad=True):
        paddle.disable_static()
        dense_data_np, dense_mask_np = generate_data(shape, dtype)

        dense_data_pd = paddle.to_tensor(
            dense_data_np, dtype=dtype, place=place
        )
        dense_data_pd.stop_gradient = False

        if self.format == 'coo':
            sparse_mask_pd = paddle.to_tensor(
                dense_mask_np, dtype=dtype, place=place
            ).to_sparse_coo(len(shape))
        else:
            sparse_mask_pd = paddle.to_tensor(
                dense_mask_np, dtype=dtype, place=place
            ).to_sparse_csr()

        sparse_out_pd = paddle.sparse.mask_as(dense_data_pd, sparse_mask_pd)

        # compare the tensor from sparse->dense with reference numpy data
        # the result only keeps the values where mask not zero, like:
        # dense_data_np
        # [[ 38.  15.  76.]
        #  [-98. -75.  10.]
        #  [-52.  49. -48.]]
        # dense_mask_np
        # [[-70.   0.   0.]
        #  [-50.  34.  60.]
        #  [-34.   0. -18.]]
        # dense_data_np_ref
        # [[ 38.   0.   0.]
        #  [-98. -75.  10.]
        #  [-52.   0. -48.]]
        dense_data_np_ref = dense_data_np * (dense_mask_np != 0)
        np.testing.assert_allclose(
            sparse_out_pd.to_dense().numpy(), dense_data_np_ref
        )

        if check_grad:
            # with sparse_out_pd backward, we get the grad from dense_data_pd
            sparse_out_pd.backward()
            dense_data_grad = dense_data_pd.grad

            self.assertEqual(
                list(dense_data_grad.shape), list(dense_data_pd.shape)
            )
            self.assertEqual(dense_data_grad.dtype, dense_data_pd.dtype)

            # make a dense data to compare the grad from sparse_out_pd
            grad_ref = np.ones_like(dense_mask_np) * (dense_mask_np != 0)

            np.testing.assert_allclose(
                dense_data_pd.grad.numpy(),
                grad_ref,
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
