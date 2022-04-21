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

from __future__ import print_function
import unittest
import numpy as np
import paddle
from paddle.fluid.framework import _test_eager_guard
import copy


class TestSparseBatchNorm(unittest.TestCase):
    def test(self):
        with _test_eager_guard():
            paddle.seed(0)
            channels = 4
            shape = [2, 3, 6, 6, channels]
            #there is no zero in dense_x
            dense_x = paddle.randn(shape)
            dense_x.stop_gradient = False

            batch_norm = paddle.nn.BatchNorm3D(channels, data_format="NDHWC")
            dense_y = batch_norm(dense_x)
            dense_y.backward(dense_y)

            sparse_dim = 4
            dense_x2 = copy.deepcopy(dense_x)
            dense_x2.stop_gradient = False
            sparse_x = dense_x2.to_sparse_coo(sparse_dim)
            sparse_batch_norm = paddle.sparse.BatchNorm(channels)
            # set same params
            sparse_batch_norm._mean.set_value(batch_norm._mean)
            sparse_batch_norm._variance.set_value(batch_norm._variance)
            sparse_batch_norm.weight.set_value(batch_norm.weight)

            sparse_y = sparse_batch_norm(sparse_x)
            # compare the result with dense batch_norm
            assert np.allclose(
                dense_y.flatten().numpy(),
                sparse_y.values().flatten().numpy(),
                atol=1e-5,
                rtol=1e-5)

            # test backward
            sparse_y.backward(sparse_y)
            assert np.allclose(
                dense_x.grad.flatten().numpy(),
                sparse_x.grad.values().flatten().numpy(),
                atol=1e-5,
                rtol=1e-5)

    def test_error_layout(self):
        with _test_eager_guard():
            with self.assertRaises(ValueError):
                shape = [2, 3, 6, 6, 3]
                x = paddle.randn(shape)
                sparse_x = x.to_sparse_coo(4)
                sparse_batch_norm = paddle.sparse.BatchNorm(
                    3, data_format='NCDHW')
                sparse_batch_norm(sparse_x)

    def test2(self):
        with _test_eager_guard():
            paddle.seed(123)
            channels = 3
            x_data = paddle.randn((1, 6, 6, 6, channels)).astype('float32')
            dense_x = paddle.to_tensor(x_data)
            sparse_x = dense_x.to_sparse_coo(4)
            batch_norm = paddle.sparse.BatchNorm(channels)
            batch_norm_out = batch_norm(sparse_x)
            print(batch_norm_out.shape)
            # [1, 6, 6, 6, 3]


if __name__ == "__main__":
    unittest.main()
