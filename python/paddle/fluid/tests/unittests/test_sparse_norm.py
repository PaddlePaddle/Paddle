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


class TestSparseBatchNorm(unittest.TestCase):
    def test_sparse_batch_norm(self):
        with _test_eager_guard():
            channels = 1
            shape = [1, 1, 6, 6, channels]
            dense_x = paddle.randn(shape)
            batch_norm = paddle.nn.BatchNorm3D(channels, data_format="NDHWC")
            dense_y = batch_norm(dense_x)

            sparse_dim = 4
            sparse_x = dense_x.to_sparse_coo(sparse_dim)
            sparse_batch_norm = paddle.sparse.BatchNorm(channels)
            sparse_y = sparse_batch_norm(sparse_x)
            assert np.allclose(
                dense_y.flatten().numpy(),
                sparse_y.values().flatten().numpy(),
                atol=1e-5,
                rtol=1e-5)

    def test(self):
        with _test_eager_guard():
            np.random.seed(123)
            channels = 3
            x_data = np.random.random(size=(1, 6, 6, 6,
                                            channels)).astype('float32')
            dense_x = paddle.to_tensor(x_data)
            sparse_x = dense_x.to_sparse_coo(4)
            batch_norm = paddle.sparse.BatchNorm(channels)
            batch_norm_out = batch_norm(sparse_x)
            print(batch_norm_out)


#TODO(zkh2016): add more test
