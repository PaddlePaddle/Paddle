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
            shape = [2, 6, 6, 6, 4]
            dense_x = paddle.randn(shape)
            print(dense_x)
            batch_norm = paddle.nn.BatchNorm3D(4, data_format="NDHWC")
            dense_y = batch_norm(dense_x)
            sparse_dim = 4
            sparse_x = dense_x.to_sparse_coo(sparse_dim)
            batch_norm = paddle.sparse.BatchNorm(4)
