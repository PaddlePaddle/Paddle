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
from paddle import _C_ops, _legacy_C_ops
from paddle.fluid import core
from paddle.fluid.framework import _test_eager_guard


class TestSparseCopy(unittest.TestCase):

    def test_copy_sparse_coo(self):
        with _test_eager_guard():
            np_x = [[0, 1.0, 0], [2.0, 0, 0], [0, 3.0, 0]]
            np_values = [1.0, 2.0, 3.0]
            dense_x = paddle.to_tensor(np_x, dtype='float32')
            coo_x = dense_x.to_sparse_coo(2)

            np_x_2 = [[0, 3.0, 0], [2.0, 0, 0], [0, 3.0, 0]]
            dense_x_2 = paddle.to_tensor(np_x_2, dtype='float32')
            coo_x_2 = dense_x_2.to_sparse_coo(2)
            coo_x_2.copy_(coo_x, True)
            assert np.array_equal(np_values, coo_x_2.values().numpy())

    def test_copy_sparse_csr(self):
        with _test_eager_guard():
            np_x = [[0, 1.0, 0], [2.0, 0, 0], [0, 3.0, 0]]
            np_values = [1.0, 2.0, 3.0]
            dense_x = paddle.to_tensor(np_x, dtype='float32')
            csr_x = dense_x.to_sparse_csr()

            np_x_2 = [[0, 3.0, 0], [2.0, 0, 0], [0, 3.0, 0]]
            dense_x_2 = paddle.to_tensor(np_x_2, dtype='float32')
            csr_x_2 = dense_x_2.to_sparse_csr()
            csr_x_2.copy_(csr_x, True)
            assert np.array_equal(np_values, csr_x_2.values().numpy())
