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
from paddle import _C_ops
from paddle.fluid.framework import _test_eager_guard


class TestSparseUtils(unittest.TestCase):
    def test_to_sparse_coo(self):
        with _test_eager_guard():
            x = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]]
            non_zero_indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            non_zero_elements = [1, 2, 3, 4, 5]
            dense_x = paddle.to_tensor(x)
            #TODO(zhangkaihuo): change to test the corresponding API
            out = _C_ops.final_state_to_sparse_coo(dense_x, 2)
            print(out)
            assert np.array_equal(out.non_zero_indices().numpy(),
                                  non_zero_indices)
            assert np.array_equal(out.non_zero_elements().numpy(),
                                  non_zero_elements)

            dense_tensor = _C_ops.final_state_to_dense(out)
            assert np.array_equal(dense_tensor.numpy(), x)

    def test_to_sparse_csr(self):
        with _test_eager_guard():
            x = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]]
            non_zero_crows = [0, 2, 3, 5]
            non_zero_cols = [1, 3, 2, 0, 1]
            non_zero_elements = [1, 2, 3, 4, 5]
            dense_x = paddle.to_tensor(x)
            out = _C_ops.final_state_to_sparse_csr(dense_x)
            assert np.array_equal(out.non_zero_crows().numpy(), non_zero_crows)
            assert np.array_equal(out.non_zero_cols().numpy(), non_zero_cols)
            assert np.array_equal(out.non_zero_elements().numpy(),
                                  non_zero_elements)

            dense_tensor = _C_ops.final_state_to_dense(out)
            assert np.array_equal(dense_tensor.numpy(), x)


if __name__ == "__main__":
    unittest.main()
