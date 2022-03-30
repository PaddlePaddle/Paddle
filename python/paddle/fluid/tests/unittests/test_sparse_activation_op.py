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


class TestSparseActivation(unittest.TestCase):
    def test_sparse_relu(self):
        with _test_eager_guard():
            x = [[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]]
            dense_x = paddle.to_tensor(x, dtype='float32')
            dense_shape = [3, 4]
            stop_gradient = True
            sparse_dim = 2
            sparse_coo_x = dense_x.to_sparse_coo(sparse_dim)
            #TODO(zhangkaihuo): change to test the corresponding API: paddle.sparse.relu(sparse_coo_x)
            sparse_act_out = _C_ops.final_state_sparse_relu(sparse_coo_x)
            correct_result = [0, 2, 0, 4, 5]
            actual_result = sparse_act_out.non_zero_elements().numpy()
            assert np.array_equal(correct_result, actual_result)


if __name__ == "__main__":
    unittest.main()
