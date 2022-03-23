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
from paddle.fluid import core
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
            print(out)
            assert np.array_equal(out.non_zero_crows().numpy(), non_zero_crows)
            assert np.array_equal(out.non_zero_cols().numpy(), non_zero_cols)
            assert np.array_equal(out.non_zero_elements().numpy(),
                                  non_zero_elements)

            dense_tensor = _C_ops.final_state_to_dense(out)
            assert np.array_equal(dense_tensor.numpy(), x)

    def test_backward(self):
        with _test_eager_guard():
            non_zero_indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2],
                                [1, 3, 2, 3]]
            non_zero_elements = [1, 2, 3, 4]
            indices = paddle.to_tensor(non_zero_indices, dtype='int32')
            elements = paddle.to_tensor(
                non_zero_elements, dtype='float32', stop_gradient=True)
            dense_shape = [1, 1, 3, 4, 1]
            #sparse_input = _C_ops.final_state_sparse_coo_tensor(indices, elements, dense_shape)
            sparse_input = core.eager.sparse_coo_tensor(indices, elements,
                                                        dense_shape, False)
            print("sparse_input:", sparse_input)
            print(sparse_input.non_zero_elements())
            #out, rulebook = _C_ops.final_state_conv3d(sparse_input, dense_kernel, paddings, dilations, strides, 1, False)
            #out = _C_ops.final_state_sparse_test(sparse_input)
            out1 = _C_ops.final_state_sparse_relu(sparse_input)
            print("out1:", out1)
            out2 = _C_ops.final_state_sparse_relu(sparse_input)
            print("out2:", out2)
            out = _C_ops.final_state_sparse_test(out1, out2)
            print("out", out)
            out.backward(out)
            print("sparse_input.grad:", sparse_input.grad)


if __name__ == "__main__":
    unittest.main()
