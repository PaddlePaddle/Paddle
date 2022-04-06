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
    def test_create_sparse_coo_tensor(self):
        with _test_eager_guard():
            indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            values = [1, 2, 3, 4, 5]
            dense_shape = [3, 4]
            dense_indices = paddle.to_tensor(indices)
            dense_elements = paddle.to_tensor(values, dtype='float32')
            stop_gradient = False
            coo = core.eager.sparse_coo_tensor(dense_indices, dense_elements,
                                               dense_shape, stop_gradient)
            # for-ci-coverage: test to_string.py
            print(coo)

    def test_create_sparse_csr_tensor(self):
        with _test_eager_guard():
            crows = [0, 2, 3, 5]
            cols = [1, 3, 2, 0, 1]
            values = [1, 2, 3, 4, 5]
            dense_shape = [3, 4]
            dense_crows = paddle.to_tensor(crows)
            dense_cols = paddle.to_tensor(cols)
            dense_elements = paddle.to_tensor(values, dtype='float32')
            stop_gradient = False
            csr = core.eager.sparse_csr_tensor(dense_crows, dense_cols,
                                               dense_elements, dense_shape,
                                               stop_gradient)
            # for-ci-coverage: test to_string.py
            print(csr)

    def test_to_sparse_coo(self):
        with _test_eager_guard():
            x = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]]
            indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            dense_x = paddle.to_tensor(x, dtype='float32', stop_gradient=False)
            out = dense_x.to_sparse_coo(2)
            assert np.array_equal(out.indices().numpy(), indices)
            assert np.array_equal(out.values().numpy(), values)
            #test to_sparse_coo_grad backward
            out_grad_indices = [[0, 1], [0, 1]]
            out_grad_values = [2.0, 3.0]
            out_grad = core.eager.sparse_coo_tensor(
                paddle.to_tensor(out_grad_indices),
                paddle.to_tensor(out_grad_values), out.shape, True)
            out.backward(out_grad)
            assert np.array_equal(dense_x.grad.numpy(),
                                  out_grad.to_dense().numpy())

    def test_coo_to_dense(self):
        with _test_eager_guard():
            indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            sparse_x = core.eager.sparse_coo_tensor(
                paddle.to_tensor(indices),
                paddle.to_tensor(values), [3, 4], False)
            dense_tensor = sparse_x.to_dense()
            #test to_dense_grad backward
            out_grad = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0]]
            dense_tensor.backward(paddle.to_tensor(out_grad))
            #mask the out_grad by sparse_x.indices() 
            correct_x_grad = [2.0, 4.0, 7.0, 9.0, 10.0]
            assert np.array_equal(correct_x_grad,
                                  sparse_x.grad.values().numpy())

    def test_to_sparse_csr(self):
        with _test_eager_guard():
            x = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]]
            crows = [0, 2, 3, 5]
            cols = [1, 3, 2, 0, 1]
            values = [1, 2, 3, 4, 5]
            dense_x = paddle.to_tensor(x)
            out = dense_x.to_sparse_csr()
            assert np.array_equal(out.crows().numpy(), crows)
            assert np.array_equal(out.cols().numpy(), cols)
            assert np.array_equal(out.values().numpy(), values)

            dense_tensor = out.to_dense()
            assert np.array_equal(dense_tensor.numpy(), x)

    def test_coo_values_grad(self):
        with _test_eager_guard():
            indices = [[0, 0, 1, 2, 2], [1, 3, 2, 0, 1]]
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            sparse_x = core.eager.sparse_coo_tensor(
                paddle.to_tensor(indices),
                paddle.to_tensor(values), [3, 4], False)
            values_tensor = sparse_x.values()
            # random out_grad
            out_grad = [2.0, 3.0, 5.0, 8.0, 9.0]
            # test coo_values_grad
            values_tensor.backward(paddle.to_tensor(out_grad))
            assert np.array_equal(out_grad, sparse_x.grad.values().numpy())


if __name__ == "__main__":
    unittest.main()
