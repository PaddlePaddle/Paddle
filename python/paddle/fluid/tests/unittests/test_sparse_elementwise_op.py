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
import time
from paddle.fluid.framework import _test_eager_guard
from paddle import _C_ops, in_dynamic_mode
from paddle.fluid.op import get_all_op_protos
from paddle import _C_ops
from paddle.fluid.framework import core, dygraph_only


class TestSparseActivation(unittest.TestCase):
    def test_sparse_elwise_add(self):
        with _test_eager_guard():
            x = [[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]]
            y = [[0, 0, 0, -2], [0, 2, -3, 0], [2, 3, 4, 8]]

            # def dense_relu(x):
            #     dense_x = paddle.to_tensor(
            #         x, dtype='float32', stop_gradient=False)
            #     dense_relu = paddle.nn.ReLU()
            #     dense_out = dense_relu(dense_x)
            #     dense_out.backward(dense_out)
            #     return dense_out, dense_x.grad

            # ops = get_all_op_protos()

            # for op in ops:
            #     if 'sparse' not in op.type:
            #         ops.remove(op)

            dense_x = paddle.to_tensor(x, dtype='float32', stop_gradient=False)
            dense_y = paddle.to_tensor(y, dtype='float32', stop_gradient=False)
            sparse_x = dense_x.to_sparse_csr()
            sparse_y = dense_y.to_sparse_csr()
            # time.sleep(30)
            sparse_out = paddle.sparse.add(sparse_x, sparse_y)

            # sparse_out = _C_ops.final_state_sparse_elementwise_add(sparse_x, sparse_y)
            dense_out = dense_x + dense_y

            assert np.array_equal(dense_out.numpy(), sparse_out.to_dense().numpy())

            # sparse_relu = paddle.sparse.ReLU()
            # sparse_out = sparse_relu(sparse_x)
            # sparse_out.backward(sparse_out)
            #
            # dense_out, dense_x_grad = dense_relu(x)
            # assert np.array_equal(dense_out.numpy(),
            #                       sparse_out.to_dense().numpy())
            # assert np.array_equal(dense_x_grad.numpy(),
            #                       sparse_x.grad.to_dense().numpy())


if __name__ == "__main__":
    unittest.main()
