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
from typing import Union, Callable
import numpy as np
import paddle
from paddle.fluid.framework import _test_eager_guard
from paddle import _C_ops


class TestSparseActivation(unittest.TestCase):
    def compare_with_dense(
        self,
        x,
        to_sparse: Callable[[paddle.Tensor], paddle.Tensor],
        dense_func: Callable[[paddle.Tensor], paddle.Tensor],
        sparse_func: Callable[[paddle.Tensor], paddle.Tensor],
        test_gradient: bool,
    ):
        def tensor_allclose(dense_tensor: paddle.Tensor, sparse_tensor: paddle.Tensor):
            dense_numpy = dense_tensor.numpy()
            mask = ~(np.isnan(dense_numpy) | np.isinf(dense_numpy))
            return np.allclose(
                dense_numpy[mask], sparse_tensor.to_dense().numpy()[mask]
            )

        with _test_eager_guard():
            dense_x = paddle.to_tensor(
                x, dtype="float32", stop_gradient=not test_gradient
            )

            sparse_x = to_sparse(dense_x)
            sparse_out = sparse_func(sparse_x)

            dense_x = paddle.to_tensor(
                x, dtype="float32", stop_gradient=not test_gradient
            )
            dense_out = dense_func(dense_x)

            assert tensor_allclose(dense_out, sparse_out)

            if test_gradient:
                dense_out.backward(dense_out)
                sparse_out.backward(sparse_out)
                assert tensor_allclose(dense_x.grad, sparse_x.grad)

    def test_sparse_relu(self):
        x = [[0, -1, 0, 2], [0, 0, -3, 0], [4, 5, 0, 0]]
        sparse_dim = 2
        self.compare_with_dense(
            x,
            lambda x: x.to_sparse_coo(sparse_dim),
            lambda x: paddle.nn.ReLU()(x),
            lambda x: paddle.sparse.ReLU()(x),
            True,
        )
        self.compare_with_dense(
            x,
            lambda x: x.to_sparse_csr(),
            lambda x: paddle.nn.ReLU()(x),
            lambda x: paddle.sparse.ReLU()(x),
            False,
        )

    def test_sparse_sqrt(self):
        x = [[0, 16, 0, 0], [0, 0, 0, 0], [0, 4, 2, 0]]
        sparse_dim = 2
        self.compare_with_dense(
            x,
            lambda x: x.to_sparse_coo(sparse_dim),
            paddle.sqrt,
            paddle.sparse.functional.sqrt,
            True,
        )
        self.compare_with_dense(
            x,
            lambda x: x.to_sparse_csr(),
            paddle.sqrt,
            paddle.sparse.functional.sqrt,
            False,
        )

    def test_sparse_sin(self):
        x = [[0, 16, 0, 0], [0, 0, 0, 0], [0, 4, 2, 0]]
        sparse_dim = 2
        self.compare_with_dense(
            x,
            lambda x: x.to_sparse_coo(sparse_dim),
            paddle.sin,
            paddle.sparse.functional.sin,
            True,
        )
        self.compare_with_dense(
            x,
            lambda x: x.to_sparse_csr(),
            paddle.sin,
            paddle.sparse.functional.sin,
            False,
        )

    def test_sparse_tanh(self):
        x = [[0, 16, 0, 0], [0, 0, 0, 0], [0, -4, 2, 0]]
        sparse_dim = 2
        self.compare_with_dense(
            x,
            lambda x: x.to_sparse_coo(sparse_dim),
            paddle.tanh,
            paddle.sparse.functional.tanh,
            True,
        )
        self.compare_with_dense(
            x,
            lambda x: x.to_sparse_csr(),
            paddle.tanh,
            paddle.sparse.functional.tanh,
            False,
        )

if __name__ == "__main__":
    unittest.main()
