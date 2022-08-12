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
from paddle.incubate import sparse
from paddle.incubate.sparse import nn
from paddle.fluid.framework import _test_eager_guard


class TestGradientAdd(unittest.TestCase):

    def sparse(self, sp_x):
        indentity = sp_x
        out = nn.functional.relu(sp_x)
        values = out.values() + indentity.values()
        out = sparse.sparse_coo_tensor(out.indices(),
                                       values,
                                       shape=out.shape,
                                       stop_gradient=out.stop_gradient)
        return out

    def dense(self, x):
        indentity = x
        out = paddle.nn.functional.relu(x)
        out = out + indentity
        return out

    def test(self):
        with _test_eager_guard():
            x = paddle.randn((3, 3))
            sparse_x = x.to_sparse_coo(sparse_dim=2)

            x.stop_gradient = False
            sparse_x.stop_gradient = False

            dense_out = self.dense(x)
            loss = dense_out.mean()
            loss.backward(retain_graph=True)

            sparse_out = self.sparse(sparse_x)
            sparse_loss = sparse_out.values().mean()
            sparse_loss.backward(retain_graph=True)

            assert np.allclose(dense_out.numpy(), sparse_out.to_dense().numpy())
            assert np.allclose(loss.numpy(), loss.numpy())
            assert np.allclose(x.grad.numpy(), sparse_x.grad.to_dense().numpy())

            loss.backward()
            sparse_loss.backward()

            assert np.allclose(x.grad.numpy(), sparse_x.grad.to_dense().numpy())


if __name__ == "__main__":
    unittest.main()
