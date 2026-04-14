#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from paddle import _legacy_C_ops, base


class TestVariable(unittest.TestCase):
    def setUp(self):
        self.shape = [512, 768]
        self.dtype = np.float32
        self.array = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)

    def test_elementwise_add(self):
        with base.dygraph.guard():
            a = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            b = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            x = paddle.to_tensor(a)
            y = paddle.to_tensor(b)
            x.stop_gradient = False

            res1 = paddle.add(x, y)
            res2 = _legacy_C_ops.elementwise_add(x, y)

            np.testing.assert_array_equal(res1.numpy(), res2.numpy())

    def test_elementwise_mul(self):
        with base.dygraph.guard():
            a = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            b = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            x = paddle.to_tensor(a)
            y = paddle.to_tensor(b)

            res1 = paddle.multiply(x, y)
            res2 = _legacy_C_ops.elementwise_mul(x, y)

            np.testing.assert_array_equal(res1.numpy(), res2.numpy())

    def test_relu(self):
        with base.dygraph.guard():
            a = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            x = paddle.to_tensor(a)

            res1 = F.relu(x)
            res2 = _legacy_C_ops.relu(x)

            np.testing.assert_array_equal(res1.numpy(), res2.numpy())

    def test_trace_backward(self):
        with base.dygraph.guard():
            a = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            b = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            x = paddle.to_tensor(a)
            y = paddle.to_tensor(b)
            x.stop_gradient = False
            y.stop_gradient = False
            x.retain_grads()
            y.retain_grads()

            loss = _legacy_C_ops.elementwise_mul(x, y)
            loss.retain_grads()

            loss.backward()
            x_grad = x.gradient()
            y_grad = y.gradient()

            np.testing.assert_array_equal(x_grad, loss.gradient() * b)
            np.testing.assert_array_equal(y_grad, loss.gradient() * a)

    def test_retain_grad(self):
        """Test retain_grad() for both leaf nodes and intermediate nodes (new API)"""
        with base.dygraph.guard():
            # Prepare input data
            a = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            b = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            x = paddle.to_tensor(a)
            y = paddle.to_tensor(b)
            x.stop_gradient = False
            y.stop_gradient = False

            # ===== Test leaf nodes (x, y) =====
            # Create scalar loss for leaf nodes (must be scalar)
            loss_leaf = paddle.sum(_legacy_C_ops.elementwise_mul(x, y))
            x.retain_grad()
            y.retain_grad()
            loss_leaf.backward()

            # Verify leaf node gradients (x.grad = y, y.grad = x)
            np.testing.assert_array_equal(x.gradient(), b)
            np.testing.assert_array_equal(y.gradient(), a)

            # ===== Test intermediate node (z = x * y) =====
            # Create intermediate node z
            z = _legacy_C_ops.elementwise_mul(x, y)
            z.retain_grad()  # Retain gradient for intermediate node

            # Create scalar loss for intermediate node
            loss_mid = paddle.sum(z)
            loss_mid.backward()

            # Verify intermediate node gradient (d(loss_mid)/dz = 1)
            expected_z_grad = np.ones_like(a)
            np.testing.assert_array_equal(z.gradient(), expected_z_grad)


if __name__ == '__main__':
    unittest.main()
