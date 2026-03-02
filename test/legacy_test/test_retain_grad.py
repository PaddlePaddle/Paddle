好的，这是修正后的最终版单元测试代码。它修复了梯度期望值的错误，并确保每个测试都独立创建了所需的张量。

Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

import unittest
import paddle
import numpy as np

class TestRetainGrad(unittest.TestCase):
    def setUp(self):
        # Use GPU Frist
        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
        else:
            paddle.set_device('cpu')

    def _create_tensors(self):
        """
        Helper function to create input tensors for tests.
        This ensures each test starts with a fresh computational graph.
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        y = x + x
        return x, y

    def test_retain_grad(self):
        """Test new API: retain_grad()"""
        x, y = self._create_tensors()
        y.retain_grad()  # New API
        loss = y.sum()
        loss.backward()
        # Gradient of sum(y) w.r.t. y is [1., 1., 1.]
        np.testing.assert_array_equal(y.grad.numpy(), [1.0, 1.0, 1.0])
        print("✅ retain_grad() test passed")

    def test_retain_grads(self):
        """Test historical alias: retain_grads()"""
        x, y = self._create_tensors()
        y.retain_grads()  # Historical alias
        loss = y.sum()
        loss.backward()
        # Gradient of sum(y) w.r.t. y is [1., 1., 1.]
        np.testing.assert_array_equal(y.grad.numpy(), [1.0, 1.0, 1.0])
        print("✅ retain_grads() test passed")

    def test_both_methods(self):
        """Test using both new and old APIs simultaneously to ensure idempotency."""
        x, y = self._create_tensors()
        
        # First use new API
        y.retain_grad()
        # Then use alias. This should be a no-op if implemented correctly.
        y.retain_grads()
        
        loss = y.sum()
        loss.backward()
        
        # Main assertion: the gradient should exist and be correct ([1., 1., 1.]), not doubled.
        self.assertIsNotNone(y.grad, "Gradient should not be None after calling both methods.")
        self.assertEqual(y.grad.shape, y.shape, "Gradient shape mismatch.")
        
        expected_grad_for_y = np.array([1., 1., 1.]) # Because d(sum(y))/dy = 1 for each element
        np.testing.assert_array_equal(y.grad.numpy(), expected_grad_for_y)
        
        print("✅ Using both new and old APIs test passed - gradient is correct and not duplicated.")

    def test_gradient_values_consistency(self):
        """Test that retained gradients match the expected mathematical result."""
        x, y = self._create_tensors()
        
        y.retain_grad()
        z = y * 2  # y = [2, 4, 6], z = [4, 8, 12]
        loss = z.sum() # loss = 4 + 8 + 12 = 24
        
        loss.backward()
        
        # dz/dy = 2, so gradient flowing back to y is [2., 2., 2.]
        expected_grad_for_y = np.array([2., 2., 2.])
        np.testing.assert_array_equal(y.grad.numpy(), expected_grad_for_y)
        
        # dx/dloss = dz/dy * dy/dx * dloss/dz = 2 * 2 * 1 = 4 for each element
        expected_grad_for_x = np.array([4., 4., 4.])
        np.testing.assert_array_equal(x.grad.numpy(), expected_grad_for_x)
        
        print("✅ Gradient values consistency test passed")

if name == 'main':
    unittest.main()
