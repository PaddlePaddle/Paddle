#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class TestTensorRequiresGrad(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        paddle.disable_static()
        np.random.seed(1919)

    def tearDown(self):
        """Clean up after each test method."""
        paddle.disable_static()

    def test_basic_requires_grad_property(self):
        """Test basic requires_grad property functionality"""
        # Test default behavior - new tensors have stop_gradient=True by default
        x = paddle.randn([2, 3])
        self.assertFalse(x.requires_grad)
        self.assertTrue(x.stop_gradient)

        # Test setting requires_grad to True
        x.requires_grad = True
        self.assertTrue(x.requires_grad)
        self.assertFalse(x.stop_gradient)

        # Test setting requires_grad to False
        x.requires_grad = False
        self.assertFalse(x.requires_grad)
        self.assertTrue(x.stop_gradient)

    def test_requires_grad_consistency_with_stop_gradient(self):
        """Test that requires_grad is always the opposite of stop_gradient"""
        x = paddle.randn([3, 4])

        # Test multiple state changes
        states = [True, False, True, False]
        for requires_grad_state in states:
            x.requires_grad = requires_grad_state
            self.assertEqual(x.requires_grad, requires_grad_state)
            self.assertEqual(x.stop_gradient, not requires_grad_state)

            # Also test setting stop_gradient directly
            x.stop_gradient = requires_grad_state
            self.assertEqual(x.requires_grad, not requires_grad_state)
            self.assertEqual(x.stop_gradient, requires_grad_state)

    def test_requires_grad_type_checking(self):
        """Test type checking for requires_grad setter"""
        x = paddle.randn([2, 2])

        # Valid boolean values should work
        x.requires_grad = True
        x.requires_grad = False

        # Invalid types should raise TypeError
        invalid_values = ["true", 1, 0, None, [], {}]
        for invalid_value in invalid_values:
            with self.assertRaises(TypeError) as cm:
                x.requires_grad = invalid_value
            self.assertIn("requires_grad must be bool", str(cm.exception))

    def test_requires_grad_with_parameter(self):
        """Test requires_grad behavior with Parameter tensors"""
        # Create a parameter - Parameters have stop_gradient=False by default (trainable)
        param = paddle.create_parameter([3, 4], dtype='float32')
        self.assertTrue(
            param.requires_grad
        )  # Parameters require grad by default
        self.assertFalse(
            param.stop_gradient
        )  # Parameters are trainable by default

        # Test changing requires_grad on parameter
        param.requires_grad = False
        self.assertFalse(param.requires_grad)
        self.assertTrue(param.stop_gradient)

    def test_requires_grad_in_gradient_computation(self):
        """Test requires_grad behavior in actual gradient computation"""
        x = paddle.randn([2, 3])
        y = paddle.randn([2, 3])

        # Set both tensors to require grad
        x.requires_grad = True
        y.requires_grad = True

        z = x * y + x.sum()
        z.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

        # Clear gradients and test with requires_grad=False
        x.grad._clear_data()
        y.grad._clear_data()

        x.requires_grad = False
        y.requires_grad = True

        z = x * y + x.sum()
        z.backward()

        self.assertIsNone(x.grad)  # x doesn't require grad
        self.assertIsNotNone(y.grad)  # y requires grad

    def test_requires_grad_with_different_tensor_types(self):
        """Test requires_grad with different tensor creation methods"""
        # Test with different tensor creation functions
        tensor_creators = [
            lambda: paddle.randn([2, 3]),
            lambda: paddle.zeros([2, 3]),
            lambda: paddle.ones([2, 3]),
            lambda: paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32'),
            lambda: paddle.arange(6, dtype='float32').reshape([2, 3]),
        ]

        for creator in tensor_creators:
            x = creator()
            # All newly created tensors should have requires_grad=False by default
            self.assertFalse(x.requires_grad)
            self.assertTrue(x.stop_gradient)

            # Test modification
            x.requires_grad = True
            self.assertTrue(x.requires_grad)
            self.assertFalse(x.stop_gradient)

    def test_requires_grad_with_tensor_operations(self):
        """Test requires_grad preservation through tensor operations"""
        x = paddle.randn([3, 3])
        y = paddle.randn([3, 3])

        x.requires_grad = True
        y.requires_grad = False

        # Operations should preserve requires_grad appropriately
        z1 = x + y  # Should require grad (x requires grad)
        z2 = x * 2.0  # Should require grad (x requires grad)
        z3 = y.sin()  # Should not require grad (y doesn't require grad)

        self.assertTrue(z1.requires_grad)
        self.assertTrue(z2.requires_grad)
        self.assertFalse(z3.requires_grad)

    def test_requires_grad_with_detach(self):
        """Test requires_grad behavior with detach operation"""
        x = paddle.randn([2, 3])
        x.requires_grad = True

        y = x.detach()

        # Detached tensor should not require grad
        self.assertTrue(x.requires_grad)
        self.assertFalse(y.requires_grad)
        self.assertTrue(y.stop_gradient)

    def test_requires_grad_static_mode(self):
        """Test requires_grad behavior in static mode"""
        paddle.enable_static()

        try:
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')

                # In static mode, variables also have stop_gradient=True by default
                self.assertFalse(x.requires_grad)
                self.assertTrue(x.stop_gradient)

                # Test setting requires_grad in static mode
                x.requires_grad = True
                self.assertTrue(x.requires_grad)
                self.assertFalse(x.stop_gradient)

        finally:
            paddle.disable_static()

    def test_requires_grad_edge_cases(self):
        """Test edge cases for requires_grad"""
        # Test with scalar tensor
        scalar = paddle.to_tensor(3.14)
        self.assertFalse(scalar.requires_grad)  # False
        scalar.requires_grad = True
        self.assertTrue(scalar.requires_grad)

        # Test with empty tensor
        empty = paddle.empty([0, 3])
        self.assertFalse(empty.requires_grad)  # False
        empty.requires_grad = True
        self.assertTrue(empty.requires_grad)

        # Test with different dtypes
        dtypes = [paddle.float32, paddle.float64, paddle.int32, paddle.int64]
        for dtype in dtypes:
            x = paddle.ones([2, 2], dtype=dtype)
            # All tensors should have requires_grad=False by default
            self.assertFalse(x.requires_grad)

            # Float tensors should support requires_grad
            if dtype in [paddle.float32, paddle.float64]:
                x.requires_grad = True
                self.assertTrue(x.requires_grad)


if __name__ == '__main__':
    unittest.main()
