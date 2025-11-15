# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.autograd.function import Function

SEED = 2023
np.random.seed(SEED)


def compare_result(result1, result2, rtol=1e-5, atol=0):
    np.testing.assert_allclose(
        result1.detach().numpy(),
        result2.detach().numpy(),
        rtol=rtol,
        atol=atol,
        err_msg=f'result1 is {result1}\nresult2 is {result2}',
    )


class TestFunctionBasic(unittest.TestCase):
    """Test basic Function usage with PyTorch-compatible API."""

    def test_simple_forward_backward(self):
        """Test simple forward and backward pass."""

        class ScaledLayer(Function):
            @staticmethod
            def forward(ctx, x):
                y = x * 3
                return y

            @staticmethod
            def backward(ctx, dy):
                dx = paddle.sin(dy)
                return dx

        paddle.seed(SEED)
        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False
        y = ScaledLayer.apply(x)
        y.sum().backward()

        self.assertEqual(y.shape, x.shape)
        self.assertIsNotNone(x.grad)

    def test_multiple_inputs(self):
        """Test Function with multiple inputs."""

        class ScaledLayer2(Function):
            @staticmethod
            def forward(ctx, x1, x2):
                y = 3 * x1 + x2 / 5
                return y

            @staticmethod
            def backward(ctx, dy):
                dx1 = paddle.sin(dy)
                dx2 = paddle.cos(dy)
                return dx1, dx2

        paddle.seed(SEED)
        x1 = paddle.randn([2, 3], dtype="float32")
        x2 = paddle.randn([2, 3], dtype="float32")
        x1.stop_gradient = False
        x2.stop_gradient = False

        y = ScaledLayer2.apply(x1, x2)
        y.sum().backward()

        self.assertEqual(y.shape, x1.shape)
        self.assertIsNotNone(x1.grad)
        self.assertIsNotNone(x2.grad)


class TestFunctionSaveForBackward(unittest.TestCase):
    """Test save_for_backward and saved_tensors (PyTorch-compatible)."""

    def test_saved_tensors_property(self):
        """Test that saved_tensors works as a property (PyTorch style)."""

        class CusTanh(Function):
            @staticmethod
            def forward(ctx, x):
                y = paddle.tanh(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, dy):
                # Use saved_tensors as property (PyTorch style)
                (y,) = ctx.saved_tensors
                grad = dy * (1 - paddle.square(y))
                return grad

        paddle.seed(SEED)
        x = paddle.randn([2, 3], dtype="float64")
        x.stop_gradient = False
        y = CusTanh.apply(x)
        y.sum().backward()

        self.assertIsNotNone(x.grad)

    def test_multiple_saved_tensors(self):
        """Test saving multiple tensors."""

        class CusTanh2(Function):
            @staticmethod
            def forward(ctx, x1, x2):
                y1 = paddle.tanh(x1)
                y2 = paddle.tanh(x2)
                ctx.save_for_backward(y1, y2)
                return y1 + y2

            @staticmethod
            def backward(ctx, dy):
                # Unpack multiple saved tensors
                y1, y2 = ctx.saved_tensors
                grad1 = dy * (1 - paddle.square(y1))
                grad2 = dy * (1 - paddle.square(y2))
                return grad1, grad2

        paddle.seed(SEED)
        x1 = paddle.randn([2, 3], dtype="float64")
        x2 = paddle.randn([2, 3], dtype="float64")
        x1.stop_gradient = False
        x2.stop_gradient = False

        y = CusTanh2.apply(x1, x2)
        y.sum().backward()

        self.assertIsNotNone(x1.grad)
        self.assertIsNotNone(x2.grad)


class TestFunctionContextAttributes(unittest.TestCase):
    """Test custom attributes on context."""

    def test_custom_context_attributes(self):
        """Test storing custom attributes in ctx."""

        class CusTanhWithFunc(Function):
            @staticmethod
            def forward(ctx, x, func=paddle.square):
                ctx.func = func
                y = paddle.tanh(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, dy):
                (y,) = ctx.saved_tensors
                grad = dy * (1 - ctx.func(y))
                return grad

        paddle.seed(SEED)
        x = paddle.randn([2, 3], dtype="float64")
        x.stop_gradient = False
        y = CusTanhWithFunc.apply(x, paddle.square)
        y.sum().backward()

        self.assertIsNotNone(x.grad)


class TestFunctionMarkNonDifferentiable(unittest.TestCase):
    """Test mark_non_differentiable method."""

    def test_mark_non_differentiable(self):
        """Test that mark_non_differentiable works correctly."""

        class CustomFunc(Function):
            @staticmethod
            def forward(ctx, x):
                a = x + x
                b = x + x + x
                ctx.mark_non_differentiable(a)
                return a, b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                # grad_a should be zeros since a is marked non-differentiable
                grad_x = 3 * grad_b
                return grad_x

        paddle.seed(SEED)
        x = paddle.randn([2, 3], dtype="float64")
        x.stop_gradient = False
        a, b = CustomFunc.apply(x)
        b.sum().backward()

        self.assertIsNotNone(x.grad)


class TestFunctionSetMaterializeGrads(unittest.TestCase):
    """Test set_materialize_grads method."""

    def test_set_materialize_grads_false(self):
        """Test set_materialize_grads(False) - gradients can be None."""

        class CustomFunc(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.set_materialize_grads(False)
                return x + x + x, x + x

            @staticmethod
            def backward(ctx, grad1, grad2):
                # When set_materialize_grads(False), unused grads are None
                # grad2 should be None if only first output is used
                return grad1

        paddle.seed(SEED)
        x = paddle.randn([2, 3], dtype="float64")
        x.stop_gradient = False
        y1, y2 = CustomFunc.apply(x)
        y1.sum().backward()  # Only backprop through y1

        self.assertIsNotNone(x.grad)


class TestFunctionCompatibility(unittest.TestCase):
    """Test compatibility between Function and PyLayer."""

    def test_function_vs_pylayer(self):
        """Verify that Function and PyLayer produce same results."""
        from paddle.autograd import PyLayer

        class TanhFunction(Function):
            @staticmethod
            def forward(ctx, x):
                y = paddle.tanh(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, dy):
                (y,) = ctx.saved_tensors
                grad = dy * (1 - paddle.square(y))
                return grad

        class TanhPyLayer(PyLayer):
            @staticmethod
            def forward(ctx, x):
                y = paddle.tanh(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, dy):
                (y,) = ctx.saved_tensor()
                grad = dy * (1 - paddle.square(y))
                return grad

        paddle.seed(SEED)
        x1 = paddle.randn([2, 3], dtype="float64")
        x1.stop_gradient = False

        paddle.seed(SEED)
        x2 = paddle.randn([2, 3], dtype="float64")
        x2.stop_gradient = False

        # Test forward
        y1 = TanhFunction.apply(x1)
        y2 = TanhPyLayer.apply(x2)
        compare_result(y1, y2)

        # Test backward
        y1.sum().backward()
        y2.sum().backward()
        compare_result(x1.grad, x2.grad)


if __name__ == '__main__':
    unittest.main()
