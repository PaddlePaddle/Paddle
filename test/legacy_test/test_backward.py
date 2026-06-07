# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import backward


class BackwardNet:
    """
    Abstract Base Class.
    All Net inherited this Class should implement two functions:
        build_model: build net to test the logic of backward
        init_data: fake input data to test all programs.
    """

    def __init__(self):
        self.stop_gradient_grad_vars = set()
        self.no_grad_vars = set()
        self.params_names = set()
        self.op_path = []

    def build_model(self):
        """
        Build net to test the logic of backward.
        :return: loss
        """
        raise NotImplementedError

    def init_data(self):
        """
        Fake input data to test all programs.
        :return: dict, {'var_name': var_data}
        """
        raise NotImplementedError


# TODO(Aurelius84): add conditional network test
class ConditionalNet(BackwardNet):
    def __init__(self):
        super().__init__()


class TestBackwardUninitializedVariable(unittest.TestCase):
    """this case is found in yolov5 while to_static.
    gradient aggregation may cause sum a invalid variable.
    """

    def test(self):
        paddle.enable_static()
        main_prg, startup_prg = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(main_prg, startup_prg):
            gt = paddle.static.data(name='gt', shape=[4], dtype='float32')
            x = paddle.static.data(name='x', shape=[2], dtype='float32')
            gt.stop_gradient = True
            x.stop_gradient = False
            gt = gt.reshape([4, 1]).reshape([4])
            loss = (
                paddle.nn.functional.binary_cross_entropy(x, gt[:2])
                + (gt[2:4] * x).sum()
            )
            exe = paddle.static.Executor()
            paddle.base.backward.gradients(loss, [])
            exe.run(startup_prg)
            # Optimizer
            out = exe.run(
                main_prg,
                feed={
                    'gt': np.array([1.0, 1.0, 0.0, 0.0], dtype='float32'),
                    'x': np.array([0.5, 0.5], dtype='float32'),
                },
                fetch_list=[loss],
            )
            print(out)


class TestStripGradSuffix(unittest.TestCase):
    def test_strip_grad_suffix(self):
        cases = (
            ('x@GRAD', 'x'),
            ('x@GRAD@GRAD', 'x'),
            ('x@GRAD@RENAME@1', 'x'),
            ('x@GRAD_slice_0@GRAD', 'x@GRAD_slice_0'),
            ('grad/grad/x@GRAD@RENAME@block0@1@GRAD', 'x'),
        )
        for input_, desired in cases:
            self.assertEqual(backward._strip_grad_suffix_(input_), desired)


class TestBackwardParamAlias(unittest.TestCase):
    """Test backward() with parameter alias: grad_tensor -> gradient"""

    def setUp(self):
        paddle.disable_static()

    def test_backward_with_grad_tensor_param(self):
        """Test backward using grad_tensor parameter name."""
        x = paddle.to_tensor([1.0, 2.0], dtype='float32', stop_gradient=False)
        y = x**2
        z = y.sum()
        grad_tensor = paddle.to_tensor(1.0, dtype='float32')
        z.backward(grad_tensor=grad_tensor)
        expected = [2.0, 4.0]
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5)

    def test_backward_with_gradient_alias(self):
        """Test backward using gradient alias parameter name."""
        x = paddle.to_tensor([1.0, 2.0], dtype='float32', stop_gradient=False)
        y = x**2
        z = y.sum()
        grad_tensor = paddle.to_tensor(1.0, dtype='float32')
        z.backward(gradient=grad_tensor)
        expected = [2.0, 4.0]
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5)

    def test_backward_alias_with_custom_grad(self):
        """Test gradient alias with custom gradient values."""
        x = paddle.to_tensor([[1.0, 2.0]], dtype='float32', stop_gradient=False)
        y = x * 3
        loss = y.sum()
        custom_grad = paddle.to_tensor(2.0, dtype='float32')
        loss.backward(gradient=custom_grad)
        expected = [[6.0, 6.0]]
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5)

    def test_backward_alias_with_retain_graph(self):
        """Test gradient alias combined with retain_graph parameter."""
        x = paddle.to_tensor([2.0], dtype='float32', stop_gradient=False)
        y = x**2
        loss = y.sum()
        grad_tensor = paddle.to_tensor(1.0, dtype='float32')
        loss.backward(gradient=grad_tensor, retain_graph=True)
        expected = [4.0]
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5)
        x.clear_grad()
        loss.backward(gradient=grad_tensor)
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5)

    def test_backward_alias_with_create_graph(self):
        """Test gradient alias combined with create_graph parameter."""
        x = paddle.to_tensor([1.0], dtype='float32', stop_gradient=False)
        y = x**2
        loss = y.sum()
        grad_tensor = paddle.to_tensor(1.0, dtype='float32')
        loss.backward(gradient=grad_tensor, create_graph=True)
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(x.grad.numpy(), [2.0], rtol=1e-5)


class TestBackwardCreateGraph(unittest.TestCase):
    """Test backward with create_graph parameter for higher-order gradients."""

    def setUp(self):
        paddle.disable_static()

    def test_tensor_backward_with_create_graph(self):
        """Test backward with create_graph=True for second-order gradients"""
        x = paddle.to_tensor(
            np.array([[1.0, 2.0], [3.0, 4.0]]), dtype='float32'
        )
        x.stop_gradient = False
        y = x * x
        loss = paddle.sum(y)
        # First backward with create_graph=True
        loss.backward(create_graph=True)
        # Verify first-order gradients
        self.assertIsNotNone(x.grad)
        first_grad = x.grad.numpy()
        np.testing.assert_allclose(
            first_grad, np.array([[2.0, 4.0], [6.0, 8.0]]), rtol=1e-7
        )
        # Compute second-order gradients
        grad_sum = paddle.sum(x.grad)
        grad_sum.backward()
        # Verify second-order gradients
        self.assertIsNotNone(x.grad)
        second_grad = x.grad.numpy()
        np.testing.assert_allclose(
            second_grad, np.array([[4.0, 6.0], [8.0, 10.0]]), rtol=1e-7
        )

    def test_backward_create_graph_second_order(self):
        """Test computing second-order gradients using create_graph=True."""
        x = paddle.to_tensor([1.0, 2.0], dtype='float32', stop_gradient=False)
        y = x**2
        loss = y.sum()
        # First backward with create_graph=True
        # x.grad should be [2.0, 4.0]
        paddle.autograd.backward(loss, create_graph=True)
        grad_sum = x.grad.sum()
        grad_sum.backward()
        # Check second-order gradients
        # sum up to [4.0, 6.0]
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(x.grad.numpy(), [4.0, 6.0], rtol=1e-5)

    def test_backward_create_graph_with_multiple_tensors(self):
        """Test backward with create_graph on multiple output tensors."""
        x = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype='float32', stop_gradient=False
        )
        z1 = x**2
        z2 = x * 3
        # Backward on z1
        paddle.autograd.backward(z1, create_graph=True)
        self.assertIsNotNone(x.grad)
        self.assertFalse(x.grad.stop_gradient)
        x.clear_grad()
        # Backward on z2
        paddle.autograd.backward(z2, create_graph=True)
        self.assertIsNotNone(x.grad)
        self.assertFalse(x.grad.stop_gradient)

    def test_backward_create_graph_with_grad_tensors(self):
        """Test backward with create_graph and custom grad_tensors."""
        x = paddle.to_tensor([1.0, 2.0], dtype='float32', stop_gradient=False)
        y = x**2
        z = y.sum()
        grad_tensor = paddle.to_tensor([1.0, 2.0], dtype='float32')
        paddle.autograd.backward(y, grad_tensors=grad_tensor, create_graph=True)
        # Check gradients with custom weights
        self.assertIsNotNone(x.grad)
        expected = [2.0 * 1.0, 4.0 * 2.0]  # dy/dx * weights
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5)
        self.assertFalse(x.grad.stop_gradient)

    def test_backward_create_graph_retain_graph(self):
        """Test backward with create_graph=True and retain_graph=True."""
        x = paddle.to_tensor([2.0], dtype='float32', stop_gradient=False)
        y = x**3
        loss = y.sum()
        # First backward
        paddle.autograd.backward(loss, create_graph=True, retain_graph=True)
        grad1 = x.grad.clone()
        x.clear_grad()
        # Second backward with same graph
        paddle.autograd.backward(loss, create_graph=True, retain_graph=False)
        grad2 = x.grad
        # Gradients should be the same
        np.testing.assert_allclose(grad1.numpy(), grad2.numpy(), rtol=1e-5)

    def test_backward_create_graph_chain_rule(self):
        """Test chain rule with higher-order gradients."""
        x = paddle.to_tensor([1.0], dtype='float32', stop_gradient=False)
        y = x**3
        loss = y**2
        # First backward
        paddle.autograd.backward(loss, create_graph=True, retain_graph=True)
        # At x=1: x.grad should be 6.0
        np.testing.assert_allclose(x.grad.numpy(), [6.0], rtol=1e-5)
        # Second backward
        grad_sum = paddle.sum(x.grad)
        paddle.autograd.backward(grad_sum)
        # At x=1: x.grad should be 30
        # Sum up to 36
        np.testing.assert_allclose(x.grad.numpy(), [36.0], rtol=1e-5)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
