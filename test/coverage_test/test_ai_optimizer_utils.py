# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# Unit test for paddle.optimizer optimizer error paths
# Target: cover uncovered lines 94-116 in paddle/python/paddle/optimizer/optimizer.py

import unittest

import paddle
import paddle.optimizer as opt


class TestSGDOptimizer(unittest.TestCase):
    """Test SGD optimizer.
    SGD does not accept momentum kwarg.
    """

    def setUp(self):
        paddle.disable_static()

    def test_sgd_basic(self):
        """SGD basic step."""
        x = paddle.to_tensor([1.0], dtype='float32')
        x.stop_gradient = False
        linear = paddle.nn.Linear(1, 1)
        sgd = opt.SGD(
            learning_rate=0.01,
            parameters=linear.parameters(),
        )
        out = linear(x)
        loss = out.mean()
        loss.backward()
        sgd.step()
        sgd.clear_grad()
        # Should not raise

    def test_sgd_with_weight_decay(self):
        """SGD with weight decay."""
        linear = paddle.nn.Linear(2, 2)
        sgd = opt.SGD(
            learning_rate=0.01,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
        x = paddle.randn([4, 2])
        out = linear(x)
        loss = out.mean()
        loss.backward()
        sgd.step()


class TestMomentumOptimizer(unittest.TestCase):
    """Test Momentum optimizer (for momentum tests)."""

    def setUp(self):
        paddle.disable_static()

    def test_momentum_basic(self):
        """Momentum basic step."""
        linear = paddle.nn.Linear(2, 2)
        momentum = opt.Momentum(
            learning_rate=0.01,
            momentum=0.9,
            parameters=linear.parameters(),
        )
        x = paddle.randn([4, 2])
        out = linear(x)
        loss = out.mean()
        loss.backward()
        momentum.step()

    def test_momentum_state_dict(self):
        """Momentum optimizer state_dict."""
        linear = paddle.nn.Linear(2, 2)
        momentum = opt.Momentum(
            learning_rate=0.01,
            momentum=0.9,
            parameters=linear.parameters(),
        )
        x = paddle.randn([4, 2])
        out = linear(x)
        loss = out.mean()
        loss.backward()
        momentum.step()
        state = momentum.state_dict()
        self.assertIsInstance(state, dict)

    def test_momentum_set_state_dict(self):
        """Momentum optimizer set_state_dict."""
        linear = paddle.nn.Linear(2, 2)
        mom1 = opt.Momentum(
            learning_rate=0.01,
            momentum=0.9,
            parameters=linear.parameters(),
        )
        x = paddle.randn([4, 2])
        out = linear(x)
        loss = out.mean()
        loss.backward()
        mom1.step()
        state = mom1.state_dict()
        # Create new optimizer and load state
        linear2 = paddle.nn.Linear(2, 2)
        mom2 = opt.Momentum(
            learning_rate=0.01,
            momentum=0.9,
            parameters=linear2.parameters(),
        )
        mom2.set_state_dict(state)


class TestAdamOptimizer(unittest.TestCase):
    """Test Adam optimizer."""

    def setUp(self):
        paddle.disable_static()

    def test_adam_basic(self):
        """Adam basic step."""
        linear = paddle.nn.Linear(2, 2)
        adam = opt.Adam(
            learning_rate=0.001,
            parameters=linear.parameters(),
        )
        x = paddle.randn([4, 2])
        out = linear(x)
        loss = out.mean()
        loss.backward()
        adam.step()
        adam.clear_grad()

    def test_adam_with_weight_decay(self):
        """Adam with weight decay."""
        linear = paddle.nn.Linear(2, 2)
        adam = opt.Adam(
            learning_rate=0.001,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
        x = paddle.randn([4, 2])
        out = linear(x)
        loss = out.mean()
        loss.backward()
        adam.step()

    def test_adam_with_beta(self):
        """Adam with custom beta1/beta2."""
        linear = paddle.nn.Linear(2, 2)
        adam = opt.Adam(
            learning_rate=0.001,
            parameters=linear.parameters(),
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        x = paddle.randn([4, 2])
        out = linear(x)
        loss = out.mean()
        loss.backward()
        adam.step()


class TestOptimizerUtils(unittest.TestCase):
    """Test optimizer utility methods."""

    def setUp(self):
        paddle.disable_static()

    def test_minimize_with_grad_clip(self):
        """minimize with gradient clipping."""
        linear = paddle.nn.Linear(2, 2)
        sgd = opt.SGD(learning_rate=0.01, parameters=linear.parameters())
        x = paddle.randn([4, 2])
        out = linear(x)
        loss = out.mean()
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        loss.backward()
        sgd.step()

    def test_lr_scheduler_with_optimizer(self):
        """Learning rate scheduler with optimizer."""
        linear = paddle.nn.Linear(2, 2)
        scheduler = paddle.optimizer.lr.StepDecay(
            learning_rate=0.01, step_size=10, gamma=0.1
        )
        sgd = opt.SGD(learning_rate=scheduler, parameters=linear.parameters())
        x = paddle.randn([4, 2])
        out = linear(x)
        loss = out.mean()
        loss.backward()
        sgd.step()
        scheduler.step()

    def test_get_lr(self):
        """Get learning rate from optimizer."""
        linear = paddle.nn.Linear(2, 2)
        sgd = opt.SGD(learning_rate=0.01, parameters=linear.parameters())
        lr = sgd.get_lr()
        self.assertAlmostEqual(lr, 0.01)

    def test_set_lr(self):
        """Set learning rate."""
        linear = paddle.nn.Linear(2, 2)
        sgd = opt.SGD(learning_rate=0.01, parameters=linear.parameters())
        sgd.set_lr(0.001)
        lr = sgd.get_lr()
        self.assertAlmostEqual(lr, 0.001)

    def test_state_dict(self):
        """SGD optimizer state_dict."""
        linear = paddle.nn.Linear(2, 2)
        sgd = opt.SGD(learning_rate=0.01, parameters=linear.parameters())
        x = paddle.randn([4, 2])
        out = linear(x)
        loss = out.mean()
        loss.backward()
        sgd.step()
        state = sgd.state_dict()
        self.assertIsInstance(state, dict)

    def test_set_state_dict(self):
        """SGD optimizer set_state_dict."""
        linear = paddle.nn.Linear(2, 2)
        sgd1 = opt.SGD(learning_rate=0.01, parameters=linear.parameters())
        x = paddle.randn([4, 2])
        out = linear(x)
        loss = out.mean()
        loss.backward()
        sgd1.step()
        state = sgd1.state_dict()
        # Create new optimizer and load state
        linear2 = paddle.nn.Linear(2, 2)
        sgd2 = opt.SGD(learning_rate=0.01, parameters=linear2.parameters())
        sgd2.set_state_dict(state)


if __name__ == '__main__':
    unittest.main()
