# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import threading
import unittest

import numpy as np

import paddle
from paddle import nn
from paddle.distributed.fleet.recompute import is_in_recompute
from paddle.distributed.fleet.recompute.recompute import _recompute_context
from paddle.distributed.fleet.utils import recompute


class TestRecomputeContext(unittest.TestCase):
    """Test _recompute_context and is_in_recompute."""

    def test_is_in_recompute_default_false(self):
        """By default is_in_recompute should return False."""
        self.assertFalse(is_in_recompute())

    def test_decorator_sets_active_true(self):
        """Inside decorated function, is_in_recompute should return True."""

        @_recompute_context
        def foo():
            self.assertTrue(is_in_recompute())
            return 42

        result = foo()
        self.assertEqual(result, 42)

    def test_decorator_resets_after_call(self):
        """After decorated function returns, is_in_recompute should be False."""

        @_recompute_context
        def foo():
            pass

        foo()
        self.assertFalse(is_in_recompute())

    def test_decorator_resets_on_exception(self):
        """Even if decorated function raises, active should be reset to False."""

        @_recompute_context
        def boom():
            self.assertTrue(is_in_recompute())
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            boom()

        self.assertFalse(is_in_recompute())

    def test_decorator_preserves_return_value(self):
        """Decorator should not alter the return value."""

        @_recompute_context
        def add(a, b):
            return a + b

        self.assertEqual(add(3, 4), 7)

    def test_decorator_passes_kwargs(self):
        """Decorator should forward *args and **kwargs correctly."""

        @_recompute_context
        def greet(name, greeting="hello"):
            return f"{greeting} {name}"

        self.assertEqual(greet("world", greeting="hi"), "hi world")

    def test_thread_isolation(self):
        """is_in_recompute should be thread-local; other threads are unaffected."""

        @_recompute_context
        def blocked():
            # This function will block until the main thread signals
            barrier.wait()
            # While inside this thread's recompute context, main thread should still be False
            results["inside_thread"] = is_in_recompute()
            barrier.wait()

        results = {}
        barrier = threading.Barrier(2, timeout=5)

        t = threading.Thread(target=blocked)
        t.start()

        # Wait until the decorated function is running (active=True in that thread)
        barrier.wait()
        # Main thread is NOT in a recompute context
        results["main_thread"] = is_in_recompute()
        # Release the child thread
        barrier.wait()
        t.join()

        self.assertTrue(results["inside_thread"])
        self.assertFalse(results["main_thread"])
        # After thread finishes, main thread still False
        self.assertFalse(is_in_recompute())

    def test_context_manager_sets_active(self):
        """Using RecomputeContext as a with-statement should set is_in_recompute."""
        from paddle.distributed.fleet.recompute.recompute import (
            _recompute_context,
        )

        self.assertFalse(_recompute_context.active)

        with _recompute_context:
            self.assertTrue(_recompute_context.active)
            self.assertTrue(is_in_recompute())

        self.assertFalse(_recompute_context.active)
        self.assertFalse(is_in_recompute())

    def test_context_manager_resets_on_exception(self):
        """Context manager should reset active even if body raises."""
        from paddle.distributed.fleet.recompute.recompute import (
            _recompute_context,
        )

        with self.assertRaises(ValueError), _recompute_context:
            self.assertTrue(_recompute_context.active)
            raise ValueError("test")

        self.assertFalse(_recompute_context.active)

    def test_backward_compat_alias(self):
        """_recompute_context should still work as a decorator."""

        @_recompute_context
        def fn():
            return is_in_recompute()

        self.assertTrue(fn())
        self.assertFalse(is_in_recompute())


class SimpleRecomputeModel(nn.Layer):
    """A simple MLP that uses recompute on the middle layer."""

    def __init__(self, input_size=16, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, use_recompute=False):
        x = self.relu(self.fc1(x))
        if use_recompute:
            x = recompute(self._middle_block, x)
        else:
            x = self._middle_block(x)
        x = self.fc3(x)
        return x

    def _middle_block(self, x):
        return self.relu(self.fc2(x))


class TestRecomputeContextWithModel(unittest.TestCase):
    """Test recompute context with a real simple model forward/backward pass."""

    def setUp(self):
        paddle.seed(42)

    def test_recompute_produces_same_loss_as_normal(self):
        """Recompute and normal forward should produce identical loss values."""
        input_size, hidden_size, batch_size = 16, 32, 4

        # Run without recompute
        paddle.seed(42)
        model_normal = SimpleRecomputeModel(input_size, hidden_size)
        x = paddle.randn([batch_size, input_size])
        loss_normal = model_normal(x, use_recompute=False).mean()

        # Run with recompute using the same weights
        paddle.seed(42)
        model_recompute = SimpleRecomputeModel(input_size, hidden_size)
        loss_recompute = model_recompute(x, use_recompute=True).mean()

        np.testing.assert_allclose(
            loss_normal.numpy(), loss_recompute.numpy(), rtol=1e-5
        )

    def test_recompute_produces_same_gradients(self):
        """Gradients with recompute should match those without."""
        input_size, hidden_size, batch_size = 16, 32, 4

        paddle.seed(42)
        model_normal = SimpleRecomputeModel(input_size, hidden_size)
        x = paddle.randn([batch_size, input_size])
        loss = model_normal(x, use_recompute=False).mean()
        loss.backward()
        grads_normal = [
            p.grad.numpy().copy()
            for p in model_normal.parameters()
            if p.grad is not None
        ]

        paddle.seed(42)
        model_recompute = SimpleRecomputeModel(input_size, hidden_size)
        loss = model_recompute(x, use_recompute=True).mean()
        loss.backward()
        grads_recompute = [
            p.grad.numpy().copy()
            for p in model_recompute.parameters()
            if p.grad is not None
        ]

        self.assertEqual(len(grads_normal), len(grads_recompute))
        for g_normal, g_recompute in zip(grads_normal, grads_recompute):
            np.testing.assert_allclose(g_normal, g_recompute, rtol=1e-5)

    def test_is_in_recompute_true_during_recompute_forward(self):
        """is_in_recompute should be True inside the recomputed function."""
        observed_states = []

        class ObservingModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                observed_states.append(is_in_recompute())
                return self.fc(x)

        model = ObservingModel()
        x = paddle.randn([2, 8])

        # Call via recompute — should observe True
        recompute(model, x)
        self.assertTrue(observed_states[-1])

        # After recompute finishes, context should be reset
        self.assertFalse(is_in_recompute())

    def test_training_step_with_recompute(self):
        """A full training step (forward + backward + optimizer) works with recompute."""
        input_size, hidden_size, batch_size = 16, 32, 4

        paddle.seed(42)
        model = SimpleRecomputeModel(input_size, hidden_size)
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=model.parameters()
        )

        initial_params = [p.numpy().copy() for p in model.parameters()]

        x = paddle.randn([batch_size, input_size])
        loss = model(x, use_recompute=True).mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        # Parameters should have been updated
        for p_init, p_updated in zip(initial_params, model.parameters()):
            self.assertFalse(
                np.array_equal(p_init, p_updated.numpy()),
                "Parameters should change after optimization step",
            )


if __name__ == "__main__":
    unittest.main()
