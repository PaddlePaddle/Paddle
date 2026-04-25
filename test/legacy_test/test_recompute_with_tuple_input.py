# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distributed.fleet.recompute.recompute_hybrid import recompute_hybrid
from paddle.distributed.fleet.utils import recompute


class Layer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = paddle.nn.Linear(10, 10)
        self.linear2 = paddle.nn.Linear(10, 10)
        self.linear3 = paddle.nn.Linear(10, 10)
        self.silu1 = paddle.nn.Silu()
        self.silu2 = paddle.nn.Silu()
        self.silu3 = paddle.nn.Silu()

    def forward(self, x, y):
        assert type(x) is tuple
        assert len(x) == 2
        o1 = self.silu1(self.linear1(x[0]))
        o2 = self.silu2(self.linear2(x[1]))
        o3 = self.silu3(self.linear3(y))
        o = o1 + o2 + o3
        return o


class TestPyLayer(unittest.TestCase):
    def test_tuple_input(self):
        layer = Layer()
        x1 = paddle.rand(shape=[10, 10])
        x1.stop_gradient = False
        x2 = paddle.rand(shape=[10, 10])
        x2.stop_gradient = False
        y = paddle.rand(shape=[10, 10])
        y.stop_gradient = False
        o = recompute(layer, (x1, x2), y)
        loss = paddle.mean(o, keepdim=True)
        loss.backward()

    def test_tuple_input_with_non_tensor(self):
        layer = Layer()
        x1 = paddle.rand(shape=[10, 10])
        x1.stop_gradient = False
        y = paddle.rand(shape=[10, 10])
        y.stop_gradient = False
        try:
            o = recompute(layer, (x1, True), y)
        except ValueError:
            pass

    def test_tuple_input_with_different_stop_gradient(self):
        layer = Layer()
        x1 = paddle.rand(shape=[10, 10])
        x1.stop_gradient = False
        x2 = paddle.rand(shape=[10, 10])
        y = paddle.rand(shape=[10, 10])
        y.stop_gradient = False
        try:
            o = recompute(layer, (x1, True), y)
        except ValueError:
            pass

    def test_tuple_input_all_no_gradient(self):
        layer = Layer()
        x1 = paddle.rand(shape=[10, 10])
        x2 = paddle.rand(shape=[10, 10])
        y = paddle.rand(shape=[10, 10])
        y.stop_gradient = False
        o = recompute(layer, (x1, x2), y)
        loss = paddle.mean(o, keepdim=True)
        loss.backward()


class _MockMpGroup:
    """Minimal mock of a model-parallel group for single-GPU unit tests.
    Only used when partition=False, so nranks/rank are never actually accessed.
    """

    nranks = 1
    rank = 0


class TestRecomputeHybridProtectTensors(unittest.TestCase):
    """Tests for recompute_hybrid() with various tensor inputs.
    Uses a MockMpGroup so no distributed init is needed,
    and partition=False so mp_group is never actually invoked.
    """

    @classmethod
    def setUpClass(cls):
        if not paddle.is_compiled_with_cuda():
            raise unittest.SkipTest("Requires GPU")

    def test_forward_backward_plain_tensor(self):
        """recompute_hybrid with a plain tensor input completes correctly."""
        linear = paddle.nn.Linear(8, 8)
        x = paddle.rand([4, 8])
        x.stop_gradient = False
        ctx = {'mp_group': _MockMpGroup(), 'offload': False, 'partition': False}
        out = recompute_hybrid(ctx, linear, x)
        out.mean().backward()
        self.assertIsNotNone(x.grad)

    def test_forward_backward_multiple_tensors(self):
        """recompute_hybrid with multiple tensor inputs completes correctly."""

        class TwoInputLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.linear = paddle.nn.Linear(8, 8)

            def forward(self, x, y):
                return self.linear(x) + self.linear(y)

        layer = TwoInputLayer()
        x = paddle.rand([4, 8])
        x.stop_gradient = False
        y = paddle.rand([4, 8])
        y.stop_gradient = False
        ctx = {'mp_group': _MockMpGroup(), 'offload': False, 'partition': False}
        out = recompute_hybrid(ctx, layer, x, y)
        out.mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)


class TestRecomputeWithClosureTensors(unittest.TestCase):
    """End-to-end tests for recompute() with closure-captured tensors."""

    # ------------------------------------------------------------------
    # basic: plain function with a closure tensor, no release
    # ------------------------------------------------------------------

    def test_plain_function_with_closure_tensor(self):
        """recompute() on a plain function that captures a tensor in its
        closure must complete forward and backward correctly."""
        grid = paddle.rand([4, 8])

        def fn(x):
            return x * grid

        x = paddle.rand([4, 8])
        x.stop_gradient = False
        out = recompute(fn, x)
        out.mean().backward()
        self.assertIsNotNone(x.grad)

    # ------------------------------------------------------------------
    # basic: no closure at all
    # ------------------------------------------------------------------

    def test_function_without_closure(self):
        """recompute() on a function that has no closure must not raise."""

        def simple_fn(x):
            return x * 2.0

        x = paddle.rand([4, 8])
        x.stop_gradient = False
        out = recompute(simple_fn, x)
        out.mean().backward()
        self.assertIsNotNone(x.grad)

    # ------------------------------------------------------------------
    # basic: closure with only non-tensor values
    # ------------------------------------------------------------------

    def test_function_with_non_tensor_closure(self):
        """Closure holding only non-tensor values must be handled safely."""
        scale = 3.0

        def fn(x):
            return x * scale

        x = paddle.rand([4, 8])
        x.stop_gradient = False
        out = recompute(fn, x)
        out.mean().backward()
        self.assertIsNotNone(x.grad)

    # ------------------------------------------------------------------
    # basic: nn.Layer (uses run_function.forward.__closure__)
    # ------------------------------------------------------------------

    def test_layer_with_closure_in_forward(self):
        """recompute() on an nn.Layer that captures a tensor in forward's
        closure must complete forward and backward correctly."""

        class ClosureLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.linear = paddle.nn.Linear(8, 8)
                mask = paddle.ones([4, 8])

                # Define forward as a closure over `mask`
                def _forward_impl(x):
                    return self.linear(x) * mask

                self._forward_impl = _forward_impl

            def forward(self, x):
                return self._forward_impl(x)

        layer = ClosureLayer()
        x = paddle.rand([4, 8])
        x.stop_gradient = False
        out = recompute(layer, x)
        out.mean().backward()
        self.assertIsNotNone(x.grad)

    # ------------------------------------------------------------------
    # gradient correctness with closure tensor
    # ------------------------------------------------------------------

    def test_gradient_correctness_with_closure_tensor(self):
        """Gradients computed via recompute (with closure tensor) must match
        those computed without recompute."""
        paddle.seed(42)
        grid = paddle.rand([4, 8])
        linear = paddle.nn.Linear(8, 8)

        def fn(x):
            return linear(x) + grid

        x = paddle.rand([4, 8])
        x.stop_gradient = False

        # reference: no recompute
        out_ref = fn(x)
        loss_ref = out_ref.mean()
        loss_ref.backward()
        grad_ref = x.grad.numpy().copy()
        x.clear_gradient()

        # with recompute
        out_rc = recompute(fn, x)
        loss_rc = out_rc.mean()
        loss_rc.backward()
        np.testing.assert_allclose(x.grad.numpy(), grad_ref, rtol=1e-5)

    def test_non_reentrant_with_closure_tensor(self):
        """use_reentrant=False path with a closure-captured tensor must
        complete forward and backward correctly."""
        grid = paddle.rand([4, 8])

        def fn(x):
            return x * grid

        x = paddle.rand([4, 8])
        x.stop_gradient = False
        out = recompute(fn, x, use_reentrant=False)
        out.mean().backward()
        self.assertIsNotNone(x.grad)

    def test_closure_tensor_preserve_rng_state_false(self):
        """recompute() with preserve_rng_state=False and a closure tensor must
        execute the else-branch in backward (L441), completing forward and
        backward correctly."""
        grid = paddle.rand([4, 8])

        def fn(x):
            return x * grid

        x = paddle.rand([4, 8])
        x.stop_gradient = False
        out = recompute(fn, x, preserve_rng_state=False)
        out.mean().backward()
        self.assertIsNotNone(x.grad)


class TestRecomputeRegression(unittest.TestCase):
    """Regression tests for recompute() basic scenarios.

    These cover the plain-tensor, tuple-only-input, and keyword-argument
    code paths and serve as a safety net against regressions regardless of
    the internal protection mechanism used.
    """

    def test_plain_tensor(self):
        """recompute() with a plain tensor arg completes forward/backward."""
        linear = paddle.nn.Linear(8, 8)
        x = paddle.rand([4, 8])
        x.stop_gradient = False
        out = recompute(linear, x)
        out.mean().backward()
        self.assertIsNotNone(x.grad)

    def test_tuple_only_input(self):
        """recompute() with a tuple-only tensor arg completes forward/backward."""

        class TupleInputLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.linear = paddle.nn.Linear(8, 8)

            def forward(self, xy):
                x, y = xy
                return self.linear(x) + self.linear(y)

        layer = TupleInputLayer()
        x = paddle.rand([4, 8])
        x.stop_gradient = False
        y = paddle.rand([4, 8])
        y.stop_gradient = False
        out = recompute(layer, (x, y))
        out.mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_with_kwargs(self):
        """recompute() with extra keyword arguments completes forward/backward."""

        class ScaledLinear(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.linear = paddle.nn.Linear(8, 8)

            def forward(self, x, scale=1.0):
                return self.linear(x) * scale

        layer = ScaledLinear()
        x = paddle.rand([4, 8])
        x.stop_gradient = False
        out = recompute(layer, x, scale=2.0)
        out.mean().backward()
        self.assertIsNotNone(x.grad)


if __name__ == '__main__':
    unittest.main()
