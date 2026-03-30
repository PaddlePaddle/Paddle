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
from paddle.distributed.fleet.recompute.recompute import _protect_tensors
from paddle.distributed.fleet.recompute.recompute_hybrid import recompute_hybrid
from paddle.distributed.fleet.utils import recompute
from paddle.framework import core


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


class TestProtectTensors(unittest.TestCase):
    """Unit tests for _protect_tensors(), covering all code branches."""

    def test_returns_list(self):
        """_protect_tensors always returns a list regardless of input type."""
        t = paddle.rand([3, 4])
        result = _protect_tensors((t,))
        self.assertIsInstance(result, list)

    def test_plain_tensor_new_python_object(self):
        """Plain tensor input: result element must be a different Python object."""
        t = paddle.rand([3, 4])
        result = _protect_tensors([t])
        self.assertIsNot(result[0], t)
        self.assertIsInstance(result[0], core.eager.Tensor)

    def test_plain_tensor_shares_data(self):
        """Plain tensor input: protected copy shares the same underlying data."""
        t = paddle.rand([3, 4])
        result = _protect_tensors([t])
        np.testing.assert_array_equal(result[0].numpy(), t.numpy())

    def test_multiple_plain_tensors(self):
        """Multiple plain tensor inputs are all individually protected."""
        t1 = paddle.rand([2, 3])
        t2 = paddle.rand([4, 5])
        result = _protect_tensors([t1, t2])
        self.assertEqual(len(result), 2)
        self.assertIsNot(result[0], t1)
        self.assertIsNot(result[1], t2)
        np.testing.assert_array_equal(result[0].numpy(), t1.numpy())
        np.testing.assert_array_equal(result[1].numpy(), t2.numpy())

    def test_tuple_tensor_elements_protected(self):
        """Tuple arg: each tensor element inside the tuple is a new Python object."""
        t1 = paddle.rand([3, 4])
        t2 = paddle.rand([3, 4])
        result = _protect_tensors([(t1, t2)])
        self.assertIsInstance(result[0], tuple)
        self.assertIsNot(result[0][0], t1)
        self.assertIsNot(result[0][1], t2)
        np.testing.assert_array_equal(result[0][0].numpy(), t1.numpy())
        np.testing.assert_array_equal(result[0][1].numpy(), t2.numpy())

    def test_tuple_non_tensor_elements_passthrough(self):
        """Non-tensor elements inside a tuple arg are passed through unchanged."""
        t = paddle.rand([3, 4])
        mask = True
        idx = 42
        result = _protect_tensors([(t, mask, idx)])
        self.assertIs(result[0][1], mask)
        self.assertIs(result[0][2], idx)

    def test_non_tensor_non_tuple_passthrough(self):
        """Non-tensor, non-tuple elements (e.g. int, bool, None) are passed through unchanged."""
        non_tensors = [1, True, None, "string", 3.14]
        result = _protect_tensors(non_tensors)
        for orig, got in zip(non_tensors, result):
            self.assertIs(got, orig)

    def test_mixed_seq(self):
        """Mixed sequence: tensors are protected, non-tensors pass through."""
        t = paddle.rand([2, 2])
        scalar = 5
        result = _protect_tensors([t, scalar])
        self.assertIsNot(result[0], t)
        np.testing.assert_array_equal(result[0].numpy(), t.numpy())
        self.assertIs(result[1], scalar)

    def test_empty_seq(self):
        """Empty sequence returns empty list."""
        result = _protect_tensors([])
        self.assertEqual(result, [])

    def test_pipeline_release_simulation(self):
        """Simulate pipeline-parallel tensor release: after clearing the data
        pointer of the original tensor, the protected copy should still be valid
        and hold the original data values.
        """
        data = np.random.rand(4, 4).astype('float32')
        original = paddle.to_tensor(data)
        result = _protect_tensors([original])
        protected = result[0]

        # The protected tensor must be a different Python object.
        self.assertIsNot(protected, original)

        # Simulate pipeline-parallel releasing the original tensor's data.
        # _clear_dataptr() is the C++ method called by _release_input/output.
        if hasattr(original, '_clear_dataptr'):
            original._clear_dataptr()
            # Protected copy must still hold valid data.
            np.testing.assert_array_equal(protected.numpy(), data)
        else:
            # If _clear_dataptr() is not available in this build, at minimum
            # verify the protected copy holds the correct data.
            np.testing.assert_array_equal(protected.numpy(), data)

    def test_recompute_uses_protect_tensors_plain(self):
        """End-to-end: recompute() with plain tensor args should complete
        forward and backward correctly (exercises _protect_tensors fast path).
        """
        linear = paddle.nn.Linear(8, 8)
        x = paddle.rand([4, 8])
        x.stop_gradient = False
        out = recompute(linear, x)
        loss = out.mean()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_recompute_uses_protect_tensors_tuple(self):
        """End-to-end: recompute() with a tuple tensor arg exercises the tuple
        branch of _protect_tensors and should complete forward/backward correctly.
        """

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
        loss = out.mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_recompute_uses_protect_tensors_with_kwargs(self):
        """End-to-end: recompute() with kwargs triggers the slow path which
        also calls _protect_tensors(input_args) before RecomputeFunction.apply().
        """

        class KwargsLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.linear = paddle.nn.Linear(8, 8)

            def forward(self, x, scale=1.0):
                return self.linear(x) * scale

        layer = KwargsLayer()
        x = paddle.rand([4, 8])
        x.stop_gradient = False
        # passing 'scale' as a kwarg forces the slow path in recompute()
        out = recompute(layer, x, scale=2.0)
        loss = out.mean()
        loss.backward()
        self.assertIsNotNone(x.grad)


class _MockMpGroup:
    """Minimal mock of a model-parallel group for single-GPU unit tests.
    Only used when partition=False, so nranks/rank are never actually accessed.
    """

    nranks = 1
    rank = 0


class TestRecomputeHybridProtectTensors(unittest.TestCase):
    """Tests that _protect_tensors is exercised inside _HPRecomputeFunction.forward()
    via recompute_hybrid().  Uses a MockMpGroup so no distributed init is needed,
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
        """recompute_hybrid with multiple tensor inputs: _protect_tensors
        processes each arg in the args tuple.
        """

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


if __name__ == '__main__':
    unittest.main()
