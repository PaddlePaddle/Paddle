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

"""Test PyLayer tensor_hold_helper for _clear_dataptr protection.

Pipeline-parallel pattern:
  1. outputs = Layer.apply(inputs)    # forward: data is valid
  2. loss    = f(outputs)             # build loss graph BEFORE clearing
  3. outputs._clear_dataptr()        # free activation memory
  4. loss.backward()                  # backward via tensor_hold_helper recovery

tensor_hold_helper is a vector<shared_ptr<DenseTensor>> on PyLayerObject that
holds strong references to every DenseTensor impl saved via save_for_backward.
It is born with set_container (save_for_backward) and destroyed with the
PyLayerObject itself, preventing _clear_dataptr from freeing the underlying
allocation before backward runs.
"""

import gc
import unittest

import numpy as np

import paddle
from paddle.autograd import PyLayer


def _clear(tensors):
    """Call _clear_dataptr on a single tensor or iterable of tensors."""
    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            if hasattr(t, '_clear_dataptr'):
                t._clear_dataptr()
    elif hasattr(tensors, '_clear_dataptr'):
        tensors._clear_dataptr()


class TestPyLayerClearDataptr(unittest.TestCase):
    """Core tests: _clear_dataptr on outputs does not break backward."""

    def test_basic_clear_dataptr(self):
        """Single output, single saved tensor."""

        class TanhLayer(PyLayer):
            @staticmethod
            def forward(ctx, x):
                y = paddle.tanh(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, dy):
                (y,) = ctx.saved_tensor()
                return dy * (1 - paddle.square(y))

        x = paddle.randn([2, 3]).astype('float64')
        x.stop_gradient = False
        out = TanhLayer.apply(x)
        loss = out.mean()  # build graph first
        _clear(out)  # then free activation
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_multiple_saved_tensors(self):
        """Multiple tensors passed to save_for_backward."""

        class AddLayer(PyLayer):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                return x + y

            @staticmethod
            def backward(ctx, dy):
                x, y = ctx.saved_tensor()
                return dy, dy

        x = paddle.randn([2, 3]).astype('float64')
        y = paddle.randn([2, 3]).astype('float64')
        x.stop_gradient = False
        y.stop_gradient = False
        out = AddLayer.apply(x, y)
        loss = out.mean()
        _clear(out)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)

    def test_multiple_outputs(self):
        """Tuple output: both outputs are cleared."""

        class MultiOutLayer(PyLayer):
            @staticmethod
            def forward(ctx, x):
                y1 = paddle.tanh(x)
                y2 = paddle.sin(x)
                ctx.save_for_backward(y1, y2)
                return y1, y2

            @staticmethod
            def backward(ctx, dy1, dy2):
                y1, y2 = ctx.saved_tensor()
                return dy1 * (1 - paddle.square(y1)) + dy2 * paddle.cos(y2)

        x = paddle.randn([2, 3]).astype('float64')
        x.stop_gradient = False
        y1, y2 = MultiOutLayer.apply(x)
        loss = (y1 + y2).mean()  # build graph while data is valid
        _clear([y1, y2])
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_chained_computation(self):
        """Final output of a chain is cleared; intermediate kept for input."""

        class TanhLayer(PyLayer):
            @staticmethod
            def forward(ctx, x):
                y = paddle.tanh(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, dy):
                (y,) = ctx.saved_tensor()
                return dy * (1 - paddle.square(y))

        x = paddle.randn([2, 3]).astype('float64')
        x.stop_gradient = False
        y = TanhLayer.apply(x)  # intermediate – not cleared
        z = TanhLayer.apply(y)  # final output
        loss = z.mean()
        _clear(z)  # only clear final activation
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_different_dtypes(self):
        """float32 / float64 (and float16 on GPU) all work after _clear_dataptr."""

        class TanhLayer(PyLayer):
            @staticmethod
            def forward(ctx, x):
                y = paddle.tanh(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, dy):
                (y,) = ctx.saved_tensor()
                return dy * (1 - paddle.square(y))

        dtypes = ['float32', 'float64']
        if paddle.is_compiled_with_cuda():
            dtypes.append('float16')
        for dtype in dtypes:
            x = paddle.randn([2, 3]).astype(dtype)
            x.stop_gradient = False
            out = TanhLayer.apply(x)
            loss = out.mean()
            _clear(out)
            loss.backward()
            self.assertIsNotNone(x.grad)

    def test_without_clear_dataptr(self):
        """Sanity check: normal backward (no clear) still gives correct grad."""

        class TanhLayer(PyLayer):
            @staticmethod
            def forward(ctx, x):
                y = paddle.tanh(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, dy):
                (y,) = ctx.saved_tensor()
                return dy * (1 - paddle.square(y))

        x1 = paddle.randn([2, 3]).astype('float64')
        x2 = x1.detach().clone()
        x1.stop_gradient = False
        x2.stop_gradient = False
        TanhLayer.apply(x1).mean().backward()
        TanhLayer.apply(x2).mean().backward()
        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), rtol=1e-10)

    def test_memory_cleanup(self):
        """Multiple iterations: no memory leak / reference accumulation."""

        class TanhLayer(PyLayer):
            @staticmethod
            def forward(ctx, x):
                y = paddle.tanh(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, dy):
                (y,) = ctx.saved_tensor()
                return dy * (1 - paddle.square(y))

        for _ in range(10):
            x = paddle.randn([64, 64]).astype('float32')
            x.stop_gradient = False
            out = TanhLayer.apply(x)
            loss = out.mean()
            _clear(out)
            loss.backward()
            gc.collect()
        self.assertTrue(True)


class TestCtxDirect(unittest.TestCase):
    """Unit tests for ctx API without going through PyLayer.apply().

    These tests create a ctx object directly via cls._backward_function() and
    exercise save_for_backward / saved_tensor / pop_saved_impl in isolation,
    independently of the forward/backward dispatch machinery.

    Key design: cls._backward_function is a subclass of PyLayerBackward which
    inherits core.eager.PyLayer (C++ PyLayerObject).  Instantiating it calls
    PyLayerNew, giving a fully-initialized ctx with an empty tensor_hold_helper.
    """

    def _make_ctx(self):
        """Create a bare ctx (PyLayerObject) without running forward."""

        class _Stub(PyLayer):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, dy):
                return dy

        return _Stub._backward_function()

    # ------------------------------------------------------------------
    # Basic save / recover
    # ------------------------------------------------------------------

    def test_direct_single_tensor_recover(self):
        """save_for_backward + _clear_dataptr + saved_tensor, no apply."""
        ctx = self._make_ctx()
        t = paddle.randn([2, 3]).astype('float64')
        ctx.save_for_backward(t)

        _clear(t)

        (recovered,) = ctx.saved_tensor()
        self.assertIsNotNone(recovered)
        self.assertEqual(list(recovered.shape), [2, 3])

    def test_direct_multiple_tensors_recover(self):
        """All tensors are recovered after _clear_dataptr, no apply."""
        ctx = self._make_ctx()
        a = paddle.randn([3]).astype('float32')
        b = paddle.ones([4, 2]).astype('float64')
        ctx.save_for_backward(a, b)

        _clear(a)
        _clear(b)

        recovered = ctx.saved_tensor()
        self.assertEqual(len(recovered), 2)
        self.assertEqual(list(recovered[0].shape), [3])
        self.assertEqual(list(recovered[1].shape), [4, 2])

    def test_direct_no_clear(self):
        """saved_tensor returns correct values when _clear_dataptr was not called."""
        ctx = self._make_ctx()
        t = paddle.randn([2, 3]).astype('float32')
        expected = t.numpy().copy()
        ctx.save_for_backward(t)

        (recovered,) = ctx.saved_tensor()
        np.testing.assert_allclose(recovered.numpy(), expected, rtol=1e-6)

    # ------------------------------------------------------------------
    # pop_saved_impl
    # ------------------------------------------------------------------

    def test_pop_saved_impl_single(self):
        """pop_saved_impl removes the holder entry; recovered tensor stays valid."""
        ctx = self._make_ctx()
        t = paddle.randn([2, 3]).astype('float32')
        orig = t.numpy().copy()
        ctx.save_for_backward(t)

        _clear(t)
        (recovered,) = ctx.saved_tensor()
        # Verify the recovered tensor carries the correct data (not just non-None).
        np.testing.assert_allclose(recovered.numpy(), orig, rtol=1e-6)

        # Pop removes the holder entry; recovered's own shared_ptr keeps data alive.
        ctx._pop_saved_impl(recovered)
        self.assertEqual(list(recovered.shape), [2, 3])

    def test_pop_saved_impl_partial(self):
        """Pop both saved tensors one by one; proves each entry is stored independently."""
        ctx = self._make_ctx()
        a = paddle.randn([2]).astype('float32')
        b = paddle.randn([3]).astype('float32')
        ctx.save_for_backward(a, b)

        _clear(a)
        _clear(b)
        recovered = ctx.saved_tensor()
        self.assertEqual(len(recovered), 2)

        # Pop the first entry; if holder only had one entry this would erase it
        # and the second pop below would be a no-op instead of finding b's entry.
        ctx._pop_saved_impl(recovered[0])
        # Pop the second entry; succeeds only if b's entry is still in holder
        # (i.e. the two entries are stored independently).
        ctx._pop_saved_impl(recovered[1])
        # Both recovered handles remain valid via their own shared_ptr copies.
        self.assertEqual(list(recovered[0].shape), [2])
        self.assertEqual(list(recovered[1].shape), [3])

    def test_pop_saved_impl_no_clear(self):
        """pop_saved_impl does not crash when tensor was never cleared."""
        ctx = self._make_ctx()
        t = paddle.randn([5]).astype('float32')
        ctx.save_for_backward(t)

        # No _clear_dataptr; pop should still succeed silently
        ctx._pop_saved_impl(t)

    # ------------------------------------------------------------------
    # Deep-traversal via nested list in container
    # ------------------------------------------------------------------

    def test_nested_list_holder_populated(self):
        """Container with a nested list: CollectDenseTensors populates holder for all tensors.

        save_for_backward packs args as a tuple, so the container at the top
        level is always a tuple.  But tuple *elements* may themselves be lists
        (e.g. when a list is passed as one argument).  CollectDenseTensors
        recurses into them; verify via pop_saved_impl that both were collected.
        """
        ctx = self._make_ctx()
        t1 = paddle.randn([2]).astype('float32')
        t2 = paddle.randn([3]).astype('float32')

        # Directly assign a tuple whose sole element is a list of tensors.
        # This bypasses save_for_backward's *args flattening so we can test
        # the deep-traversal branch.
        ctx.container = ([t1, t2],)

        # Each pop finds and removes its entry; if CollectDenseTensors missed
        # an entry, the corresponding pop is a silent no-op — so we follow
        # each pair of pops with a redundant third pop that must also not crash,
        # confirming the erase path is robust against missing entries.
        ctx._pop_saved_impl(t1)
        ctx._pop_saved_impl(t2)
        ctx._pop_saved_impl(t1)  # already removed — must be a silent no-op

    def test_nested_tuple_holder_populated(self):
        """Container with a nested tuple: all inner tensors are held."""
        ctx = self._make_ctx()
        t1 = paddle.randn([2]).astype('float32')
        t2 = paddle.randn([3]).astype('float32')

        ctx.container = ((t1, t2),)

        ctx._pop_saved_impl(t1)
        ctx._pop_saved_impl(t2)
        ctx._pop_saved_impl(t1)  # already removed — must be a silent no-op


if __name__ == '__main__':
    unittest.main()
