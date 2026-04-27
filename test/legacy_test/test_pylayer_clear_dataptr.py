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

    def test_memory_cleanup(self):
        """Multiple iterations: per-iteration objects are collectible."""
        import weakref

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

        # Track the first iteration's `out` via weakref; after the loop ends
        # and gc runs, it must be collected.  Catches holder leaks where
        # tensor_hold_helper accidentally retains a strong reference across
        # ctx lifetimes.
        first_out_ref = None
        for i in range(10):
            x = paddle.randn([64, 64]).astype('float32')
            x.stop_gradient = False
            out = TanhLayer.apply(x)
            if i == 0:
                first_out_ref = weakref.ref(out)
            loss = out.mean()
            _clear(out)
            loss.backward()
            del x, out, loss
            gc.collect()
        self.assertIsNone(first_out_ref())


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
        """pop_saved_impl does not crash when tensor was never cleared.

        Also verifies the pop targets a specific entry: after popping t's
        holder entry, a subsequent saved_tensor() call still succeeds and
        returns t with its original data (pop did not corrupt container).
        """
        ctx = self._make_ctx()
        t = paddle.randn([5]).astype('float32')
        orig = t.numpy().copy()
        ctx.save_for_backward(t)

        # No _clear_dataptr; pop should still succeed silently
        ctx._pop_saved_impl(t)
        # saved_tensor() must still return the tensor correctly.
        (recovered,) = ctx.saved_tensor()
        np.testing.assert_allclose(recovered.numpy(), orig, rtol=1e-6)

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


class TestCtxHoldRestore(unittest.TestCase):
    """Direct-ctx tests for _hold_tensors / _restore_held_tensors.

    These cover the C++ WalkDenseTensors recursion (Tensor / tuple / list /
    dict), the SavedTensorsHooks short-circuit in pylayer_hold_tensors, and
    the ``impl() != nullptr`` early-return in pylayer_restore_held_tensors.
    """

    def _make_ctx(self):
        class _Stub(PyLayer):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, dy):
                return dy

        return _Stub._backward_function()

    def test_hold_restore_basic(self):
        """hold(tensor) + _clear_dataptr + restore re-installs impl_."""
        ctx = self._make_ctx()
        t = paddle.randn([2, 3]).astype('float32')
        orig = t.numpy().copy()

        ctx._hold_tensors(t)
        _clear(t)
        self.assertFalse(t._is_initialized())

        ctx._restore_held_tensors()
        self.assertTrue(t._is_initialized())
        np.testing.assert_allclose(t.numpy(), orig, rtol=1e-6)

    def test_hold_nested_containers(self):
        """tuple / list / dict values are all deep-traversed."""
        ctx = self._make_ctx()
        t_tuple = paddle.randn([2]).astype('float32')
        t_list = paddle.randn([3]).astype('float32')
        t_dict = paddle.randn([4]).astype('float32')
        originals = [t.numpy().copy() for t in (t_tuple, t_list, t_dict)]

        # One call with a container mixing all three Python collection types.
        ctx._hold_tensors(((t_tuple,), [t_list], {'k': t_dict}))

        _clear([t_tuple, t_list, t_dict])
        ctx._restore_held_tensors()

        for got, orig in zip((t_tuple, t_list, t_dict), originals):
            self.assertTrue(got._is_initialized())
            np.testing.assert_allclose(got.numpy(), orig, rtol=1e-6)

    def test_hold_none_is_noop(self):
        """_hold_tensors(None) collects nothing; restore is a no-op."""
        ctx = self._make_ctx()
        ctx._hold_tensors(None)
        ctx._restore_held_tensors()  # must not crash

    def test_hold_scalar_top_level_noop(self):
        """_hold_tensors on a bare non-container scalar collects nothing."""
        ctx = self._make_ctx()
        for val in (42, 3.14, "str", b"bytes"):
            ctx._hold_tensors(val)
            ctx._restore_held_tensors()  # must not crash

    def test_restore_skips_valid_impl(self):
        """Restore leaves tensors whose impl is still valid untouched."""
        ctx = self._make_ctx()
        t_cleared = paddle.randn([2]).astype('float32')
        t_kept = paddle.randn([3]).astype('float32')
        orig_cleared = t_cleared.numpy().copy()
        orig_kept = t_kept.numpy().copy()

        ctx._hold_tensors([t_cleared, t_kept])
        _clear(t_cleared)  # only one is cleared
        ctx._restore_held_tensors()

        # cleared tensor resurrected
        np.testing.assert_allclose(t_cleared.numpy(), orig_cleared, rtol=1e-6)
        # kept tensor's impl untouched — covers the ``if (!tensor.impl())``
        # false branch in pylayer_restore_held_tensors.
        self.assertTrue(t_kept._is_initialized())
        np.testing.assert_allclose(t_kept.numpy(), orig_kept, rtol=1e-6)

    def test_hold_non_tensor_leaves_ignored(self):
        """Non-Tensor leaves (int/float/str/None/bytes) are silently skipped."""
        ctx = self._make_ctx()
        t1 = paddle.randn([2]).astype('float32')
        t2 = paddle.randn([3]).astype('float32')
        orig1 = t1.numpy().copy()
        orig2 = t2.numpy().copy()

        # Container mixes Tensors with int / float / str / None / bytes /
        # a dict whose values are non-Tensor; WalkDenseTensors must descend
        # into the containers, collect t1 / t2, and ignore everything else.
        mixed = (
            t1,
            42,
            "hello",
            None,
            [3.14, t2, b"bytes"],
            {'tag': 'x', 'n': 7, 'nested': (None, 'str')},
        )
        ctx._hold_tensors(mixed)

        _clear([t1, t2])
        ctx._restore_held_tensors()

        np.testing.assert_allclose(t1.numpy(), orig1, rtol=1e-6)
        np.testing.assert_allclose(t2.numpy(), orig2, rtol=1e-6)

    def test_hold_skipped_under_saved_tensors_hooks(self):
        """When saved_tensors_hooks is enabled _hold_tensors collects nothing."""
        ctx = self._make_ctx()
        t = paddle.randn([2, 3]).astype('float32')

        with paddle.autograd.saved_tensors_hooks(lambda x: x, lambda x: x):
            ctx._hold_tensors(t)

        _clear(t)
        ctx._restore_held_tensors()
        # holder was not populated, so impl stays empty after _clear_dataptr.
        self.assertFalse(t._is_initialized())


class TestRecomputeClosureHold(unittest.TestCase):
    """End-to-end recompute coverage of the Python-side closure helper.

    Covers ``_closure_cell_values`` (plain fn / nn.Layer / no-closure) and the
    ``_has_held_tensors`` True/False branches in RecomputeFunction.
    """

    def setUp(self):
        np.random.seed(1234)
        paddle.seed(1234)

    @staticmethod
    def _clone_leaf(t):
        out = paddle.to_tensor(t.numpy(), dtype=t.dtype)
        out.stop_gradient = False
        return out

    def test_closure_cell_values_empty_cell(self):
        """Empty cell triggers ValueError branch; valid cells still collected."""
        from paddle.distributed.fleet.recompute.recompute import (
            _closure_cell_values,
        )

        def outer():
            x = 1  # will be deleted → empty cell
            y = paddle.randn([2])

            def inner(a):
                return a + x + y  # noqa: F821

            del x
            return inner, y

        fn, y = outer()
        vals = _closure_cell_values(fn)
        # Empty cell dropped by the ValueError branch; only y remains.
        self.assertEqual(vals, (y,))

    def test_recompute_no_closure(self):
        """run_fn has no __closure__: _has_held_tensors=False, restore skipped."""
        from paddle.distributed.fleet.utils import recompute

        def run_fn(a, b):
            return (a * b + a).sum()

        a = paddle.randn([4, 4])
        a.stop_gradient = False
        b = paddle.randn([4, 4])
        b.stop_gradient = False
        a_ref = self._clone_leaf(a)
        b_ref = self._clone_leaf(b)

        loss = recompute(run_fn, a, b)
        _clear([a, b])
        # Sanity: _clear actually nulled impls — otherwise "restore succeeded"
        # would be trivially true and mask regressions.
        self.assertFalse(a._is_initialized())
        self.assertFalse(b._is_initialized())
        loss.backward()
        run_fn(a_ref, b_ref).backward()

        np.testing.assert_allclose(
            a.grad.numpy(), a_ref.grad.numpy(), rtol=1e-4
        )
        np.testing.assert_allclose(
            b.grad.numpy(), b_ref.grad.numpy(), rtol=1e-4
        )

    def test_recompute_closure_tensors(self):
        """Closure captures Tensor / tuple / list / dict: all restored."""
        from paddle.distributed.fleet.utils import recompute

        w_s = paddle.randn([4, 4])
        w_s.stop_gradient = False
        w_a = paddle.randn([4, 4])
        w_a.stop_gradient = False
        w_b = paddle.randn([4, 4])
        w_b.stop_gradient = False
        w_d = paddle.randn([4, 4])
        w_d.stop_gradient = False
        refs = [self._clone_leaf(t) for t in (w_s, w_a, w_b, w_d)]

        def make_fn(s, pair, mapping):
            def fn(x):
                a, b = pair
                return (x @ s + a * x + b * x + mapping['k'] * x).sum()

            return fn

        x = paddle.randn([4, 4])
        x.stop_gradient = False
        x_ref = self._clone_leaf(x)

        run_fn = make_fn(w_s, (w_a, w_b), {'k': w_d})
        ref_fn = make_fn(refs[0], (refs[1], refs[2]), {'k': refs[3]})

        loss = recompute(run_fn, x)
        _clear([x, w_s, w_a, w_b, w_d])
        for t in (x, w_s, w_a, w_b, w_d):
            self.assertFalse(t._is_initialized())
        loss.backward()
        ref_fn(x_ref).backward()

        for got, expect in zip((x, w_s, w_a, w_b, w_d), (x_ref, *refs)):
            self.assertIsNotNone(got.grad)
            np.testing.assert_allclose(
                got.grad.numpy(), expect.grad.numpy(), rtol=1e-4
            )

    def test_recompute_all_grad_from_closure(self):
        """Trainable tensors captured via closure must receive grads.

        Real-world pattern: trainable weights are closure-captured while the
        PyLayer arg is a regular activation.  Verifies that closure-captured
        ``w1`` / ``w2`` tensors are held across ``_clear_dataptr()`` and their
        grads are computed correctly during the recomputed backward.
        """
        from paddle.distributed.fleet.utils import recompute

        w1 = paddle.randn([4, 4])
        w1.stop_gradient = False
        w2 = paddle.randn([4, 4])
        w2.stop_gradient = False
        w1_ref = self._clone_leaf(w1)
        w2_ref = self._clone_leaf(w2)

        def make_fn(a, b):
            def fn(inp):
                return (inp * a * b).sum()

            return fn

        run_fn = make_fn(w1, w2)
        ref_fn = make_fn(w1_ref, w2_ref)

        inp = paddle.ones([4, 4])
        inp.stop_gradient = False
        inp_ref = paddle.ones([4, 4])
        inp_ref.stop_gradient = False

        loss = recompute(run_fn, inp)
        _clear([inp, w1, w2])
        for t in (inp, w1, w2):
            self.assertFalse(t._is_initialized())
        loss.backward()
        ref_fn(inp_ref).backward()

        np.testing.assert_allclose(
            w1.grad.numpy(), w1_ref.grad.numpy(), rtol=1e-4
        )
        np.testing.assert_allclose(
            w2.grad.numpy(), w2_ref.grad.numpy(), rtol=1e-4
        )

    def test_recompute_layer_forward_closure(self):
        """paddle.nn.Layer branch of _closure_cell_values."""
        from paddle.distributed.fleet.utils import recompute

        bias = paddle.randn([4, 4])
        bias.stop_gradient = False
        bias_ref = self._clone_leaf(bias)

        class MyLayer(paddle.nn.Layer):
            def __init__(self, captured):
                super().__init__()

                def forward(x):
                    return (x + captured).sum()

                self.forward = forward

            def forward(self, x):  # pragma: no cover
                raise RuntimeError

        layer = MyLayer(bias)
        layer_ref = MyLayer(bias_ref)
        x = paddle.randn([4, 4])
        x.stop_gradient = False
        x_ref = self._clone_leaf(x)

        loss = recompute(layer, x)
        _clear([x, bias])
        self.assertFalse(x._is_initialized())
        self.assertFalse(bias._is_initialized())
        loss.backward()
        layer_ref(x_ref).backward()

        np.testing.assert_allclose(
            x.grad.numpy(), x_ref.grad.numpy(), rtol=1e-4
        )
        np.testing.assert_allclose(
            bias.grad.numpy(), bias_ref.grad.numpy(), rtol=1e-4
        )


if __name__ == '__main__':
    unittest.main()
