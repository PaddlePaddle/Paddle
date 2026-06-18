# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import pathlib
import shutil
import unittest

import numpy as np
from op_test import is_custom_device

import paddle
from paddle.device.cuda.graphs import CUDAGraph


def can_use_cuda_graph():
    return (
        paddle.is_compiled_with_cuda() or is_custom_device()
    ) and not paddle.is_compiled_with_rocm()


@unittest.skipIf(
    not (paddle.is_compiled_with_cuda() or is_custom_device())
    or float(paddle.version.cuda()) < 11.0,
    "only support cuda >= 11.0",
)
class TestCUDAGraphInDygraphMode(unittest.TestCase):
    def setUp(self):
        if can_use_cuda_graph():
            paddle.set_flags(
                {
                    'FLAGS_allocator_strategy': 'auto_growth',
                    'FLAGS_sync_nccl_allreduce': False,
                    'FLAGS_cudnn_deterministic': True,
                    'FLAGS_use_stream_safe_cuda_allocator': False,
                }
            )

    def random_tensor(self, shape):
        return paddle.to_tensor(
            np.random.randint(low=0, high=10, size=shape).astype("float32")
        )

    def test_cuda_graph_dynamic_graph(self):
        if not can_use_cuda_graph():
            return

        shape = [2, 3]
        x = self.random_tensor(shape)
        z = self.random_tensor(shape)

        g = CUDAGraph()
        g.capture_begin()
        y = x + 10
        b = x.numel()
        z.add_(x)
        g.capture_end()

        for _ in range(10):
            z_np_init = z.numpy()
            x_new = self.random_tensor(shape)
            x.copy_(x_new, False)
            g.replay()
            x_np = x_new.numpy()
            y_np = y.numpy()
            z_np = z.numpy()
            self.assertTrue((y_np - x_np == 10).all())
            self.assertTrue((z_np - z_np_init == x_np).all())

        g.reset()

    def test_concat_and_split(self):
        if not can_use_cuda_graph():
            return

        concat_num = 100
        xs = []
        xs_np = []

        for i in range(concat_num):
            x_np = np.random.random(size=[1]).astype(np.float32)
            xs.append(paddle.to_tensor(x_np))
            xs_np.append(x_np)

        graph = CUDAGraph()
        graph.capture_begin()
        y = paddle.concat(xs)
        zs = paddle.split(y, len(xs))
        graph.capture_end()
        graph.replay()

        y_np = y.numpy()
        y_np_expected = np.concatenate(xs_np)
        np.testing.assert_array_equal(y_np, y_np_expected)
        self.assertEqual(len(zs), len(xs_np))
        for i, z in enumerate(zs):
            np.testing.assert_array_equal(z.numpy(), xs_np[i])

        output_dir = f'cuda_graph_dot_{os.getpid()}'
        try:
            graph.print_to_dot_files(pathlib.Path(output_dir))
            graph.reset()
            shutil.rmtree(output_dir)
        except Exception as e:
            msg = str(e)
            sub_msg = "The print_to_dot_files() method is only supported when CUDA version >= 11.3"
            self.assertTrue(sub_msg in msg)
        finally:
            graph.reset()

    def test_dataloader(self):
        if not can_use_cuda_graph():
            return

        class AutoIncDataset(paddle.io.Dataset):
            def __init__(self, n, dtype):
                self.n = n
                self.dtype = dtype

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return np.array([idx]).astype(self.dtype)

        n = 100
        dtype = 'int64'
        dataset = AutoIncDataset(n, dtype)
        data_loader = paddle.io.DataLoader(
            dataset, batch_size=1, num_workers=2, use_buffer_reader=True
        )
        x = None
        y = None

        graph = None
        for i, data in enumerate(data_loader):
            if graph is None:
                x = data
                x = x.cuda()
                graph = CUDAGraph()
                graph.capture_begin()
                y = x * x
                graph.capture_end()
            else:
                x.copy_(data, False)
                x = x.cuda()

            graph.replay()
            actual_x = np.array([[i]]).astype(dtype)
            actual_y = np.array([[i * i]]).astype(dtype)
            np.testing.assert_array_equal(actual_x, x.numpy())
            np.testing.assert_array_equal(actual_y, y.numpy())

    def test_dev_ctx_alloc(self):
        if not can_use_cuda_graph():
            return

        x = paddle.to_tensor([2], dtype='float32')
        graph = CUDAGraph()
        graph.capture_begin()
        y = paddle.cast(x, dtype='float16')
        graph.capture_end()

    def test_cuda_graph_with_enable_replace(self):
        """Test that CUDAGraph created with enable_replace=True captures and replays correctly."""
        if not can_use_cuda_graph():
            return

        shape = [4, 4]
        x = self.random_tensor(shape)
        x_val = x.numpy().copy()

        g = CUDAGraph(enable_replace=True)
        g.capture_begin()
        y = x * 2.0
        g.capture_end()

        g.replay()
        np.testing.assert_allclose(y.numpy(), x_val * 2.0, rtol=1e-5)
        g.reset()

    def test_replace_input_ptrs(self):
        """Test replace_input_ptrs exercises CacheKernelNodeInfos and ReplaceInputPtrs code paths."""
        if not can_use_cuda_graph():
            return

        shape = [4, 4]
        x = self.random_tensor(shape)
        x_val = x.numpy().copy()

        g = CUDAGraph(enable_replace=True)
        g.capture_begin()
        y = x * 2.0
        g.capture_end()

        # First replay: result should match x * 2
        g.replay()
        np.testing.assert_allclose(y.numpy(), x_val * 2.0, rtol=1e-5)

        # Create a new input buffer and replace the pointer
        x_new = self.random_tensor(shape)
        x_new_val = x_new.numpy().copy()
        old_ptr = x.data_ptr()
        new_ptr = x_new.data_ptr()

        # Should not raise; exercises ReplaceInputPtrs
        g.replace_input_ptrs([old_ptr], [new_ptr])
        g.replay()

        # On CUDA >= 12.4 the replacement is effective; on older CUDA
        # GetKernelParamInfos returns empty so replacement is a no-op.
        # Either way we validate no crash and replay succeeds.
        result = y.numpy()
        cuda_ver = (
            float(paddle.version.cuda())
            if paddle.version.cuda() != 'False'
            else 0.0
        )
        if cuda_ver >= 12.4:
            np.testing.assert_allclose(result, x_new_val * 2.0, rtol=1e-5)

        g.reset()

    def test_replace_input_ptrs_requires_enable_replace(self):
        """Test that replace_input_ptrs raises an error when enable_replace=False."""
        if not can_use_cuda_graph():
            return

        shape = [2, 3]
        x = self.random_tensor(shape)

        g = CUDAGraph()  # enable_replace defaults to False
        g.capture_begin()
        y = x + 1.0
        g.capture_end()

        raised = False
        try:
            g.replace_input_ptrs([x.data_ptr()], [x.data_ptr()])
        except Exception:
            raised = True
        finally:
            g.reset()

        self.assertTrue(raised, "Expected exception when enable_replace=False")

    def test_replace_input_ptrs_empty(self):
        """Test replace_input_ptrs with empty pointer lists (no-op path)."""
        if not can_use_cuda_graph():
            return

        shape = [2, 3]
        x = self.random_tensor(shape)

        g = CUDAGraph(enable_replace=True)
        g.capture_begin()
        y = x + 1.0
        g.capture_end()

        # Calling with empty lists should be a no-op and not raise
        g.replace_input_ptrs([], [])
        g.replay()
        np.testing.assert_allclose(y.numpy(), x.numpy() + 1.0, rtol=1e-5)
        g.reset()

    def test_replace_input_ptrs_no_match(self):
        """Test replace_input_ptrs when old_ptr does not match any kernel param (no modification)."""
        if not can_use_cuda_graph():
            return

        shape = [2, 3]
        x = self.random_tensor(shape)
        x_val = x.numpy().copy()

        g = CUDAGraph(enable_replace=True)
        g.capture_begin()
        y = x + 1.0
        g.capture_end()

        g.replay()

        # Pass a dummy pointer that won't match anything
        dummy_ptr = 0
        g.replace_input_ptrs([dummy_ptr], [dummy_ptr])
        g.replay()

        # Output should remain unchanged from original x
        np.testing.assert_allclose(y.numpy(), x_val + 1.0, rtol=1e-5)
        g.reset()

    def test_replace_multiple_input_ptrs(self):
        """Test replacing multiple input tensor pointers simultaneously.
        Exercises the j-loop over multiple old_ptrs entries in ReplaceInputPtrs.
        """
        if not can_use_cuda_graph():
            return

        shape = [4, 4]
        x = self.random_tensor(shape)
        w = self.random_tensor(shape)
        x_val = x.numpy().copy()
        w_val = w.numpy().copy()

        g = CUDAGraph(enable_replace=True)
        g.capture_begin()
        y = x + w
        g.capture_end()

        g.replay()
        np.testing.assert_allclose(y.numpy(), x_val + w_val, rtol=1e-5)

        x_new = self.random_tensor(shape)
        w_new = self.random_tensor(shape)
        x_new_val = x_new.numpy().copy()
        w_new_val = w_new.numpy().copy()

        # Replace both input pointers at once
        g.replace_input_ptrs(
            [x.data_ptr(), w.data_ptr()],
            [x_new.data_ptr(), w_new.data_ptr()],
        )
        g.replay()

        cuda_ver = (
            float(paddle.version.cuda())
            if paddle.version.cuda() != 'False'
            else 0.0
        )
        if cuda_ver >= 12.4:
            np.testing.assert_allclose(
                y.numpy(), x_new_val + w_new_val, rtol=1e-5
            )
        g.reset()

    def test_replace_input_ptrs_repeated(self):
        """Test calling replace_input_ptrs multiple times in a row.
        Verifies that each replacement overwrites the previous one.
        """
        if not can_use_cuda_graph():
            return

        shape = [4, 4]
        x = self.random_tensor(shape)

        g = CUDAGraph(enable_replace=True)
        g.capture_begin()
        y = x * 3.0
        g.capture_end()

        g.replay()

        x_new1 = self.random_tensor(shape)
        x_new2 = self.random_tensor(shape)
        x_new2_val = x_new2.numpy().copy()

        # First replacement: x -> x_new1
        g.replace_input_ptrs([x.data_ptr()], [x_new1.data_ptr()])
        # Second replacement: x_new1 -> x_new2 (chain replacement)
        g.replace_input_ptrs([x_new1.data_ptr()], [x_new2.data_ptr()])
        g.replay()

        cuda_ver = (
            float(paddle.version.cuda())
            if paddle.version.cuda() != 'False'
            else 0.0
        )
        if cuda_ver >= 12.4:
            np.testing.assert_allclose(y.numpy(), x_new2_val * 3.0, rtol=1e-5)
        g.reset()

    def test_replace_input_ptrs_after_multiple_replays(self):
        """Test replace_input_ptrs interleaved with multiple replays.
        Ensures ReplaceInputPtrs works correctly on non-first-run graphs.
        """
        if not can_use_cuda_graph():
            return

        shape = [4, 4]
        x = self.random_tensor(shape)
        x_val = x.numpy().copy()

        g = CUDAGraph(enable_replace=True)
        g.capture_begin()
        y = x + 5.0
        g.capture_end()

        # Multiple replays before replacing
        for _ in range(3):
            g.replay()
        np.testing.assert_allclose(y.numpy(), x_val + 5.0, rtol=1e-5)

        x_new = self.random_tensor(shape)
        x_new_val = x_new.numpy().copy()
        g.replace_input_ptrs([x.data_ptr()], [x_new.data_ptr()])
        g.replay()

        cuda_ver = (
            float(paddle.version.cuda())
            if paddle.version.cuda() != 'False'
            else 0.0
        )
        if cuda_ver >= 12.4:
            np.testing.assert_allclose(y.numpy(), x_new_val + 5.0, rtol=1e-5)
        g.reset()

    def test_replace_input_ptrs_multiple_kernel_nodes(self):
        if not can_use_cuda_graph():
            return

        shape = [4, 4]
        x = self.random_tensor(shape)
        x_val = x.numpy().copy()

        g = CUDAGraph(enable_replace=True)
        g.capture_begin()
        y = x * 2.0
        z = y + x
        g.capture_end()

        g.replay()
        np.testing.assert_allclose(z.numpy(), x_val * 3.0, rtol=1e-5)

        x_new = self.random_tensor(shape)
        x_new_val = x_new.numpy().copy()
        g.replace_input_ptrs([x.data_ptr()], [x_new.data_ptr()])
        g.replay()

        cuda_ver = (
            float(paddle.version.cuda())
            if paddle.version.cuda() != 'False'
            else 0.0
        )
        if cuda_ver >= 12.4:
            np.testing.assert_allclose(y.numpy(), x_new_val * 2.0, rtol=1e-5)
            np.testing.assert_allclose(z.numpy(), x_new_val * 3.0, rtol=1e-5)
        g.reset()

    def test_replace_input_ptrs_with_pool_id(self):
        if not can_use_cuda_graph():
            return

        shape = [4, 4]
        x = self.random_tensor(shape)
        x_val = x.numpy().copy()

        g = CUDAGraph(pool_id=0, enable_replace=True)
        g.capture_begin()
        y = x - 1.0
        g.capture_end()

        g.replay()
        np.testing.assert_allclose(y.numpy(), x_val - 1.0, rtol=1e-5)

        x_new = self.random_tensor(shape)
        x_new_val = x_new.numpy().copy()
        g.replace_input_ptrs([x.data_ptr()], [x_new.data_ptr()])
        g.replay()

        cuda_ver = (
            float(paddle.version.cuda())
            if paddle.version.cuda() != 'False'
            else 0.0
        )
        if cuda_ver >= 12.4:
            np.testing.assert_allclose(y.numpy(), x_new_val - 1.0, rtol=1e-5)
        g.reset()


@unittest.skipIf(
    not (paddle.is_compiled_with_cuda() or is_custom_device())
    or float(paddle.version.cuda()) < 11.0,
    "only support cuda >= 11.0",
)
class TestPaddleCudaCUDAGraphCompat(unittest.TestCase):
    """Tests for the PyTorch-compat surface of ``paddle.cuda.CUDAGraph``.

    ``paddle.cuda.CUDAGraph`` is the same class as
    ``paddle.device.cuda.graphs.CUDAGraph`` (re-exported via
    ``paddle.cuda.graphs``). These tests cover the PyTorch-compat keyword
    arguments and helper methods (``keep_graph``, ``pool``, ``instantiate``,
    ``enable_debug_mode``/``debug_dump``, ``raw_cuda_graph[_exec]``) plus an
    end-to-end capture/replay round-trip.
    """

    def setUp(self):
        if can_use_cuda_graph():
            paddle.set_flags(
                {
                    'FLAGS_allocator_strategy': 'auto_growth',
                    'FLAGS_sync_nccl_allreduce': False,
                    'FLAGS_cudnn_deterministic': True,
                    'FLAGS_use_stream_safe_cuda_allocator': False,
                }
            )

    def test_paddle_cuda_class_is_consolidated(self):
        # The PyTorch-facing name and the native name must be the same class
        # to avoid duplicated implementations.
        from paddle.device.cuda.graphs import CUDAGraph as DeviceCUDAGraph

        self.assertIs(paddle.cuda.CUDAGraph, DeviceCUDAGraph)
        self.assertIs(paddle.cuda.graphs.CUDAGraph, DeviceCUDAGraph)

    def test_constructor_keep_graph(self):
        g = paddle.cuda.CUDAGraph()
        self.assertFalse(g._keep_graph)
        g2 = paddle.cuda.CUDAGraph(keep_graph=True)
        self.assertTrue(g2._keep_graph)

    def test_constructor_positional_bool_is_keep_graph(self):
        # ``torch.cuda.CUDAGraph(True)`` passes the bool positionally; our
        # signature would otherwise bind it to ``place`` and surface only as
        # an obscure pybind error inside ``capture_begin``. The bool-shim in
        # ``__init__`` re-routes it to ``keep_graph``.
        g_true = paddle.cuda.CUDAGraph(True)
        self.assertTrue(g_true._keep_graph)
        g_false = paddle.cuda.CUDAGraph(False)
        self.assertFalse(g_false._keep_graph)

        # Conflict: ``CUDAGraph(True, keep_graph=True)`` is ambiguous and
        # must raise.
        with self.assertRaises(TypeError):
            paddle.cuda.CUDAGraph(True, keep_graph=True)

    def test_pool_returns_int_idempotent(self):
        g = paddle.cuda.CUDAGraph()
        token = g.pool()
        self.assertIsInstance(token, int)
        self.assertEqual(g.pool(), token)

    def test_graph_pool_handle_returns_int(self):
        token = paddle.cuda.graph_pool_handle()
        self.assertIsInstance(token, int)
        # Each call should yield a fresh token.
        self.assertNotEqual(paddle.cuda.graph_pool_handle(), token)

    def test_capture_begin_pool_propagation(self):
        g = paddle.cuda.CUDAGraph()
        # Stub the C++ binding so we don't enter an actual capture; we only
        # care that the pool argument is propagated into ``_pool_id``.
        from paddle.device.cuda import graphs as graphs_module

        original = graphs_module.CoreCUDAGraph.begin_capture_with_pool_id
        graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = (
            lambda *args, **kwargs: None
        )
        try:
            g.capture_begin(pool=42)
            self.assertEqual(g._pool_id, 42)
        finally:
            graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = original

    def test_capture_error_mode_propagated_and_validated(self):
        # capture_error_mode must be validated against ALL_MODES and the
        # resolved mode index must be forwarded to the C++ binding rather
        # than ignored. Construct with the matching mode so the precedence
        # warning (covered separately) does not fire here.
        from paddle.device.cuda import graphs as graphs_module

        captured = {}
        original = graphs_module.CoreCUDAGraph.begin_capture_with_pool_id

        def fake_begin(place, mode, pool_id, enable_replace):
            captured["mode"] = mode

        graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = fake_begin
        try:
            for name, expected in [
                ("global", 0),
                ("thread_local", 1),
                ("relaxed", 2),
            ]:
                g = paddle.cuda.CUDAGraph(mode=name)
                g.capture_begin(capture_error_mode=name)
                self.assertEqual(captured["mode"], expected)

            g = paddle.cuda.CUDAGraph()
            with self.assertRaises(ValueError):
                g.capture_begin(capture_error_mode="not-a-mode")
        finally:
            graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = original

    def test_capture_error_mode_override_warns(self):
        # An explicit ``capture_error_mode`` overrides the constructor's
        # ``mode`` for this capture; surface a ``UserWarning`` so callers
        # know which one wins. Matching values must not warn.
        import warnings as _w

        from paddle.device.cuda import graphs as graphs_module

        original = graphs_module.CoreCUDAGraph.begin_capture_with_pool_id
        graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = (
            lambda *args, **kwargs: None
        )
        try:
            g = paddle.cuda.CUDAGraph(mode='thread_local')
            with _w.catch_warnings(record=True) as ws:
                _w.simplefilter('always')
                g.capture_begin(capture_error_mode='global')
            msgs = [str(w.message) for w in ws]
            self.assertTrue(
                any('takes precedence' in m for m in msgs),
                f'expected precedence warning, got: {msgs}',
            )

            g = paddle.cuda.CUDAGraph(mode='thread_local')
            with _w.catch_warnings(record=True) as ws:
                _w.simplefilter('always')
                g.capture_begin(capture_error_mode='thread_local')
            msgs = [str(w.message) for w in ws]
            self.assertFalse(
                any('takes precedence' in m for m in msgs),
                f'unexpected precedence warning: {msgs}',
            )
        finally:
            graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = original

    def test_capture_begin_uses_constructor_mode_when_unset(self):
        # When ``capture_error_mode`` is omitted, ``capture_begin`` must
        # fall back to the constructor's ``mode`` rather than silently
        # forcing 'global'. Existing Paddle callers like
        # ``CUDAGraph(mode='relaxed').capture_begin()`` rely on this.
        from paddle.device.cuda import graphs as graphs_module

        captured = {}
        original = graphs_module.CoreCUDAGraph.begin_capture_with_pool_id

        def fake_begin(place, mode, pool_id, enable_replace):
            captured["mode"] = mode

        graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = fake_begin
        try:
            for name, expected in [
                ("global", 0),
                ("thread_local", 1),
                ("relaxed", 2),
            ]:
                g = paddle.cuda.CUDAGraph(mode=name)
                g.capture_begin()
                self.assertEqual(captured["mode"], expected)
        finally:
            graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = original

    def test_capture_begin_materializes_pool_id(self):
        # ``g.pool()`` must return the same id that ``capture_begin`` passed
        # to the C++ side, so that ``g2.capture_begin(pool=g.pool())`` shares
        # the same memory pool. Before materialization, capture_begin would
        # forward ``None`` and pool() would later mint a *different* id.
        from paddle.device.cuda import graphs as graphs_module

        captured = {}
        original = graphs_module.CoreCUDAGraph.begin_capture_with_pool_id

        def fake_begin(place, mode, pool_id, enable_replace):
            captured["pool_id"] = pool_id

        graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = fake_begin
        try:
            g = paddle.cuda.CUDAGraph()
            g.capture_begin()
            self.assertIsNotNone(captured["pool_id"])
            self.assertEqual(g.pool(), captured["pool_id"])
        finally:
            graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = original

    def test_instantiate_requires_capture(self):
        # ``instantiate`` returns the core graph built by ``capture_end``; on a
        # freshly constructed graph there is nothing captured yet, so it raises.
        g = paddle.cuda.CUDAGraph()
        with self.assertRaises(RuntimeError):
            g.instantiate()

    def test_debug_dump_requires_enable_debug_mode(self):
        g = paddle.cuda.CUDAGraph()
        with self.assertRaises(RuntimeError):
            g.debug_dump("/tmp/no_such_path")
        g.enable_debug_mode()
        self.assertTrue(g._debug_mode)

    def test_raw_handles_not_implemented(self):
        g = paddle.cuda.CUDAGraph()
        with self.assertRaises(NotImplementedError):
            g.raw_cuda_graph()
        with self.assertRaises(NotImplementedError):
            g.raw_cuda_graph_exec()

    def test_capture_replay_round_trip(self):
        if not can_use_cuda_graph():
            return

        shape = [2, 3]
        x = paddle.to_tensor(
            np.random.randint(0, 10, size=shape).astype("float32")
        )
        z = paddle.to_tensor(
            np.random.randint(0, 10, size=shape).astype("float32")
        )

        g = paddle.cuda.CUDAGraph()
        g.capture_begin()
        y = x + 10
        z.add_(x)
        g.capture_end()
        # ``instantiate`` returns the core CUDA graph held after capture.
        self.assertIs(g.instantiate(), g._graph)

        z_before = z.numpy().copy()
        x_new = paddle.to_tensor(
            np.random.randint(0, 10, size=shape).astype("float32")
        )
        x.copy_(x_new, False)
        g.replay()
        self.assertTrue((y.numpy() - x.numpy() == 10).all())
        self.assertTrue((z.numpy() - z_before == x.numpy()).all())
        g.reset()


@unittest.skipIf(
    not (paddle.is_compiled_with_cuda() or is_custom_device())
    or float(paddle.version.cuda()) < 11.0,
    "only support cuda >= 11.0",
)
class TestPaddleCudaGraphContextManager(unittest.TestCase):
    """Tests for ``paddle.cuda.graph``, the PyTorch-compat context manager."""

    def setUp(self):
        if can_use_cuda_graph():
            paddle.set_flags(
                {
                    'FLAGS_allocator_strategy': 'auto_growth',
                    'FLAGS_sync_nccl_allreduce': False,
                    'FLAGS_cudnn_deterministic': True,
                    'FLAGS_use_stream_safe_cuda_allocator': False,
                }
            )

    def test_graph_context_manager_propagates_args(self):
        # Verify ``with paddle.cuda.graph(g, pool=...)`` invokes
        # ``capture_begin``/``capture_end`` and threads ``pool`` through.
        g = paddle.cuda.CUDAGraph()
        from paddle.device.cuda import graphs as graphs_module

        called = {"begin": 0, "end": 0}
        original_begin = graphs_module.CoreCUDAGraph.begin_capture_with_pool_id
        original_end = graphs_module.CoreCUDAGraph.end_capture
        graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = (
            lambda *args, **kwargs: called.__setitem__(
                "begin", called["begin"] + 1
            )
        )
        graphs_module.CoreCUDAGraph.end_capture = lambda: called.__setitem__(
            "end", called["end"] + 1
        )
        try:
            with paddle.cuda.graph(g, pool=7):
                pass
            self.assertEqual(called["begin"], 1)
            self.assertEqual(called["end"], 1)
            self.assertEqual(g._pool_id, 7)
        finally:
            graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = (
                original_begin
            )
            graphs_module.CoreCUDAGraph.end_capture = original_end

    def test_graph_context_manager_capture_replay(self):
        if not can_use_cuda_graph():
            return

        shape = [2, 3]
        x = paddle.to_tensor(
            np.random.randint(0, 10, size=shape).astype("float32")
        )

        g = paddle.cuda.CUDAGraph()
        with paddle.cuda.graph(g):
            y = x + 1
        g.replay()
        np.testing.assert_allclose(y.numpy(), x.numpy() + 1, rtol=1e-5)
        g.reset()

    def test_graph_context_manager_switches_stream(self):
        # When ``stream`` is provided, the body must execute on that stream
        # and the previous stream must be restored on exit.
        if not can_use_cuda_graph():
            return

        side_stream = paddle.device.Stream()
        prev_stream = paddle.device.current_stream()
        observed = {}

        g = paddle.cuda.CUDAGraph()
        from paddle.device.cuda import graphs as graphs_module

        original_begin = graphs_module.CoreCUDAGraph.begin_capture_with_pool_id
        original_end = graphs_module.CoreCUDAGraph.end_capture
        graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = (
            lambda *args, **kwargs: observed.__setitem__(
                "inside", paddle.device.current_stream()
            )
        )
        graphs_module.CoreCUDAGraph.end_capture = lambda: None
        try:
            with paddle.cuda.graph(g, stream=side_stream):
                pass
        finally:
            graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = (
                original_begin
            )
            graphs_module.CoreCUDAGraph.end_capture = original_end

        # ``Stream`` wrappers are fresh Python objects each call; compare by
        # repr which embeds the underlying CUDA stream pointer.
        self.assertEqual(repr(observed["inside"]), repr(side_stream))
        self.assertEqual(
            repr(paddle.device.current_stream()), repr(prev_stream)
        )

    def test_graph_context_manager_default_does_not_warn(self):
        # ``with paddle.cuda.graph(CUDAGraph()):`` is the most common
        # default usage; it must not surface the
        # ``capture_error_mode-takes-precedence`` warning, since the user
        # didn't pick ``capture_error_mode`` here.
        import warnings as _w

        from paddle.device.cuda import graphs as graphs_module

        original_begin = graphs_module.CoreCUDAGraph.begin_capture_with_pool_id
        original_end = graphs_module.CoreCUDAGraph.end_capture
        graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = (
            lambda *args, **kwargs: None
        )
        graphs_module.CoreCUDAGraph.end_capture = lambda: None
        try:
            g = paddle.cuda.CUDAGraph()
            with _w.catch_warnings(record=True) as ws:
                _w.simplefilter('always')
                with paddle.cuda.graph(g):
                    pass
            msgs = [str(w.message) for w in ws]
            self.assertFalse(
                any('takes precedence' in m for m in msgs),
                f'unexpected precedence warning: {msgs}',
            )
        finally:
            graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = (
                original_begin
            )
            graphs_module.CoreCUDAGraph.end_capture = original_end

    def test_graph_context_manager_stream_none_is_noop(self):
        # When ``stream`` is None the current stream must be unchanged.
        if not can_use_cuda_graph():
            return

        prev_stream = paddle.device.current_stream()
        g = paddle.cuda.CUDAGraph()
        from paddle.device.cuda import graphs as graphs_module

        original_begin = graphs_module.CoreCUDAGraph.begin_capture_with_pool_id
        original_end = graphs_module.CoreCUDAGraph.end_capture
        graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = (
            lambda *args, **kwargs: None
        )
        graphs_module.CoreCUDAGraph.end_capture = lambda: None
        try:
            with paddle.cuda.graph(g):
                self.assertEqual(
                    repr(paddle.device.current_stream()), repr(prev_stream)
                )
        finally:
            graphs_module.CoreCUDAGraph.begin_capture_with_pool_id = (
                original_begin
            )
            graphs_module.CoreCUDAGraph.end_capture = original_end

        self.assertEqual(
            repr(paddle.device.current_stream()), repr(prev_stream)
        )


if __name__ == "__main__":
    unittest.main()
