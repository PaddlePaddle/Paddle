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
"""
Tests for paddle.utils._print_tensor_in_gpu().

Coverage targets:
  - All 10 supported dtypes (float32, float64, float16, bfloat16,
    int32, int64, int16, int8, uint8, bool)
  - Scalar (0-D), 1-D, 2-D, 3-D tensors
  - Empty tensor (numel == 0)
  - Stream ordering (print after compute op)
  - CUDA Graph capture / replay
  - CUDA Graph with data update + replay
  - CUDA Graph with mixed ops (add + print)
  - CUDA Graph with multiple dtypes in one graph
  - Error: CPU tensor -> InvalidArgument
  - Error: non-DenseTensor (SparseCooTensor) -> InvalidArgument
  - Python API reachability via paddle.utils._print_tensor_in_gpu

All CUDA device-side printf output is captured to a string via fd-level
stdout redirection, so the tests do not pollute CI logs.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

import paddle
from paddle.device.cuda.graphs import CUDAGraph


def can_use_cuda():
    return paddle.is_compiled_with_cuda()


def can_use_cuda_graph():
    return (
        paddle.is_compiled_with_cuda()
        and not paddle.is_compiled_with_rocm()
        and float(paddle.version.cuda()) >= 11.0
    )


def set_cuda_graph_flags():
    paddle.set_flags(
        {
            'FLAGS_allocator_strategy': 'auto_growth',
            'FLAGS_sync_nccl_allreduce': False,
            'FLAGS_cudnn_deterministic': True,
            'FLAGS_use_stream_safe_cuda_allocator': False,
        }
    )


def capture_print_tensor(tensor):
    """Call _print_tensor_in_gpu and capture its C-level stdout output.

    CUDA device-side printf writes to fd 1, so we redirect fd 1 to a
    temporary file, call the API + synchronize, then read the file.
    Returns the captured output as a string.
    """
    with tempfile.NamedTemporaryFile(
        mode='w+', suffix='.txt', delete=False
    ) as tmp:
        tmp_path = tmp.name

    sys.stdout.flush()
    old_fd = os.dup(1)
    try:
        redir = os.open(tmp_path, os.O_WRONLY | os.O_TRUNC)
        os.dup2(redir, 1)
        os.close(redir)

        paddle.utils._print_tensor_in_gpu(tensor)
        paddle.device.synchronize()

        # Flush libc stdout buffer so everything lands in the file.
        sys.stdout.flush()
    finally:
        os.dup2(old_fd, 1)
        os.close(old_fd)

    with open(tmp_path, 'r') as f:
        content = f.read()
    os.remove(tmp_path)
    return content


@unittest.skipIf(not can_use_cuda(), "Requires CUDA")
@unittest.skipIf(os.name == 'nt', "Not supported on Windows")
class TestPrintTensorInGpuBasic(unittest.TestCase):
    """Basic functionality tests (no CUDA Graph)."""

    def _call(self, tensor):
        return capture_print_tensor(tensor)

    # ── dtype coverage ──────────────────────────────────────────────────

    def test_float32_2d(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32'
        )
        out = self._call(x)
        self.assertIn('FLOAT32', out)
        self.assertIn('[2, 3]', out)

    def test_float64_2d(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float64')
        out = self._call(x)
        self.assertIn('FLOAT64', out)

    def test_float16_2d(self):
        x = paddle.to_tensor([[0.5, -0.5], [1.5, -1.5]], dtype='float16')
        out = self._call(x)
        self.assertIn('FLOAT16', out)

    def test_bfloat16_2d(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='bfloat16'
        )
        out = self._call(x)
        self.assertIn('BFLOAT16', out)

    def test_int32_1d(self):
        x = paddle.to_tensor([1, 2, 3, 4, 5], dtype='int32')
        out = self._call(x)
        self.assertIn('INT32', out)

    def test_int64_1d(self):
        x = paddle.arange(10, 15, dtype='int64')
        out = self._call(x)
        self.assertIn('INT64', out)

    def test_int16_1d(self):
        x = paddle.to_tensor([1, 2, 3], dtype='int16')
        out = self._call(x)
        self.assertIn('INT16', out)

    def test_int8_1d(self):
        x = paddle.to_tensor([1, -1, 127], dtype='int8')
        out = self._call(x)
        self.assertIn('INT8', out)

    def test_uint8_1d(self):
        x = paddle.to_tensor([0, 128, 255], dtype='uint8')
        out = self._call(x)
        self.assertIn('UINT8', out)

    def test_bool_1d(self):
        x = paddle.to_tensor([True, False, True, False])
        out = self._call(x)
        self.assertIn('BOOL', out)

    # ── shape coverage ──────────────────────────────────────────────────

    def test_scalar_0d(self):
        x = paddle.to_tensor(3.14, dtype='float64')
        out = self._call(x)
        self.assertIn('FLOAT64', out)

    def test_1d(self):
        x = paddle.arange(5, dtype='float32')
        out = self._call(x)
        self.assertIn('[5]', out)

    def test_3d(self):
        x = paddle.arange(24, dtype='int32').reshape([2, 3, 4])
        out = self._call(x)
        self.assertIn('[2, 3, 4]', out)

    def test_empty_tensor(self):
        x = paddle.empty([0], dtype='float32').cuda()
        out = self._call(x)
        self.assertIn('[]', out)

    # ── stream ordering ─────────────────────────────────────────────────

    def test_stream_ordering_after_op(self):
        """Values should reflect the preceding add op."""
        a = paddle.ones([3], dtype='float32')
        b = paddle.ones([3], dtype='float32') * 2.0
        c = a + b  # expect [3, 3, 3]
        out = self._call(c)
        self.assertIn('[TensorDebug]', out)
        np.testing.assert_allclose(c.numpy(), [3.0, 3.0, 3.0])


@unittest.skipIf(not can_use_cuda(), "Requires CUDA")
@unittest.skipIf(os.name == 'nt', "Not supported on Windows")
class TestPrintTensorInGpuErrors(unittest.TestCase):
    """Error path tests."""

    def test_cpu_tensor_raises(self):
        cpu_t = paddle.to_tensor([1.0, 2.0], place=paddle.CPUPlace())
        with self.assertRaises(ValueError):
            paddle.utils._print_tensor_in_gpu(cpu_t)

    def test_sparse_tensor_raises(self):
        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        sparse_t = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
        with self.assertRaises(ValueError):
            paddle.utils._print_tensor_in_gpu(sparse_t)


@unittest.skipIf(
    not can_use_cuda_graph(),
    "CUDA Graph requires CUDA >= 11.0 and non-ROCm build",
)
@unittest.skipIf(os.name == 'nt', "Not supported on Windows")
class TestPrintTensorInGpuCUDAGraph(unittest.TestCase):
    """CUDA Graph integration tests."""

    def setUp(self):
        set_cuda_graph_flags()

    def _capture_replay(self, sync_and_read=True):
        """Helper context: redirect stdout during replay + sync."""

        class _Ctx:
            def __init__(self):
                self.output = ''
                self._tmp_path = None
                self._old_fd = None

            def __enter__(self):
                self._tmp = tempfile.NamedTemporaryFile(
                    mode='w+', suffix='.txt', delete=False
                )
                self._tmp_path = self._tmp.name
                self._tmp.close()
                sys.stdout.flush()
                self._old_fd = os.dup(1)
                redir = os.open(self._tmp_path, os.O_WRONLY | os.O_TRUNC)
                os.dup2(redir, 1)
                os.close(redir)
                return self

            def __exit__(self, *exc):
                sys.stdout.flush()
                os.dup2(self._old_fd, 1)
                os.close(self._old_fd)
                with open(self._tmp_path, 'r') as f:
                    self.output = f.read()
                os.remove(self._tmp_path)

        return _Ctx()

    def test_alone_in_graph(self):
        """Capture and replay a graph that only contains _print_tensor_in_gpu."""
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32'
        )

        g = CUDAGraph()
        g.capture_begin()
        paddle.utils._print_tensor_in_gpu(x)
        g.capture_end()

        with self._capture_replay() as ctx:
            g.replay()
            paddle.device.synchronize()

        self.assertIn('FLOAT32', ctx.output)

        # Replay again to verify repeatability.
        with self._capture_replay() as ctx2:
            g.replay()
            paddle.device.synchronize()

        self.assertIn('FLOAT32', ctx2.output)
        g.reset()

    def test_with_ops(self):
        """Graph: y = x + 1 -> _print_tensor_in_gpu(y)."""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
        y = paddle.zeros_like(x)

        g = CUDAGraph()
        g.capture_begin()
        y = x + 1.0
        paddle.utils._print_tensor_in_gpu(y)
        g.capture_end()

        with self._capture_replay() as ctx:
            g.replay()
            paddle.device.synchronize()

        self.assertIn('FLOAT32', ctx.output)
        np.testing.assert_allclose(
            y.numpy(),
            np.array([[2.0, 3.0], [4.0, 5.0]], dtype='float32'),
        )

        g.reset()

    def test_after_data_update(self):
        """
        Graph captures _print_tensor_in_gpu on tensor x.
        After overwriting x's data and replaying, the new values are printed
        because the kernel re-runs on the same device buffer.
        """
        x = paddle.zeros([2, 3], dtype='float32')

        g = CUDAGraph()
        g.capture_begin()
        paddle.utils._print_tensor_in_gpu(x)
        g.capture_end()

        # First replay: zeros.
        with self._capture_replay() as ctx1:
            g.replay()
            paddle.device.synchronize()

        self.assertIn('FLOAT32', ctx1.output)

        # Update data in-place.
        new_data = paddle.to_tensor(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype='float32'
        )
        x.copy_(new_data, False)

        # Second replay: updated values.
        with self._capture_replay() as ctx2:
            g.replay()
            paddle.device.synchronize()

        self.assertIn('FLOAT32', ctx2.output)
        g.reset()

    def test_multiple_dtypes(self):
        """Capture _print_tensor_in_gpu for float16, int32, bool in one graph."""
        f16 = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float16')
        i32 = paddle.to_tensor([[5, 6], [7, 8]], dtype='int32')
        bl = paddle.to_tensor([True, False, True], dtype='bool')

        g = CUDAGraph()
        g.capture_begin()
        paddle.utils._print_tensor_in_gpu(f16)
        paddle.utils._print_tensor_in_gpu(i32)
        paddle.utils._print_tensor_in_gpu(bl)
        g.capture_end()

        with self._capture_replay() as ctx:
            g.replay()
            paddle.device.synchronize()

        self.assertIn('FLOAT16', ctx.output)
        self.assertIn('INT32', ctx.output)
        self.assertIn('BOOL', ctx.output)
        g.reset()

    def test_bfloat16_in_graph(self):
        """Verify bfloat16 inside CUDA Graph."""
        bf = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='bfloat16'
        )

        g = CUDAGraph()
        g.capture_begin()
        paddle.utils._print_tensor_in_gpu(bf)
        g.capture_end()

        with self._capture_replay() as ctx:
            g.replay()
            paddle.device.synchronize()

        self.assertIn('BFLOAT16', ctx.output)
        g.reset()

    def test_3d_shape_in_graph(self):
        """3-D tensor nested bracket output inside a graph."""
        t = paddle.arange(24, dtype='int32').reshape([2, 3, 4])

        g = CUDAGraph()
        g.capture_begin()
        paddle.utils._print_tensor_in_gpu(t)
        g.capture_end()

        with self._capture_replay() as ctx:
            g.replay()
            paddle.device.synchronize()

        self.assertIn('[2, 3, 4]', ctx.output)
        g.reset()

    def test_scalar_in_graph(self):
        """0-D scalar tensor inside a graph."""
        s = paddle.to_tensor(42.0, dtype='float32')

        g = CUDAGraph()
        g.capture_begin()
        paddle.utils._print_tensor_in_gpu(s)
        g.capture_end()

        with self._capture_replay() as ctx:
            g.replay()
            paddle.device.synchronize()

        self.assertIn('FLOAT32', ctx.output)
        g.reset()

    def test_empty_tensor_in_graph(self):
        """Empty tensor (numel=0) inside a graph."""
        e = paddle.empty([0], dtype='float32').cuda()

        g = CUDAGraph()
        g.capture_begin()
        paddle.utils._print_tensor_in_gpu(e)
        g.capture_end()

        with self._capture_replay() as ctx:
            g.replay()
            paddle.device.synchronize()

        self.assertIn('[]', ctx.output)
        g.reset()


if __name__ == '__main__':
    unittest.main(verbosity=2)
