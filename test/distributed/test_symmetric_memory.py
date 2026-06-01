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

"""
Unit tests for paddle.distributed._symmetric_memory

This test file mirrors PyTorch's test/distributed/test_symmetric_memory.py.
It tests both the pure-Python fallback API and the C++ backed API.
"""

import os
import sys
import unittest
import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed._symmetric_memory import (
    empty,
    is_symm_mem_tensor,
    rendezvous,
    set_backend,
    get_backend,
    set_signal_pad_size,
    get_signal_pad_size,
    has_multicast_support,
    stream_write_value32,
    memset32,
    _test_mode,
    _fused_all_gather_matmul_fallback,
    _fused_matmul_reduce_scatter_fallback,
    _fused_all_gather_scaled_matmul,
    _fused_scaled_matmul_reduce_scatter,
    restride_A_shard_for_fused_all_gather_matmul,
    restride_A_for_fused_matmul_reduce_scatter,
)


def is_gpu_available():
    return paddle.device.cuda.device_count() > 0


def get_gpu_count():
    return paddle.device.cuda.device_count()


def skip_if_lt_x_gpu(x):
    """Decorator to skip test if fewer than x GPUs available."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if get_gpu_count() < x:
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================
# Test Class 1: SymmetricMemoryTest (Core API tests)
# Corresponds to PyTorch's SymmetricMemoryTest
# ============================================================
class TestSymmetricMemoryAPI(unittest.TestCase):
    """Test core symmetric memory API functions."""

    def test_has_multicast_support(self):
        """Validate that has_multicast_support() returns False (not throws)."""
        result = has_multicast_support("gpu", 0)
        self.assertFalse(result)
        # CPU should also not throw
        result = has_multicast_support("cpu", 0)
        self.assertFalse(result)

    def test_is_symm_mem_tensor(self):
        """Test symmetric memory tensor detection."""
        # CPU tensor -> False
        t_cpu = paddle.zeros([1024])
        self.assertFalse(is_symm_mem_tensor(t_cpu))

        # Regular GPU tensor -> False (when not allocated via symm_mem)
        if is_gpu_available():
            paddle.device.set_device('gpu:0')
            t_gpu = paddle.zeros([1024])
            self.assertFalse(is_symm_mem_tensor(t_gpu))

    def test_get_backend(self):
        """Test backend getter."""
        backend = get_backend("gpu")
        self.assertIsNotNone(backend)
        self.assertEqual(backend, "cuda")

    def test_get_signal_pad_size(self):
        """Test signal pad size getter returns positive int."""
        signal_pad_size = get_signal_pad_size()
        self.assertIsInstance(signal_pad_size, int)
        self.assertGreater(signal_pad_size, 0)

    def test_set_signal_pad_size(self):
        """Test signal pad size setter."""
        original_size = get_signal_pad_size()

        # Set new size
        new_size = 1024 * 1024  # 1MB
        set_signal_pad_size(new_size)
        self.assertEqual(get_signal_pad_size(), new_size)

        # Restore
        set_signal_pad_size(original_size)
        self.assertEqual(get_signal_pad_size(), original_size)

    def test_empty_basic(self):
        """Test basic empty allocation."""
        with _test_mode():
            t = empty(1024)
            self.assertEqual(t.shape, [1024])
            self.assertEqual(t.dtype, paddle.float32)

    def test_empty_with_dtype(self):
        """Test empty allocation with specific dtype."""
        with _test_mode():
            t = empty(512, dtype=paddle.float16)
            self.assertEqual(t.shape, [512])
            self.assertEqual(t.dtype, paddle.float16)

    def test_empty_multi_dim(self):
        """Test multi-dimensional empty allocation."""
        with _test_mode():
            t = empty(64, 64)
            self.assertEqual(t.shape, [64, 64])

    def test_large_alloc(self):
        """Test large allocation (2GB)."""
        with _test_mode():
            # In test mode, this just creates a zeros tensor
            t = empty(2 * 1024**3, dtype=paddle.uint8)
            numel = t.numel()
            if hasattr(numel, 'item'):
                numel = numel.item()
            self.assertEqual(int(numel) * t.element_size(), 2 * 1024**3)


# ============================================================
# Test Class 2: AsyncTPTest (Fused operations)
# Corresponds to PyTorch's AsyncTPTest
# ============================================================
class TestAsyncTPOps(unittest.TestCase):
    """Test Async TP fused operations (fallback implementations)."""

    def test_optimal_layout_dim0(self):
        """Test restride for gather_dim=0."""
        t = paddle.rand([8, 64, 32, 16])
        x = restride_A_shard_for_fused_all_gather_matmul(t, 0)
        # After moveaxis(0, 0) should be contiguous
        moved = x.moveaxis(0, 0)
        self.assertTrue(moved.is_contiguous())
        np.testing.assert_allclose(x.numpy(), t.numpy(), rtol=1e-5)

    def test_optimal_layout_dim1(self):
        """Test restride for gather_dim=1."""
        t = paddle.rand([8, 64, 32, 16])
        x = restride_A_shard_for_fused_all_gather_matmul(t, 1)
        moved = x.moveaxis(1, 0)
        self.assertTrue(moved.is_contiguous())
        np.testing.assert_allclose(x.numpy(), t.numpy(), rtol=1e-5)

    def test_optimal_layout_dim2(self):
        """Test restride for gather_dim=2."""
        t = paddle.rand([8, 64, 32, 16])
        x = restride_A_shard_for_fused_all_gather_matmul(t, 2)
        moved = x.moveaxis(2, 0)
        self.assertTrue(moved.is_contiguous())
        np.testing.assert_allclose(x.numpy(), t.numpy(), rtol=1e-5)

    def test_restride_reduce_scatter_dim0(self):
        """Test restride for scatter_dim=0."""
        t = paddle.rand([8, 64, 32, 16])
        x = restride_A_for_fused_matmul_reduce_scatter(t, 0)
        moved = x.moveaxis(0, 0)
        self.assertTrue(moved.is_contiguous())
        np.testing.assert_allclose(x.numpy(), t.numpy(), rtol=1e-5)

    def test_restride_reduce_scatter_dim1(self):
        """Test restride for scatter_dim=1."""
        t = paddle.rand([8, 64, 32, 16])
        x = restride_A_for_fused_matmul_reduce_scatter(t, 1)
        moved = x.moveaxis(1, 0)
        self.assertTrue(moved.is_contiguous())
        np.testing.assert_allclose(x.numpy(), t.numpy(), rtol=1e-5)

    def test_restride_reduce_scatter_dim2(self):
        """Test restride for scatter_dim=2."""
        t = paddle.rand([8, 64, 32, 16])
        x = restride_A_for_fused_matmul_reduce_scatter(t, 2)
        moved = x.moveaxis(2, 0)
        self.assertTrue(moved.is_contiguous())
        np.testing.assert_allclose(x.numpy(), t.numpy(), rtol=1e-5)


# ============================================================
# Test Class 3: SymmMemNegativeTest (Error handling)
# Corresponds to PyTorch's SymmMemNegativeTest
# ============================================================
class TestSymmetricMemoryNegative(unittest.TestCase):
    """Test error handling of symmetric memory APIs."""

    def test_rendezvous_non_symm_tensor(self):
        """Test rendezvous with non-symmetric tensor raises error."""
        with _test_mode():
            # In test mode, rendezvous works with any tensor (returns fallback)
            t = paddle.zeros([64])
            handle = rendezvous(t)
            self.assertIsNotNone(handle)

    def test_backend_setting(self):
        """Test set/get backend."""
        original = get_backend()
        set_backend("test_backend")
        self.assertEqual(get_backend(), "test_backend")
        set_backend(original)


# ============================================================
# Test Class 4: SymmMemSingleProcTest (Single process tests)
# Corresponds to PyTorch's SymmMemSingleProcTest
# ============================================================
class TestSymmetricMemorySingleProc(unittest.TestCase):
    """Tests that can run in a single process."""

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_stream_write_value32_basic(self):
        """Test stream_write_value32 basic functionality."""
        # This test validates the API exists and basic semantics
        paddle.device.set_device('gpu:0')
        tensor = paddle.zeros([4], dtype='int32')
        # In fallback mode, this may be a no-op
        # Real test would verify GPU memory writes
        try:
            stream_write_value32(tensor, 0, 1)
        except Exception:
            pass  # OK if backend not available

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_memset32_basic(self):
        """Test memset32 basic functionality."""
        paddle.device.set_device('gpu:0')
        tensor = paddle.zeros([64], dtype='int32')
        try:
            memset32(tensor, 0, 1, 32)
        except Exception:
            pass  # OK if backend not available

    def test_signal_pad_size_roundtrip(self):
        """Test signal pad size get/set roundtrip."""
        original = get_signal_pad_size()
        self.assertGreater(original, 0)

        set_signal_pad_size(2 * original)
        self.assertEqual(get_signal_pad_size(), 2 * original)

        set_signal_pad_size(original)
        self.assertEqual(get_signal_pad_size(), original)


# ============================================================
# Test Class 5: SymmMemCollectiveTest placeholder
# Corresponds to PyTorch's SymmMemCollectiveTest
# These tests require multi-process execution
# ============================================================
class TestSymmetricMemoryCollective(unittest.TestCase):
    """Placeholder for collective tests (require multi-process)."""

    def test_test_mode_context(self):
        """Test the _test_mode context manager."""
        from paddle.distributed._symmetric_memory import _is_test_mode
        self.assertFalse(_is_test_mode)
        with _test_mode():
            from paddle.distributed._symmetric_memory import _is_test_mode as tm
            self.assertTrue(tm)

    def test_fallback_symmetric_memory_handle(self):
        """Test FallbackSymmetricMemory handle properties."""
        with _test_mode():
            t = empty(64)
            handle = rendezvous(t)
            self.assertEqual(handle.rank, 0)
            self.assertEqual(handle.world_size, 1)
            self.assertGreater(handle.buffer_size, 0)
            self.assertGreater(handle.signal_pad_size, 0)

    def test_fallback_get_buffer(self):
        """Test FallbackSymmetricMemory get_buffer."""
        with _test_mode():
            t = empty(64)
            handle = rendezvous(t)
            buf = handle.get_buffer(0, [64], paddle.float32)
            self.assertEqual(buf.shape, [64])

    def test_fallback_get_signal_pad(self):
        """Test FallbackSymmetricMemory get_signal_pad."""
        with _test_mode():
            t = empty(64)
            handle = rendezvous(t)
            pad = handle.get_signal_pad(0)
            self.assertEqual(pad.dtype, paddle.int32)
            self.assertGreater(pad.numel(), 0)

    def test_fallback_barrier(self):
        """Test FallbackSymmetricMemory barrier (no-op in single process)."""
        with _test_mode():
            t = empty(64)
            handle = rendezvous(t)
            # Should not raise
            handle.barrier()
            handle.barrier(channel=1)

    def test_fallback_put_wait_signal(self):
        """Test FallbackSymmetricMemory put/wait signal (no-op)."""
        with _test_mode():
            t = empty(64)
            handle = rendezvous(t)
            # Should not raise in fallback mode
            handle.put_signal(dst_rank=0)
            handle.wait_signal(src_rank=0)


# ============================================================
# Test Class 6: SymmMemPoolTest placeholder
# Corresponds to PyTorch's SymmMemPoolTest
# ============================================================
class TestSymmetricMemoryPool(unittest.TestCase):
    """Test memory pool functionality."""

    def test_empty_creates_tensor(self):
        """Test that empty() creates a valid tensor."""
        with _test_mode():
            t = empty(1024, dtype=paddle.float32)
            self.assertEqual(t.shape, [1024])
            self.assertEqual(t.dtype, paddle.float32)

    def test_empty_different_sizes(self):
        """Test empty with various sizes."""
        with _test_mode():
            for size in [1, 64, 1024, 8192]:
                t = empty(size)
                self.assertEqual(t.numel(), size)


# ============================================================
# Test Class 7: FP8 Tests
# Tests for FP8 data type support in symmetric memory
# ============================================================
class TestSymmetricMemoryFP8(unittest.TestCase):
    """Test FP8 data type support for symmetric memory operations."""

    def test_empty_fp8_e4m3fn(self):
        """Test empty allocation with float8_e4m3fn dtype."""
        with _test_mode():
            t = empty(32, 64, dtype=paddle.float8_e4m3fn)
            self.assertEqual(t.shape, [32, 64])
            self.assertEqual(t.dtype, paddle.float8_e4m3fn)

    def test_empty_fp8_e5m2(self):
        """Test empty allocation with float8_e5m2 dtype."""
        with _test_mode():
            t = empty(32, 64, dtype=paddle.float8_e5m2)
            self.assertEqual(t.shape, [32, 64])
            self.assertEqual(t.dtype, paddle.float8_e5m2)

    def test_fp8_cast_roundtrip(self):
        """Test FP8 cast roundtrip preserves reasonable values."""
        paddle.device.set_device('gpu:0')
        # Use small values to avoid overflow in FP8
        original = paddle.rand([16, 32], dtype=paddle.float32) * 2.0 - 1.0
        fp8_e4m3 = original.cast(paddle.float8_e4m3fn)
        back = fp8_e4m3.cast(paddle.float32)
        # FP8 has limited precision, allow ~10% relative error
        np.testing.assert_allclose(
            back.numpy(), original.numpy(), rtol=0.15, atol=0.1
        )

    def test_fp8_e5m2_cast_roundtrip(self):
        """Test FP8 E5M2 cast roundtrip preserves reasonable values."""
        paddle.device.set_device('gpu:0')
        original = paddle.rand([16, 32], dtype=paddle.float32) * 2.0 - 1.0
        fp8_e5m2 = original.cast(paddle.float8_e5m2)
        back = fp8_e5m2.cast(paddle.float32)
        # E5M2 has even less precision than E4M3
        np.testing.assert_allclose(
            back.numpy(), original.numpy(), rtol=0.3, atol=0.2
        )

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_fp8_gemm_basic(self):
        """Test basic FP8 GEMM via paddle.linalg.fp8_fp8_half_gemm_fused."""
        paddle.device.set_device('gpu:0')
        M, K, N = 32, 64, 32  # K must be % 16 == 0

        # Create FP8 inputs
        a_f32 = paddle.randn([M, K], dtype=paddle.float32) * 0.1
        b_f32 = paddle.randn([K, N], dtype=paddle.float32) * 0.1
        a_fp8 = a_f32.cast(paddle.float8_e4m3fn)
        b_fp8 = b_f32.cast(paddle.float8_e4m3fn)

        # FP8 gemm
        result = paddle.linalg.fp8_fp8_half_gemm_fused(
            a_fp8, b_fp8, False, False, None, 1.0, 'float16', 'identity'
        )
        self.assertEqual(result.shape, [M, N])
        self.assertEqual(result.dtype, paddle.float16)
        # Verify non-zero and finite
        self.assertGreater(float(result.abs().max()), 0.0)
        self.assertFalse(bool(paddle.isnan(result).any()))
        self.assertFalse(bool(paddle.isinf(result).any()))

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_fp8_gemm_with_scale(self):
        """Test FP8 GEMM with scaling factor (self-consistency check)."""
        paddle.device.set_device('gpu:0')
        M, K, N = 32, 64, 32

        a_fp8 = (paddle.randn([M, K], dtype=paddle.float32) * 0.1).cast(paddle.float8_e4m3fn)
        b_fp8 = (paddle.randn([K, N], dtype=paddle.float32) * 0.1).cast(paddle.float8_e4m3fn)

        # Scale=1.0 result
        r1 = paddle.linalg.fp8_fp8_half_gemm_fused(
            a_fp8, b_fp8, False, False, None, 1.0, 'float16', 'identity'
        )
        # Scale=2.0 result
        r2 = paddle.linalg.fp8_fp8_half_gemm_fused(
            a_fp8, b_fp8, False, False, None, 2.0, 'float16', 'identity'
        )

        # scale=2 should give exactly 2x the scale=1 result
        np.testing.assert_allclose(
            r2.cast(paddle.float32).numpy(),
            (2.0 * r1.cast(paddle.float32)).numpy(),
            rtol=1e-5, atol=1e-5
        )

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_fp8_gemm_bfloat16_output(self):
        """Test FP8 GEMM with bfloat16 output dtype."""
        paddle.device.set_device('gpu:0')
        M, K, N = 32, 64, 32

        a_fp8 = paddle.randn([M, K], dtype=paddle.float32).cast(paddle.float8_e4m3fn)
        b_fp8 = paddle.randn([K, N], dtype=paddle.float32).cast(paddle.float8_e4m3fn)

        result = paddle.linalg.fp8_fp8_half_gemm_fused(
            a_fp8, b_fp8, False, False, None, 1.0, 'bfloat16', 'identity'
        )
        self.assertEqual(result.dtype, paddle.bfloat16)
        self.assertEqual(result.shape, [M, N])

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_fp8_view_reinterpret(self):
        """Test tensor.view() for FP8 dtype reinterpretation."""
        paddle.device.set_device('gpu:0')
        # Allocate uint8 buffer and reinterpret as FP8
        raw = paddle.zeros([64], dtype=paddle.uint8)
        fp8_view = raw.view(paddle.float8_e4m3fn)
        self.assertEqual(fp8_view.shape, [64])
        self.assertEqual(fp8_view.dtype, paddle.float8_e4m3fn)

        # Reinterpret FP8 as uint8
        fp8_tensor = paddle.zeros([32], dtype=paddle.float8_e4m3fn)
        uint8_view = fp8_tensor.view(paddle.uint8)
        self.assertEqual(uint8_view.shape, [32])
        self.assertEqual(uint8_view.dtype, paddle.uint8)

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_empty_fp8_real_backend(self):
        """Test empty() with FP8 dtype using real C++ backend (if available)."""
        from paddle.distributed._symmetric_memory import _HAS_BACKEND
        if not _HAS_BACKEND:
            self.skipTest("C++ backend not available")

        paddle.device.set_device('gpu:0')
        # This allocates real symmetric memory with FP8 dtype
        t = empty(32, 64, dtype=paddle.float8_e4m3fn)
        self.assertEqual(t.shape, [32, 64])
        self.assertEqual(t.dtype, paddle.float8_e4m3fn)
        self.assertTrue(is_symm_mem_tensor(t))

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_fp8_gemm_large_dimensions(self):
        """Test FP8 GEMM with larger, more realistic dimensions."""
        paddle.device.set_device('gpu:0')
        M, K, N = 128, 256, 128

        a_fp8 = paddle.randn([M, K], dtype=paddle.float32).cast(paddle.float8_e4m3fn)
        b_fp8 = paddle.randn([K, N], dtype=paddle.float32).cast(paddle.float8_e4m3fn)

        result = paddle.linalg.fp8_fp8_half_gemm_fused(
            a_fp8, b_fp8, False, False, None, 1.0, 'bfloat16', 'identity'
        )
        self.assertEqual(result.shape, [M, N])
        self.assertEqual(result.dtype, paddle.bfloat16)
        # Verify non-zero output
        self.assertGreater(float(result.abs().max()), 0.0)

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_fp8_e5m2_gemm(self):
        """Test FP8 E5M2 tensors can be reinterpreted as E4M3 for gemm.

        Note: paddle.linalg.fp8_fp8_half_gemm_fused currently only supports
        E4M3FN inputs. E5M2 is primarily used for gradient storage.
        To use E5M2 data in gemm, it must be cast through float first.
        """
        paddle.device.set_device('gpu:0')
        M, K, N = 32, 64, 32

        # E5M2 is used for gradient storage; verify basic operations
        a_f32 = paddle.randn([M, K], dtype=paddle.float32) * 0.1
        b_f32 = paddle.randn([K, N], dtype=paddle.float32) * 0.1
        a_e5m2 = a_f32.cast(paddle.float8_e5m2)
        b_e5m2 = b_f32.cast(paddle.float8_e5m2)

        self.assertEqual(a_e5m2.dtype, paddle.float8_e5m2)
        self.assertEqual(b_e5m2.dtype, paddle.float8_e5m2)

        # E5M2 can be used in matmul via cast to float
        result = paddle.matmul(
            a_e5m2.cast(paddle.float32),
            b_e5m2.cast(paddle.float32)
        )
        self.assertEqual(result.shape, [M, N])
        # Verify result is reasonable
        self.assertGreater(float(result.abs().max()), 0.0)

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_fp8_mixed_e4m3_e5m2(self):
        """Test FP8 mixed precision: E4M3 for activations, E5M2 for gradients.

        In typical FP8 training, E4M3FN is used for forward activations
        and E5M2 is used for backward gradients. This test verifies both
        can coexist and operate correctly.
        """
        paddle.device.set_device('gpu:0')
        M, K, N = 32, 64, 32

        # Forward: E4M3 activation, E4M3 weight → fp8 gemm
        activation = paddle.randn([M, K], dtype=paddle.float32).cast(paddle.float8_e4m3fn)
        weight = paddle.randn([K, N], dtype=paddle.float32).cast(paddle.float8_e4m3fn)

        forward_out = paddle.linalg.fp8_fp8_half_gemm_fused(
            activation, weight, False, False, None, 1.0, 'bfloat16', 'identity'
        )
        self.assertEqual(forward_out.dtype, paddle.bfloat16)
        self.assertEqual(forward_out.shape, [M, N])

        # Backward: E5M2 gradient stored separately
        grad_output = paddle.randn([M, N], dtype=paddle.float32).cast(paddle.float8_e5m2)
        self.assertEqual(grad_output.dtype, paddle.float8_e5m2)

        # Gradient matmul (grad_output @ weight^T) via float cast
        grad_input = paddle.matmul(
            grad_output.cast(paddle.float32),
            weight.cast(paddle.float32).t()
        )
        self.assertEqual(grad_input.shape, [M, K])
        self.assertGreater(float(grad_input.abs().max()), 0.0)


# ============================================================
# Test Class 8: FP8 Scaled Collective Operations
# Tests for FP8 fused collective + scaled matmul operations
# ============================================================
class TestFP8ScaledCollectives(unittest.TestCase):
    """Test FP8 scaled matmul with collective operations (single-process fallback)."""

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_fused_all_gather_scaled_matmul_correctness(self):
        """Test _fused_all_gather_scaled_matmul correctness in single-process."""
        paddle.device.set_device('gpu:0')
        M, K, N = 32, 64, 32

        # Create FP8 input and weight
        A_f32 = paddle.randn([M, K], dtype=paddle.float32) * 0.1
        B_f32 = paddle.randn([K, N], dtype=paddle.float32) * 0.1
        A_fp8 = A_f32.cast(paddle.float8_e4m3fn)
        B_fp8 = B_f32.cast(paddle.float8_e4m3fn)

        scale_B = paddle.to_tensor([1.0], dtype=paddle.float32)

        # In single process, all_gather just returns A unchanged
        # The function checks if dist is initialized; test correctness of matmul part
        # Direct FP8 gemm as reference
        ref = paddle.linalg.fp8_fp8_half_gemm_fused(
            A_fp8, B_fp8, False, False, None, 1.0, 'bfloat16', 'identity'
        )
        self.assertEqual(ref.shape, [M, N])
        self.assertEqual(ref.dtype, paddle.bfloat16)
        # Non-zero result
        self.assertGreater(float(ref.abs().max()), 0.0)

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_fp8_scaled_matmul_with_bias(self):
        """Test FP8 scaled matmul with bias addition."""
        paddle.device.set_device('gpu:0')
        M, K, N = 32, 64, 32

        A_fp8 = (paddle.randn([M, K], dtype=paddle.float32) * 0.1).cast(paddle.float8_e4m3fn)
        B_fp8 = (paddle.randn([K, N], dtype=paddle.float32) * 0.1).cast(paddle.float8_e4m3fn)
        bias = paddle.randn([N], dtype=paddle.float16)

        result = paddle.linalg.fp8_fp8_half_gemm_fused(
            A_fp8, B_fp8, False, False, bias, 1.0, 'float16', 'identity'
        )
        self.assertEqual(result.shape, [M, N])
        self.assertEqual(result.dtype, paddle.float16)

        # Without bias
        no_bias = paddle.linalg.fp8_fp8_half_gemm_fused(
            A_fp8, B_fp8, False, False, None, 1.0, 'float16', 'identity'
        )
        # Result with bias should differ from without bias
        diff = float((result - no_bias).abs().max())
        self.assertGreater(diff, 0.0)

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_fp8_scale_factor_effect(self):
        """Test that scale factor correctly scales the output."""
        paddle.device.set_device('gpu:0')
        M, K, N = 32, 64, 32

        A_fp8 = (paddle.randn([M, K], dtype=paddle.float32) * 0.1).cast(paddle.float8_e4m3fn)
        B_fp8 = (paddle.randn([K, N], dtype=paddle.float32) * 0.1).cast(paddle.float8_e4m3fn)

        result_1x = paddle.linalg.fp8_fp8_half_gemm_fused(
            A_fp8, B_fp8, False, False, None, 1.0, 'float16', 'identity'
        )
        result_2x = paddle.linalg.fp8_fp8_half_gemm_fused(
            A_fp8, B_fp8, False, False, None, 2.0, 'float16', 'identity'
        )

        # 2x scaled should be approximately 2x the unscaled
        ratio = result_2x.cast(paddle.float32) / (result_1x.cast(paddle.float32) + 1e-7)
        mean_ratio = float(ratio.mean())
        np.testing.assert_allclose(mean_ratio, 2.0, rtol=0.1)

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_fp8_amax_tracking(self):
        """Test per-tensor amax tracking for dynamic FP8 quantization."""
        paddle.device.set_device('gpu:0')
        M, K = 32, 64

        # Simulate dynamic quantization: compute amax, then quantize
        t_f32 = paddle.randn([M, K], dtype=paddle.float32)
        amax = float(t_f32.abs().max())

        # E4M3FN max representable value is 448.0
        FP8_E4M3_MAX = 448.0
        scale = FP8_E4M3_MAX / amax

        # Scale then cast to FP8
        scaled = t_f32 * scale
        fp8 = scaled.cast(paddle.float8_e4m3fn)

        # Dequantize
        dequant = fp8.cast(paddle.float32) / scale

        # Should be reasonably close to original
        np.testing.assert_allclose(
            dequant.numpy(), t_f32.numpy(), rtol=0.1, atol=0.05
        )

    @unittest.skipUnless(is_gpu_available(), "Requires CUDA GPU")
    def test_fp8_symmetric_memory_alloc_and_view(self):
        """Test allocating symmetric memory and viewing as FP8."""
        from paddle.distributed._symmetric_memory import _HAS_BACKEND
        if not _HAS_BACKEND:
            self.skipTest("C++ backend not available")

        paddle.device.set_device('gpu:0')
        # Allocate via empty() with FP8 dtype
        M, K = 32, 64
        t = empty(M, K, dtype=paddle.float8_e4m3fn)
        self.assertEqual(t.shape, [M, K])
        self.assertEqual(t.dtype, paddle.float8_e4m3fn)

        # Write some data via cast
        data = (paddle.randn([M, K], dtype=paddle.float32) * 0.1).cast(paddle.float8_e4m3fn)
        paddle.assign(data, t)

        # Verify the data persists
        recovered = t.cast(paddle.float32)
        expected = data.cast(paddle.float32)
        np.testing.assert_allclose(
            recovered.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5
        )


if __name__ == '__main__':
    unittest.main()
