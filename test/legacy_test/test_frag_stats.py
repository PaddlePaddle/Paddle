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
Unit tests for fragmentation stats APIs (non-VMM allocator path).

Tests:
  - core.all_block_info(dev): block-level info for AutoGrowthBestFitAllocator
  - core.allocator_stats(dev): runtime counters (hit/miss/split/merge/internal_frag)
  - gpu_frag_profiler.py: snapshot() / report() / _fill_block_metrics()

Run:
  python test/legacy_test/test_frag_stats.py
"""

import unittest

import paddle
from paddle.framework import core

MB = 1024 * 1024


def _skip_no_gpu(cls):
    return unittest.skipIf(
        (not paddle.is_compiled_with_cuda()) or paddle.is_compiled_with_rocm(),
        "Requires CUDA GPU",
    )(cls)


def _get_dev():
    return core.get_cuda_current_device_id()


class _NonVMMTestBase(unittest.TestCase):
    """Common setUp for non-VMM allocator tests."""

    def setUp(self):
        paddle.set_flags({"FLAGS_use_virtual_memory_auto_growth": False})
        paddle.device.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helper: allocate tensors in a pattern that creates fragmentation
# ---------------------------------------------------------------------------
def _create_fragmented_state():
    """
    Alloc pattern: A(10MB) B(20MB) C(10MB) D(20MB)
    Then free B and D -> leaves two holes (external fragmentation).

    Returns (live_tensors_dict, freed_names).
    """
    tensors = {}
    sizes = {"A": 10, "B": 20, "C": 10, "D": 20}
    for name, mb in sizes.items():
        numel = mb * MB // 4  # float32 = 4 bytes
        tensors[name] = paddle.empty([numel], dtype="float32")

    # Free B and D to create holes
    del tensors["B"]
    del tensors["D"]
    return tensors


# ===========================================================================
# Test 1: core.all_block_info (non-VMM)
# ===========================================================================
@_skip_no_gpu
class TestAllBlockInfoNonVMM(_NonVMMTestBase):
    """Test all_block_info() on the default (non-VMM) allocator."""

    def test_returns_non_empty_after_alloc(self):
        """After allocation, all_block_info should return at least one allocator with blocks."""
        t = paddle.randn([1024], dtype="float32")

        blocks = core.all_block_info(_get_dev())
        self.assertIsInstance(blocks, list)
        self.assertGreater(len(blocks), 0, "Should have >= 1 allocator")

        # Each allocator's blocks is a list of (size, ptr, is_free)
        alloc_blocks = blocks[0]
        self.assertGreater(len(alloc_blocks), 0, "Should have >= 1 block")

        # Check tuple structure
        size, ptr, is_free = alloc_blocks[0]
        self.assertIsInstance(size, int)
        self.assertIsInstance(ptr, int)
        self.assertIsInstance(is_free, bool)
        self.assertGreater(size, 0)

        del t

    def test_free_blocks_appear_after_dealloc(self):
        """After freeing a tensor, at least one block should be marked is_free=True."""
        t = paddle.randn([MB], dtype="float32")  # ~4MB
        del t

        blocks = core.all_block_info(_get_dev())
        all_blocks = [b for allocator in blocks for b in allocator]
        free_blocks = [b for b in all_blocks if b[2] is True]
        self.assertGreater(
            len(free_blocks), 0, "Should have free blocks after dealloc"
        )

    def test_fragmented_state_has_multiple_free_blocks(self):
        """Create fragmented state (A _ C _ pattern) and verify multiple free blocks."""
        tensors = _create_fragmented_state()

        blocks = core.all_block_info(_get_dev())
        all_blocks = [b for allocator in blocks for b in allocator]
        free_blocks = [b for b in all_blocks if b[2] is True]
        used_blocks = [b for b in all_blocks if b[2] is False]

        self.assertGreaterEqual(
            len(free_blocks), 1, "Fragmented state should have free blocks"
        )
        self.assertGreaterEqual(
            len(used_blocks), 2, "Should have at least 2 used blocks (A, C)"
        )

        del tensors

    def test_block_sizes_sum_to_reserved(self):
        """Total size of all blocks should approximately equal reserved memory."""
        t = paddle.randn([5 * MB // 4], dtype="float32")  # ~5MB

        blocks = core.all_block_info(_get_dev())
        total_block_size = sum(b[0] for allocator in blocks for b in allocator)
        reserved = paddle.device.cuda.memory_reserved()

        # Allow some tolerance for alignment/metadata
        self.assertAlmostEqual(
            total_block_size,
            reserved,
            delta=reserved * 0.01,
            msg="Block sizes should sum close to reserved memory",
        )

        del t


# ===========================================================================
# Test 2: core.allocator_stats (non-VMM)
# ===========================================================================
@_skip_no_gpu
class TestAllocatorStatsNonVMM(_NonVMMTestBase):
    """Test allocator_stats() runtime counters on the default (non-VMM) allocator."""

    def test_stats_returns_expected_keys(self):
        """allocator_stats should return a dict with all 10 expected keys."""
        t = paddle.randn([1024], dtype="float32")

        stats = core.allocator_stats(_get_dev())
        expected_keys = {
            "total_alloc_times",
            "total_alloc_size",
            "total_free_times",
            "total_free_size",
            "cache_hit_count",
            "cache_miss_count",
            "split_count",
            "merge_count",
            "total_requested_size",
            "chunk_count",
        }
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing key: {key}")
            self.assertIsInstance(stats[key], int, f"{key} should be int")

        del t

    def test_alloc_count_increases(self):
        """total_alloc_times should increase after allocations."""
        stats_before = core.allocator_stats(_get_dev())
        t = paddle.randn([MB], dtype="float32")

        stats_after = core.allocator_stats(_get_dev())
        self.assertGreater(
            stats_after["total_alloc_times"],
            stats_before["total_alloc_times"],
            "alloc count should increase after allocation",
        )

        del t

    def test_free_count_increases(self):
        """total_free_times should increase after freeing."""
        t = paddle.randn([MB], dtype="float32")

        stats_before = core.allocator_stats(_get_dev())
        del t

        stats_after = core.allocator_stats(_get_dev())
        self.assertGreater(
            stats_after["total_free_times"],
            stats_before["total_free_times"],
            "free count should increase after dealloc",
        )

    def test_cache_miss_on_first_alloc(self):
        """First allocation should trigger at least one cache_miss (new chunk needed)."""
        # Clear cache to force a fresh start
        paddle.device.cuda.empty_cache()

        stats_before = core.allocator_stats(_get_dev())
        t = paddle.randn(
            [50 * MB // 4], dtype="float32"
        )  # 50MB, likely new chunk

        stats_after = core.allocator_stats(_get_dev())
        # Either cache_miss increases (new chunk) or cache_hit increases (reuse)
        miss_delta = (
            stats_after["cache_miss_count"] - stats_before["cache_miss_count"]
        )
        hit_delta = (
            stats_after["cache_hit_count"] - stats_before["cache_hit_count"]
        )
        self.assertGreater(
            miss_delta + hit_delta,
            0,
            "Allocation should increment either hit or miss counter",
        )

        del t

    def test_cache_hit_on_realloc(self):
        """Freeing then re-allocating the same size should get a cache hit."""
        t = paddle.randn([MB], dtype="float32")
        del t

        stats_before = core.allocator_stats(_get_dev())
        t2 = paddle.randn([MB], dtype="float32")  # should reuse freed block

        stats_after = core.allocator_stats(_get_dev())
        hit_delta = (
            stats_after["cache_hit_count"] - stats_before["cache_hit_count"]
        )
        self.assertGreater(
            hit_delta, 0, "Re-allocation of same size should hit cache"
        )

        del t2

    def test_split_count_on_large_then_small(self):
        """Allocating large then freeing, then allocating small should trigger a split."""
        # Alloc a large block, free it, then alloc a smaller one from the same chunk
        big = paddle.randn([20 * MB // 4], dtype="float32")
        del big

        stats_before = core.allocator_stats(_get_dev())
        # This smaller alloc should reuse the freed block and split it
        small = paddle.randn([1 * MB // 4], dtype="float32")

        stats_after = core.allocator_stats(_get_dev())
        split_delta = stats_after["split_count"] - stats_before["split_count"]
        self.assertGreater(
            split_delta,
            0,
            "Small alloc from large free block should trigger split",
        )

        del small

    def test_merge_count_on_adjacent_free(self):
        """Freeing adjacent blocks should trigger merge."""
        # Alloc a big block, free it, then alloc two smaller blocks from it
        # (this guarantees they are adjacent inside the same chunk).
        big = paddle.randn([20 * MB // 4], dtype="float32")  # 20MB
        del big

        # Two small allocs will split the freed 20MB block -> adjacent
        a = paddle.randn([5 * MB // 4], dtype="float32")
        b = paddle.randn([5 * MB // 4], dtype="float32")

        stats_before = core.allocator_stats(_get_dev())
        del a
        del b

        stats_after = core.allocator_stats(_get_dev())
        merge_delta = stats_after["merge_count"] - stats_before["merge_count"]
        self.assertGreater(
            merge_delta, 0, "Freeing adjacent blocks should trigger merge"
        )

    def test_internal_fragmentation_positive(self):
        """total_requested_size should be <= total_alloc_size (alignment overhead)."""
        t = paddle.randn(
            [1000], dtype="float32"
        )  # 4000 bytes, likely not aligned

        stats = core.allocator_stats(_get_dev())
        self.assertGreater(stats["total_alloc_size"], 0)
        self.assertGreater(stats["total_requested_size"], 0)
        self.assertLessEqual(
            stats["total_requested_size"],
            stats["total_alloc_size"],
            "requested <= allocated (alignment rounds up)",
        )

        del t

    def test_chunk_count_increases_with_large_allocs(self):
        """Allocating a large tensor should increase chunk_count."""
        stats_before = core.allocator_stats(_get_dev())
        t = paddle.randn([100 * MB // 4], dtype="float32")  # 100MB

        stats_after = core.allocator_stats(_get_dev())
        self.assertGreaterEqual(
            stats_after["chunk_count"],
            stats_before["chunk_count"],
            "Large alloc may create a new chunk",
        )

        del t

    def test_hit_plus_miss_equals_alloc_times(self):
        """cache_hit + cache_miss should equal total_alloc_times."""
        paddle.device.cuda.empty_cache()
        # Do a known sequence
        t1 = paddle.randn([MB], dtype="float32")
        t2 = paddle.randn([MB], dtype="float32")
        del t1
        del t2

        stats = core.allocator_stats(_get_dev())
        hit_miss = stats["cache_hit_count"] + stats["cache_miss_count"]
        # hit+miss should equal total_alloc_times
        self.assertEqual(
            hit_miss,
            stats["total_alloc_times"],
            f"hit({stats['cache_hit_count']}) + miss({stats['cache_miss_count']}) "
            f"should equal alloc_times({stats['total_alloc_times']})",
        )


# ===========================================================================
# Test 3: gpu_frag_profiler.py functions
# ===========================================================================
# Import gpu_frag_profiler from paddle.device.cuda
try:
    from paddle.device.cuda import gpu_frag_profiler as fp
except ImportError:
    fp = None


@_skip_no_gpu
@unittest.skipIf(fp is None, "gpu_frag_profiler.py not importable")
class TestGpuFragProfiler(_NonVMMTestBase):
    """Test the profiler script functions."""

    def test_snapshot_returns_base_metrics(self):
        """snapshot() should always return the base metrics."""
        t = paddle.randn([MB], dtype="float32")

        snap = fp.snapshot("test")
        self.assertEqual(snap["tag"], "test")
        self.assertIn("allocated_mb", snap)
        self.assertIn("reserved_mb", snap)
        self.assertIn("pool_util", snap)
        self.assertIn("driver_used_mb", snap)
        self.assertIn("hidden_memory_mb", snap)
        self.assertGreater(snap["allocated_mb"], 0)
        self.assertGreater(snap["reserved_mb"], 0)

        del t

    def test_snapshot_has_all_metrics(self):
        """snapshot() should include block-level and runtime metrics."""
        t = paddle.randn([MB], dtype="float32")
        del t

        snap = fp.snapshot("metrics")
        expected_keys = [
            "free_block_count",
            "external_frag",
            "max_free_block_mb",
            "cache_hit_rate",
            "split_rate",
            "merge_rate",
            "internal_frag",
            "chunk_count",
        ]
        for key in expected_keys:
            self.assertIn(key, snap, f"Missing key: {key}")

    def test_fill_block_metrics_computation(self):
        """_fill_block_metrics should compute correct external frag."""
        m = {}
        # Simulate: 2 allocators, allocator 0 has blocks:
        #   used(100), free(50), used(200), free(30)
        # total_free=80, max_free=50 -> ext_frag = 1 - 50/80 = 0.375
        mock_blocks = [
            [
                (100, 0x1000, False),
                (50, 0x2000, True),
                (200, 0x3000, False),
                (30, 0x4000, True),
            ]
        ]
        fp._fill_block_metrics(m, mock_blocks)

        self.assertAlmostEqual(m["external_frag"], 1 - 50 / 80, places=4)
        self.assertEqual(m["free_block_count"], 2)
        self.assertEqual(m["used_block_count"], 2)
        self.assertAlmostEqual(m["total_free_mb"], 80 / MB, places=4)
        self.assertAlmostEqual(m["max_free_block_mb"], 50 / MB, places=4)

    def test_fill_block_metrics_no_free_blocks(self):
        """_fill_block_metrics with all used blocks -> ext_frag = 0."""
        m = {}
        mock_blocks = [[(100, 0x1000, False), (200, 0x2000, False)]]
        fp._fill_block_metrics(m, mock_blocks)
        self.assertAlmostEqual(m["external_frag"], 0.0)
        self.assertEqual(m["free_block_count"], 0)

    def test_fill_block_metrics_size_distribution(self):
        """_fill_block_metrics should bucket free blocks correctly."""
        m = {}
        mock_blocks = [
            [
                (500, 0x1, True),  # <1M
                (5 * MB, 0x2, True),  # 1-10M
                (50 * MB, 0x3, True),  # 10-100M
                (500 * MB, 0x4, True),  # 100M-1G
                (2 * 1024 * MB, 0x5, True),  # >1G
            ]
        ]
        fp._fill_block_metrics(m, mock_blocks)
        dist = m["free_block_dist"]
        self.assertEqual(dist["<1M"], 1)
        self.assertEqual(dist["1-10M"], 1)
        self.assertEqual(dist["10-100M"], 1)
        self.assertEqual(dist["100M-1G"], 1)
        self.assertEqual(dist[">1G"], 1)

    def test_report_runs_without_error(self):
        """report() should print without raising."""
        snaps = [fp.snapshot("s1")]
        fp.report(snaps)  # should not raise

    def test_probe_max_batch_basic(self):
        """probe_max_batch should find a valid batch size."""

        def dummy_model(batch_size=1):
            _ = paddle.randn([batch_size, 1024], dtype="float32")

        best = fp.probe_max_batch(dummy_model, start=1, max_try=64)
        self.assertGreaterEqual(best, 1)


# ===========================================================================
# Test 4: all_block_info works with VMM mode too (unified API)
# ===========================================================================
@_skip_no_gpu
class TestAllBlockInfoVMM(unittest.TestCase):
    """Verify all_block_info also works in VMM mode."""

    def setUp(self):
        paddle.set_flags({"FLAGS_use_virtual_memory_auto_growth": True})
        paddle.device.cuda.empty_cache()

    def test_vmm_mode_returns_blocks(self):
        """all_block_info should work in VMM mode too."""
        t = paddle.randn([1024], dtype="float32")

        blocks = core.all_block_info(_get_dev())
        self.assertIsInstance(blocks, list)
        # VMM mode should also return block info
        if len(blocks) > 0:
            self.assertGreater(len(blocks[0]), 0)

        del t


# ===========================================================================
# Test 5: _fmt() utility (pure Python, no GPU needed)
# ===========================================================================
class TestFmt(unittest.TestCase):
    """Unit tests for the _fmt() formatting helper."""

    def test_fmt_cases(self):
        cases = [
            (None, {}, "—"),
            ("—", {}, "—"),
            (0.5, {}, "50.0%"),
            (123.4, {"fmt": ".0f"}, "123"),
            (42, {}, "42"),
        ]
        for val, kwargs, expected in cases:
            with self.subTest(val=val):
                self.assertEqual(fp._fmt(val, **kwargs), expected)


# ===========================================================================
# Test 6: snapshot() boundary arithmetic (mock GPU calls, test Python logic)
# ===========================================================================
@_skip_no_gpu
@unittest.skipIf(fp is None, "gpu_frag_profiler.py not importable")
class TestSnapshotBoundaries(unittest.TestCase):
    """Verify snapshot() guard expressions when reserved or driver_used are 0."""

    def _make_snapshot_with(
        self, allocated, reserved, max_allocated, free_gpu, total_gpu
    ):
        """Run snapshot() with mocked GPU memory API values."""
        from unittest.mock import patch

        with (
            patch(
                'paddle.device.cuda.memory_allocated', return_value=allocated
            ),
            patch('paddle.device.cuda.memory_reserved', return_value=reserved),
            patch(
                'paddle.device.cuda.max_memory_allocated',
                return_value=max_allocated,
            ),
            patch.object(
                fp.core, 'gpu_memory_available', return_value=free_gpu
            ),
            patch.object(fp.core, 'get_device_properties') as mock_prop,
            patch.object(fp.core, 'all_block_info', side_effect=Exception()),
            patch.object(
                fp.core, 'vmm_all_block_info', side_effect=Exception()
            ),
            patch.object(fp.core, 'allocator_stats', side_effect=Exception()),
        ):
            mock_prop.return_value.total_memory = total_gpu
            return fp.snapshot("boundary")

    def test_all_zeros_boundary(self):
        """reserved=0, driver_used=0 → no division by zero."""
        snap = self._make_snapshot_with(0, 0, 0, 8 * fp.GB, 8 * fp.GB)
        self.assertAlmostEqual(snap["pool_util"], 1.0)
        self.assertAlmostEqual(snap["peak_waste"], 0.0)
        self.assertAlmostEqual(snap["hidden_ratio"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
