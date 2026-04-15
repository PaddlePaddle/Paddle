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
gpu_frag_profiler.py -- Paddle GPU fragmentation profiling tool.

Requires the C++ patches (AllBlockInfo / GetAllocatorStats) to be compiled.
Falls back gracefully to VMM-only API on unpatched builds.

Usage:
    import gpu_frag_profiler as fp
    snaps = []
    snaps.append(fp.snapshot("init"))
    # ... run some training steps ...
    snaps.append(fp.snapshot("step-100"))
    fp.report(snaps)
"""

import paddle
from paddle.framework import core

MB = 1024 * 1024
GB = 1024 * MB


def snapshot(tag=""):
    """Capture a fragmentation snapshot with full metrics."""
    dev = core.get_cuda_current_device_id()
    allocated = paddle.device.cuda.memory_allocated()
    reserved = paddle.device.cuda.memory_reserved()

    # Driver-level: total GPU memory used by this process
    free_gpu = core.gpu_memory_available()
    total_gpu = core.get_device_properties(dev).total_memory
    driver_used = total_gpu - free_gpu

    m = {
        "tag": tag,
        "allocated_mb": allocated / MB,
        "reserved_mb": reserved / MB,
        "pool_util": allocated / reserved if reserved > 0 else 1.0,
        "peak_waste": (
            (reserved - paddle.device.cuda.max_memory_allocated()) / reserved
            if reserved > 0
            else 0.0
        ),
        "driver_used_mb": driver_used / MB,
        "hidden_memory_mb": (driver_used - reserved) / MB,
        "hidden_ratio": (
            (driver_used - reserved) / driver_used if driver_used > 0 else 0
        ),
    }

    # block-level metrics (works for BOTH VMM and non-VMM after patch)
    try:
        blocks = core.all_block_info(dev)
        if blocks:
            _fill_block_metrics(m, blocks)
    except Exception:
        try:
            blocks = core.vmm_all_block_info(dev)
            if blocks:
                _fill_block_metrics(m, blocks)
        except Exception:
            pass

    # runtime counters (after patch)
    try:
        stats = core.allocator_stats(dev)
        hit = stats.get("cache_hit_count", 0)
        miss = stats.get("cache_miss_count", 0)
        alloc_size = stats.get("total_alloc_size", 0)
        req_size = stats.get("total_requested_size", 0)

        # Guard: if all counters are zero, the allocator doesn't support stats
        has_stats = (hit + miss + alloc_size) > 0
        if has_stats:
            m["cache_hit_rate"] = hit / max(hit + miss, 1)
            m["split_rate"] = stats.get("split_count", 0) / max(hit, 1)
            m["merge_rate"] = stats.get("merge_count", 0) / max(
                stats.get("total_free_times", 0), 1
            )
            m["internal_frag"] = 1 - req_size / max(alloc_size, 1)
            m["chunk_count"] = stats.get("chunk_count", 0)
        # else: leave keys absent, report() will show "—"
    except Exception:
        pass

    return m


def _fill_block_metrics(m, allocator_blocks):
    """Compute precise fragmentation from block info."""
    total_free, max_free, free_count, used_count = 0, 0, 0, 0
    free_sizes = []
    for alloc_blocks in allocator_blocks:
        for size, _ptr, is_free in alloc_blocks:
            if is_free:
                total_free += size
                max_free = max(max_free, size)
                free_count += 1
                free_sizes.append(size)
            else:
                used_count += 1

    m["external_frag"] = 1 - max_free / total_free if total_free > 0 else 0.0
    m["total_free_mb"] = total_free / MB
    m["max_free_block_mb"] = max_free / MB
    m["free_block_count"] = free_count
    m["used_block_count"] = used_count

    # size distribution
    buckets = {"<1M": 0, "1-10M": 0, "10-100M": 0, "100M-1G": 0, ">1G": 0}
    for s in free_sizes:
        if s < MB:
            buckets["<1M"] += 1
        elif s < 10 * MB:
            buckets["1-10M"] += 1
        elif s < 100 * MB:
            buckets["10-100M"] += 1
        elif s < GB:
            buckets["100M-1G"] += 1
        else:
            buckets[">1G"] += 1
    m["free_block_dist"] = buckets


def _fmt(val, fmt=".1%"):
    """Format a value; return dash for missing."""
    if val is None or val == "—":
        return "—"
    if isinstance(val, float):
        return f"{val:{fmt}}"
    return str(val)


def report(snapshots):
    """Print a summary table."""
    hdr = (
        f"{'Tag':>16} {'Alloc':>8} {'Rsrvd':>8} {'Driver':>8} {'Hidden':>8} "
        f"{'Pool%':>6} {'HidRt':>6} "
        f"{'ExtFrag':>7} {'IntFrag':>7} {'HitRt':>6} {'SplitRt':>7} "
        f"{'FreeBlk':>7} {'MaxFree':>8} {'Chunks':>6}"
    )
    sep = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{'-' * len(hdr)}")
    for s in snapshots:
        print(
            f"{s['tag']:>16} "
            f"{s['allocated_mb']:>7.0f}M {s['reserved_mb']:>7.0f}M "
            f"{_fmt(s.get('driver_used_mb'), '.0f'):>7}M "
            f"{_fmt(s.get('hidden_memory_mb'), '.0f'):>7}M "
            f"{_fmt(s.get('pool_util')):>6} "
            f"{_fmt(s.get('hidden_ratio')):>6} "
            f"{_fmt(s.get('external_frag')):>7} "
            f"{_fmt(s.get('internal_frag')):>7} "
            f"{_fmt(s.get('cache_hit_rate')):>6} "
            f"{_fmt(s.get('split_rate')):>7} "
            f"{s.get('free_block_count', '—')!s:>7} "
            f"{_fmt(s.get('max_free_block_mb'), '.0f'):>7}M "
            f"{s.get('chunk_count', '—')!s:>6}"
        )
    print(f"{sep}\n")


def probe_max_batch(model_fn, start=1, max_try=256):
    """Binary search for the max batch size before OOM.

    Uses step=1 to avoid skipping the true maximum (the original step=4
    could cause lo/hi to jump past valid values).
    """
    lo, hi, best = start, max_try, start
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            paddle.device.cuda.empty_cache()
            model_fn(batch_size=mid)
            best = mid
            lo = mid + 1
        except Exception:
            hi = mid - 1
        finally:
            paddle.device.cuda.empty_cache()
    return best


def measure_hidden_breakdown(model_fn, nccl_init_fn=None):
    """Step-by-step measurement to break down hidden memory.

    Returns dict with:
      - cuda_context_mb: CUDA context overhead
      - nccl_buffer_mb: NCCL communication buffer
      - cudnn_cublas_ws_mb: cuDNN/cuBLAS workspace
      - total_hidden_mb: total hidden memory
    """

    def _hidden():
        free, total = paddle.device.cuda.mem_get_info()
        reserved = paddle.device.cuda.memory_reserved()
        return (total - free) - reserved

    # Step 1: after CUDA context init
    _ = paddle.zeros([1])
    paddle.device.cuda.synchronize()
    h_ctx = _hidden()

    # Step 2: after NCCL init
    h_nccl = h_ctx
    if nccl_init_fn is not None:
        nccl_init_fn()
        paddle.device.cuda.synchronize()
        h_nccl = _hidden()

    # Step 3: after first forward (triggers cuDNN/cuBLAS workspace alloc)
    model_fn()
    paddle.device.cuda.synchronize()
    h_fwd = _hidden()

    return {
        "cuda_context_mb": h_ctx / MB,
        "nccl_buffer_mb": (h_nccl - h_ctx) / MB,
        "cudnn_cublas_ws_mb": (h_fwd - h_nccl) / MB,
        "total_hidden_mb": h_fwd / MB,
    }
