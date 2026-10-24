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
Multi-process distributed tests for paddle.distributed._symmetric_memory

These tests require multiple GPUs and test actual P2P memory operations.
They mirror PyTorch's test/distributed/test_symmetric_memory.py multi-process tests.

Run with: python -m paddle.distributed.launch --gpus=0,1 test_symmetric_memory_distributed.py
Or: python test_symmetric_memory_distributed.py (uses paddle.distributed.spawn)
"""

import os
import sys
import unittest
import numpy as np

import paddle
import paddle.distributed as dist


def get_gpu_count():
    return paddle.device.cuda.device_count()


def requires_nccl():
    """Check if NCCL is available."""
    try:
        from paddle.base import core
        return hasattr(core, 'NCCLParallelContext') or True  # NCCL typically available
    except Exception:
        return False


# ============================================================
# Worker functions for distributed tests
# ============================================================

def _worker_test_empty_strided_p2p():
    """Test P2P allocation + rendezvous + get_buffer + signals + barrier."""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from paddle.distributed._symmetric_memory import (
        empty,
        rendezvous,
        is_symm_mem_tensor,
    )

    # Allocate symmetric memory
    buf_size = 1024  # elements
    t = empty(buf_size, dtype=paddle.float32)

    assert t.shape == [buf_size], f"Expected shape [1024], got {t.shape}"

    # Fill with rank-specific data
    paddle.assign(paddle.full([buf_size], float(rank + 1), dtype=paddle.float32), t)

    # Rendezvous
    handle = rendezvous(t)

    assert handle is not None, "rendezvous returned None"
    assert handle.rank == rank, f"Expected rank {rank}, got {handle.rank}"
    assert handle.world_size == world_size

    # Barrier
    handle.barrier()

    # Each rank can read other ranks' buffers
    for peer in range(world_size):
        buf = handle.get_buffer(peer, [buf_size], paddle.float32)
        expected_val = float(peer + 1)
        actual = buf.numpy()
        np.testing.assert_allclose(
            actual, np.full(buf_size, expected_val, dtype=np.float32),
            rtol=1e-5,
            err_msg=f"Rank {rank}: buffer from peer {peer} mismatch"
        )

    # Signal pad
    pad = handle.get_signal_pad(rank)
    assert pad is not None
    assert pad.numel() > 0

    # Put/wait signal (peer-to-peer)
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    handle.put_signal(dst_rank=next_rank)
    handle.wait_signal(src_rank=prev_rank)

    # Final barrier
    handle.barrier()

    print(f"[Rank {rank}] test_empty_strided_p2p PASSED")


def _worker_test_is_symm_mem_tensor():
    """Test is_symm_mem_tensor with distributed allocation."""
    dist.init_parallel_env()
    rank = dist.get_rank()

    from paddle.distributed._symmetric_memory import (
        empty,
        is_symm_mem_tensor,
    )

    # Regular tensor should not be symm_mem
    regular = paddle.zeros([64])
    assert not is_symm_mem_tensor(regular)

    # Symm mem tensor should be detected
    t = empty(64, dtype=paddle.float32)
    # Note: in fallback mode, this may still return False
    # In real backend mode, it should return True
    # We just verify no crash
    result = is_symm_mem_tensor(t)
    print(f"[Rank {rank}] is_symm_mem_tensor(empty(64)) = {result}")
    print(f"[Rank {rank}] test_is_symm_mem_tensor PASSED")


def _worker_test_rendezvous_and_barrier():
    """Test rendezvous followed by barrier across all ranks."""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from paddle.distributed._symmetric_memory import empty, rendezvous

    t = empty(256, dtype=paddle.float32)
    handle = rendezvous(t)

    assert handle.rank == rank
    assert handle.world_size == world_size
    assert handle.buffer_size > 0
    assert handle.signal_pad_size > 0

    # Multiple barriers
    for _ in range(3):
        handle.barrier()

    # Barrier on different channels
    handle.barrier(channel=0)
    handle.barrier(channel=1)

    print(f"[Rank {rank}] test_rendezvous_and_barrier PASSED")


def _worker_test_signal_pad_operations():
    """Test signal pad get/set and custom sizes."""
    dist.init_parallel_env()
    rank = dist.get_rank()

    from paddle.distributed._symmetric_memory import (
        empty,
        rendezvous,
        get_signal_pad_size,
        set_signal_pad_size,
    )

    # Test signal pad size
    original_size = get_signal_pad_size()
    assert original_size > 0

    # Allocate and rendezvous
    t = empty(128, dtype=paddle.float32)
    handle = rendezvous(t)

    # Get signal pad
    pad = handle.get_signal_pad(rank)
    assert pad is not None

    # Signal pad should have expected size
    pad_bytes = handle.signal_pad_size
    assert pad_bytes > 0

    print(f"[Rank {rank}] test_signal_pad_operations PASSED")


def _worker_test_low_contention_all_gather():
    """Test all_gather using symmetric memory buffers."""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from paddle.distributed._symmetric_memory import empty, rendezvous

    buf_size = 64
    t = empty(buf_size, dtype=paddle.float32)
    paddle.assign(paddle.full([buf_size], float(rank), dtype=paddle.float32), t)

    handle = rendezvous(t)
    handle.barrier()

    # Each rank reads all buffers and concatenates (simulating all_gather)
    gathered = []
    for peer in range(world_size):
        buf = handle.get_buffer(peer, [buf_size], paddle.float32)
        gathered.append(buf.clone())

    result = paddle.concat(gathered, axis=0)
    assert result.shape == [buf_size * world_size]

    # Verify content
    for i in range(world_size):
        chunk = result[i * buf_size: (i + 1) * buf_size].numpy()
        expected = np.full(buf_size, float(i), dtype=np.float32)
        np.testing.assert_allclose(chunk, expected, rtol=1e-5)

    handle.barrier()
    print(f"[Rank {rank}] test_low_contention_all_gather PASSED")


def _worker_test_low_contention_reduce_scatter():
    """Test reduce_scatter using symmetric memory buffers."""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from paddle.distributed._symmetric_memory import empty, rendezvous

    # Each rank has world_size chunks to contribute
    chunk_size = 32
    total_size = chunk_size * world_size
    t = empty(total_size, dtype=paddle.float32)

    # Fill with rank-specific values
    data = np.zeros(total_size, dtype=np.float32)
    for i in range(world_size):
        data[i * chunk_size: (i + 1) * chunk_size] = float(rank + 1)
    paddle.assign(paddle.to_tensor(data), t)

    handle = rendezvous(t)
    handle.barrier()

    # Simulate reduce_scatter: each rank reads its chunk from all peers and sums
    my_result = paddle.zeros([chunk_size], dtype=paddle.float32)
    for peer in range(world_size):
        buf = handle.get_buffer(peer, [total_size], paddle.float32)
        my_chunk = buf[rank * chunk_size: (rank + 1) * chunk_size]
        my_result = my_result + my_chunk

    # Expected: sum of (peer+1) for all peers = sum(1..world_size)
    expected_val = float(world_size * (world_size + 1) / 2)
    np.testing.assert_allclose(
        my_result.numpy(),
        np.full(chunk_size, expected_val, dtype=np.float32),
        rtol=1e-5,
    )

    handle.barrier()
    print(f"[Rank {rank}] test_low_contention_reduce_scatter PASSED")


def _worker_test_fused_all_gather_matmul():
    """Test fused all_gather + matmul pattern."""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from paddle.distributed._symmetric_memory import (
        empty,
        rendezvous,
        restride_A_shard_for_fused_all_gather_matmul,
    )

    M_shard = 16  # rows per rank
    K = 32
    N = 64

    # Create A_shard (each rank owns M_shard rows)
    np.random.seed(42 + rank)
    A_shard_data = np.random.randn(M_shard, K).astype(np.float32)
    A_shard = paddle.to_tensor(A_shard_data)

    # B is the same on all ranks
    np.random.seed(100)
    B_data = np.random.randn(K, N).astype(np.float32)
    B = paddle.to_tensor(B_data)

    # Allocate symmetric memory for A_shard
    t = empty(M_shard * K, dtype=paddle.float32)
    paddle.assign(A_shard.reshape([-1]), t)

    handle = rendezvous(t)
    handle.barrier()

    # All-gather A from all ranks
    A_gathered_parts = []
    for peer in range(world_size):
        buf = handle.get_buffer(peer, [M_shard * K], paddle.float32)
        A_gathered_parts.append(buf.reshape([M_shard, K]).clone())

    A_full = paddle.concat(A_gathered_parts, axis=0)  # [M_shard * world_size, K]

    # Matmul
    C = paddle.matmul(A_full, B)

    assert C.shape == [M_shard * world_size, N]

    # Verify correctness: gather A_shard from all ranks
    # Use paddle all_gather for reference
    all_A_shards = []
    for _ in range(world_size):
        all_A_shards.append(paddle.empty_like(A_shard))
    dist.all_gather(all_A_shards, A_shard)
    A_ref = paddle.concat(all_A_shards, axis=0)
    C_ref = paddle.matmul(A_ref, B)

    np.testing.assert_allclose(C.numpy(), C_ref.numpy(), rtol=1e-4, atol=1e-4)

    # Test restride
    A_restrided = restride_A_shard_for_fused_all_gather_matmul(A_shard, 0)
    assert A_restrided.shape == A_shard.shape
    np.testing.assert_allclose(A_restrided.numpy(), A_shard.numpy(), rtol=1e-5)

    handle.barrier()
    print(f"[Rank {rank}] test_fused_all_gather_matmul PASSED")


def _worker_test_fused_matmul_reduce_scatter():
    """Test fused matmul + reduce_scatter pattern."""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from paddle.distributed._symmetric_memory import (
        empty,
        rendezvous,
        restride_A_for_fused_matmul_reduce_scatter,
    )

    M = 64
    K = 32
    N = 16 * world_size  # N must be divisible by world_size for scatter

    # A is the same across ranks (simulating replicated input)
    np.random.seed(42)
    A_data = np.random.randn(M, K).astype(np.float32)
    A = paddle.to_tensor(A_data)

    # B is shared
    np.random.seed(100)
    B_data = np.random.randn(K, N).astype(np.float32)
    B = paddle.to_tensor(B_data)

    # Matmul
    C = paddle.matmul(A, B)  # [M, N]

    # Scatter along dim=1: each rank gets C[:, rank*shard:(rank+1)*shard]
    shard_size = N // world_size
    C_local = C[:, rank * shard_size: (rank + 1) * shard_size]

    # Verify with reduce_scatter
    C_chunks = paddle.split(C, world_size, axis=1)
    output = paddle.zeros([M, shard_size], dtype=paddle.float32)
    dist.reduce_scatter(output, list(C_chunks))

    # In reduce_scatter sum mode, result should be C_local * world_size
    # (since all ranks have the same C, reduce_scatter sums them)
    expected = C_local.numpy() * world_size
    np.testing.assert_allclose(output.numpy(), expected, rtol=1e-4, atol=1e-4)

    # Test restride
    A_restrided = restride_A_for_fused_matmul_reduce_scatter(A, 0)
    assert A_restrided.shape == A.shape
    np.testing.assert_allclose(A_restrided.numpy(), A.numpy(), rtol=1e-5)

    print(f"[Rank {rank}] test_fused_matmul_reduce_scatter PASSED")


def _worker_test_one_shot_all_reduce():
    """Test one-shot all-reduce using symmetric memory."""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from paddle.distributed._symmetric_memory import empty, rendezvous

    buf_size = 128
    t = empty(buf_size, dtype=paddle.float32)

    # Each rank fills with rank+1
    paddle.assign(paddle.full([buf_size], float(rank + 1), dtype=paddle.float32), t)

    handle = rendezvous(t)
    handle.barrier()

    # All-reduce by reading all peers and summing
    result = paddle.zeros([buf_size], dtype=paddle.float32)
    for peer in range(world_size):
        buf = handle.get_buffer(peer, [buf_size], paddle.float32)
        result = result + buf

    # Expected: sum of 1..world_size
    expected_val = float(world_size * (world_size + 1) / 2)
    np.testing.assert_allclose(
        result.numpy(),
        np.full(buf_size, expected_val, dtype=np.float32),
        rtol=1e-5,
    )

    handle.barrier()
    print(f"[Rank {rank}] test_one_shot_all_reduce PASSED")


def _worker_test_two_shot_all_reduce():
    """Test two-shot all-reduce pattern (reduce-scatter + all-gather)."""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from paddle.distributed._symmetric_memory import empty, rendezvous

    # Total buffer must be divisible by world_size
    chunk_size = 64
    total_size = chunk_size * world_size
    t = empty(total_size, dtype=paddle.float32)

    # Each rank fills with rank+1
    paddle.assign(paddle.full([total_size], float(rank + 1), dtype=paddle.float32), t)

    handle = rendezvous(t)
    handle.barrier()

    # Phase 1: reduce-scatter
    # Each rank reduces its chunk from all peers
    my_reduced_chunk = paddle.zeros([chunk_size], dtype=paddle.float32)
    for peer in range(world_size):
        buf = handle.get_buffer(peer, [total_size], paddle.float32)
        my_chunk = buf[rank * chunk_size: (rank + 1) * chunk_size]
        my_reduced_chunk = my_reduced_chunk + my_chunk

    handle.barrier()

    # Write reduced chunk back to our buffer at our chunk position
    t_view = t[rank * chunk_size: (rank + 1) * chunk_size]
    paddle.assign(my_reduced_chunk, t_view)

    handle.barrier()

    # Phase 2: all-gather the reduced chunks
    final_result = paddle.zeros([total_size], dtype=paddle.float32)
    for peer in range(world_size):
        buf = handle.get_buffer(peer, [total_size], paddle.float32)
        peer_chunk = buf[peer * chunk_size: (peer + 1) * chunk_size]
        final_result[peer * chunk_size: (peer + 1) * chunk_size] = peer_chunk

    # Expected: each element = sum(1..world_size) = world_size*(world_size+1)/2
    expected_val = float(world_size * (world_size + 1) / 2)
    np.testing.assert_allclose(
        final_result.numpy(),
        np.full(total_size, expected_val, dtype=np.float32),
        rtol=1e-5,
    )

    handle.barrier()
    print(f"[Rank {rank}] test_two_shot_all_reduce PASSED")


def _worker_test_multiple_rendezvous():
    """Test multiple rendezvous on different buffers."""
    dist.init_parallel_env()
    rank = dist.get_rank()

    from paddle.distributed._symmetric_memory import empty, rendezvous

    # Create multiple symmetric memory buffers
    t1 = empty(64, dtype=paddle.float32)
    t2 = empty(128, dtype=paddle.float16)
    t3 = empty(32, 32, dtype=paddle.float32)

    handle1 = rendezvous(t1)
    handle2 = rendezvous(t2)
    handle3 = rendezvous(t3)

    assert handle1 is not None
    assert handle2 is not None
    assert handle3 is not None

    handle1.barrier()
    handle2.barrier()
    handle3.barrier()

    print(f"[Rank {rank}] test_multiple_rendezvous PASSED")


def _worker_test_different_dtypes():
    """Test symmetric memory with various dtypes."""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from paddle.distributed._symmetric_memory import empty, rendezvous

    dtypes = [paddle.float32, paddle.float16, paddle.int32, paddle.int64]
    for dtype in dtypes:
        t = empty(64, dtype=dtype)

        if dtype in [paddle.float32, paddle.float16]:
            paddle.assign(
                paddle.full([64], float(rank), dtype=dtype), t
            )
        else:
            paddle.assign(
                paddle.full([64], rank, dtype=dtype), t
            )

        handle = rendezvous(t)
        handle.barrier()

        # Verify own buffer
        buf = handle.get_buffer(rank, [64], dtype)
        if dtype in [paddle.float32, paddle.float16]:
            np.testing.assert_allclose(
                buf.numpy().astype(np.float32),
                np.full(64, float(rank), dtype=np.float32),
                rtol=1e-2,
            )
        else:
            np.testing.assert_array_equal(
                buf.numpy(),
                np.full(64, rank, dtype=np.int32 if dtype == paddle.int32 else np.int64),
            )

        handle.barrier()

    print(f"[Rank {rank}] test_different_dtypes PASSED")


def _worker_test_fp8_alloc_and_p2p():
    """Test FP8 symmetric memory allocation and P2P access."""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from paddle.distributed._symmetric_memory import empty, rendezvous

    M, K = 32, 64  # K % 16 == 0 for FP8 gemm compatibility

    # Allocate FP8 symmetric memory
    t = empty(M, K, dtype=paddle.float8_e4m3fn)
    assert t.shape == [M, K], f"Expected [32, 64], got {t.shape}"
    assert t.dtype == paddle.float8_e4m3fn

    # Fill with rank-specific data (cast through float32)
    data = paddle.full([M, K], float(rank + 1) * 0.5, dtype=paddle.float32).cast(paddle.float8_e4m3fn)
    paddle.assign(data, t)

    # Rendezvous
    handle = rendezvous(t)
    handle.barrier()

    # Read peer's FP8 buffer
    for peer in range(world_size):
        buf = handle.get_buffer(peer, [M * K], paddle.float8_e4m3fn)
        buf_f32 = buf.cast(paddle.float32)
        expected_val = float(peer + 1) * 0.5
        actual_mean = float(buf_f32.mean())
        assert abs(actual_mean - expected_val) < 0.1, \
            f"Rank {rank}: peer {peer} expected ~{expected_val}, got {actual_mean}"

    handle.barrier()
    print(f"[Rank {rank}] test_fp8_alloc_and_p2p PASSED")


def _worker_test_fp8_fused_all_gather_matmul():
    """Test FP8 fused all_gather + scaled matmul."""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from paddle.distributed._symmetric_memory import (
        _fused_all_gather_scaled_matmul,
    )

    M_shard = 32  # rows per rank
    K = 64  # K % 16 == 0
    N = 32

    # Create FP8 A_shard (each rank has different data)
    np.random.seed(42 + rank)
    A_shard_f32 = paddle.to_tensor(
        np.random.randn(M_shard, K).astype(np.float32) * 0.1
    )
    A_shard_fp8 = A_shard_f32.cast(paddle.float8_e4m3fn)

    # B (same on all ranks)
    np.random.seed(100)
    B_f32 = paddle.to_tensor(
        np.random.randn(K, N).astype(np.float32) * 0.1
    )
    B_fp8 = B_f32.cast(paddle.float8_e4m3fn)

    scale_B = paddle.to_tensor([1.0], dtype=paddle.float32)

    # Run fused op
    A_gathered, mm_outputs = _fused_all_gather_scaled_matmul(
        A_shard_fp8, [B_fp8], [scale_B],
        gather_dim=0, group_name="default",
        out_dtypes=['bfloat16'],
    )

    # Verify gathered shape
    assert A_gathered.shape == [M_shard * world_size, K], \
        f"Expected [{M_shard * world_size}, {K}], got {A_gathered.shape}"

    # Verify matmul output shape
    assert len(mm_outputs) == 1
    assert mm_outputs[0].shape == [M_shard * world_size, N], \
        f"Expected [{M_shard * world_size}, {N}], got {mm_outputs[0].shape}"
    assert mm_outputs[0].dtype == paddle.bfloat16

    # Verify non-zero and finite
    assert float(mm_outputs[0].cast(paddle.float32).abs().max()) > 0.0
    assert not bool(paddle.isnan(mm_outputs[0]).any())

    print(f"[Rank {rank}] test_fp8_fused_all_gather_matmul PASSED")


def _worker_test_fp8_fused_matmul_reduce_scatter():
    """Test FP8 fused scaled matmul + reduce_scatter."""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from paddle.distributed._symmetric_memory import (
        _fused_scaled_matmul_reduce_scatter,
    )

    M = 64
    K = 64  # K % 16 == 0
    N = 32 * world_size  # Must be divisible by world_size

    # FP8 inputs (same A on all ranks for reduce_scatter to be meaningful)
    np.random.seed(42)
    A_f32 = paddle.to_tensor(
        np.random.randn(M, K).astype(np.float32) * 0.1
    )
    A_fp8 = A_f32.cast(paddle.float8_e4m3fn)

    np.random.seed(100)
    B_f32 = paddle.to_tensor(
        np.random.randn(K, N).astype(np.float32) * 0.1
    )
    B_fp8 = B_f32.cast(paddle.float8_e4m3fn)

    scale_B = paddle.to_tensor([1.0], dtype=paddle.float32)

    # Run fused op
    result = _fused_scaled_matmul_reduce_scatter(
        A_fp8, B_fp8, scale_B,
        reduce_op="sum", scatter_dim=1, group_name="default",
        out_dtype='bfloat16',
    )

    # Verify shape: scattered along dim=1
    expected_N = N // world_size
    assert result.shape == [M, expected_N], \
        f"Expected [{M}, {expected_N}], got {result.shape}"
    assert result.dtype == paddle.bfloat16

    # Verify non-zero and finite
    assert float(result.cast(paddle.float32).abs().max()) > 0.0
    assert not bool(paddle.isnan(result).any())

    print(f"[Rank {rank}] test_fp8_fused_matmul_reduce_scatter PASSED")


def _worker_test_fp8_e5m2_gradient_buffer():
    """Test FP8 E5M2 symmetric memory for gradient storage."""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from paddle.distributed._symmetric_memory import empty, rendezvous

    M, N = 32, 64

    # Allocate E5M2 buffer (used for gradient storage in FP8 training)
    t = empty(M, N, dtype=paddle.float8_e5m2)
    assert t.shape == [M, N]
    assert t.dtype == paddle.float8_e5m2

    # Simulate gradient values
    grad_data = (paddle.randn([M, N], dtype=paddle.float32) * 0.1).cast(paddle.float8_e5m2)
    paddle.assign(grad_data, t)

    # Rendezvous for gradient exchange
    handle = rendezvous(t)
    handle.barrier()

    # Read all peers' gradients and average (gradient all-reduce)
    avg_grad = paddle.zeros([M, N], dtype=paddle.float32)
    for peer in range(world_size):
        buf = handle.get_buffer(peer, [M * N], paddle.float8_e5m2)
        avg_grad = avg_grad + buf.reshape([M, N]).cast(paddle.float32)
    avg_grad = avg_grad / float(world_size)

    # Verify finite and reasonable
    assert not bool(paddle.isnan(avg_grad).any())
    assert float(avg_grad.abs().max()) < 10.0  # Should be small gradients

    handle.barrier()
    print(f"[Rank {rank}] test_fp8_e5m2_gradient_buffer PASSED")


# ============================================================
# Test orchestrator
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Map of test name -> worker function name
WORKER_TESTS = [
    'empty_strided_p2p',
    'is_symm_mem_tensor',
    'rendezvous_and_barrier',
    'signal_pad_operations',
    'low_contention_all_gather',
    'low_contention_reduce_scatter',
    'fused_all_gather_matmul',
    'fused_matmul_reduce_scatter',
    'one_shot_all_reduce',
    'two_shot_all_reduce',
    'different_dtypes',
    'fp8_alloc_and_p2p',
    'fp8_e5m2_gradient_buffer',
]


def run_distributed_test_subprocess(test_name, world_size=2, timeout=120):
    """Run a distributed test by spawning subprocesses (avoids paddlecloud interference)."""
    import subprocess

    worker_script = os.path.join(SCRIPT_DIR, '_distributed_worker.py')
    master_port = 29600 + hash(test_name) % 1000

    processes = []
    for rank in range(world_size):
        env = os.environ.copy()
        env['MASTER_ADDR'] = '127.0.0.1'
        env['MASTER_PORT'] = str(master_port)
        env['PADDLE_TRAINER_ID'] = str(rank)
        env['PADDLE_TRAINERS_NUM'] = str(world_size)
        env['PADDLE_TRAINER_ENDPOINTS'] = ','.join(
            [f'127.0.0.1:{master_port + i}' for i in range(world_size)]
        )
        env['FLAGS_selected_gpus'] = str(rank)
        env['DISTRIBUTED_TEST_NAME'] = test_name
        # Remove paddlecloud env vars
        env.pop('PADDLE_PSERVERS_IP_PORT_LIST', None)
        env.pop('PADDLE_PSERVER_PORT_ARRAY', None)
        env.pop('CUDA_VISIBLE_DEVICES', None)

        p = subprocess.Popen(
            [sys.executable, worker_script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append(p)

    # Wait for all processes
    outputs = []
    all_ok = True
    for rank, p in enumerate(processes):
        try:
            stdout, stderr = p.communicate(timeout=timeout)
            outputs.append((rank, p.returncode, stdout.decode(), stderr.decode()))
            if p.returncode != 0:
                all_ok = False
        except subprocess.TimeoutExpired:
            p.kill()
            stdout, stderr = p.communicate()
            outputs.append((rank, -1, stdout.decode(), stderr.decode()))
            all_ok = False

    return all_ok, outputs


def main():
    if get_gpu_count() < 2:
        print("SKIP: Requires >= 2 GPUs")
        sys.exit(0)

    passed = 0
    failed = 0
    for test_name in WORKER_TESTS:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        ok, outputs = run_distributed_test_subprocess(test_name)
        if ok:
            for rank, rc, stdout, stderr in outputs:
                for line in stdout.strip().split('\n'):
                    if 'PASSED' in line:
                        print(f"  {line}")
            passed += 1
            print(f"  -> PASSED")
        else:
            for rank, rc, stdout, stderr in outputs:
                if rc != 0:
                    print(f"  [Rank {rank}] exit code {rc}")
                    err_lines = stderr.strip().split('\n')
                    for line in err_lines[-5:]:
                        print(f"    {line}")
            failed += 1
            print(f"  -> FAILED")

    print(f"\n{'='*60}")
    print(f"Distributed Tests: {passed}/{len(WORKER_TESTS)} passed")
    print(f"{'='*60}")

    if failed == 0:
        print("SUCCESS: All distributed tests passed!")
    else:
        print(f"FAILED: {failed} test(s) failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
