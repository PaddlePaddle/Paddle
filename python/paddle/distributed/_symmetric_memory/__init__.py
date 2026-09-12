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
Paddle Symmetric Memory API

This module provides symmetric memory allocation and P2P communication
primitives for multi-GPU distributed computing. It is inspired by
PyTorch's torch.distributed._symmetric_memory API.

Key functions:
- empty(): Allocate a tensor backed by symmetric memory
- is_symm_mem_tensor(): Check if a tensor is backed by symmetric memory
- rendezvous(): Establish symmetric memory association across ranks
- set_backend() / get_backend(): Manage symmetric memory backends
- set_signal_pad_size() / get_signal_pad_size(): Signal pad configuration
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional, Union

import paddle
import paddle.distributed as dist

__all__ = [
    "empty",
    "is_symm_mem_tensor",
    "rendezvous",
    "set_backend",
    "get_backend",
    "set_signal_pad_size",
    "get_signal_pad_size",
    "_fused_all_gather_scaled_matmul",
    "_fused_scaled_matmul_reduce_scatter",
]

# Try to import the C++ binding
try:
    from paddle.base import libpaddle
    _SymmetricMemoryAllocator = libpaddle._SymmetricMemoryAllocator
    _SymmetricMemoryHandle = libpaddle._SymmetricMemory
    _HAS_BACKEND = True
except (ImportError, AttributeError):
    _HAS_BACKEND = False

# Global state
_is_test_mode = False
_backend_name = "cuda"
_group_name_to_store = {}
_signal_pad_size = 65536  # default 64KB


@contextmanager
def _test_mode(group_names=None) -> Generator[None, None, None]:
    """
    Forces fallback implementations for testing without actual P2P hardware.
    """
    global _is_test_mode
    prev = _is_test_mode
    try:
        _is_test_mode = True
        yield
    finally:
        _is_test_mode = prev


def _get_allocator():
    """Get the SymmetricMemoryAllocator singleton."""
    if not _HAS_BACKEND:
        raise RuntimeError(
            "Symmetric memory backend is not available. "
            "Please ensure Paddle is compiled with CUDA and distributed support."
        )
    return _SymmetricMemoryAllocator.instance()


def _ensure_group_info(group=None):
    """Ensure group info is registered with the allocator."""
    # Use a fixed group name for the default group
    group_name = "default"

    if group_name not in _group_name_to_store:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Use the group's store for rendezvous
        store = dist.collective._default_store
        if store is None:
            raise RuntimeError(
                "Distributed store not initialized. "
                "Call paddle.distributed.init_parallel_env() first."
            )
        _get_allocator().set_group_info(group_name, rank, world_size, store)
        _group_name_to_store[group_name] = store

    return group_name


def empty(*size, dtype=None, device=None) -> paddle.Tensor:
    """
    Allocate a tensor backed by symmetric memory.

    The tensor is P2P accessible from all ranks in the group after rendezvous.

    Args:
        *size: Tensor shape (variable args or tuple)
        dtype: Data type (default: float32)
        device: Device (default: current GPU)

    Returns:
        paddle.Tensor: A tensor backed by symmetric memory
    """
    if dtype is None:
        dtype = paddle.float32

    # Flatten size args
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        shape = list(size[0])
    else:
        shape = list(size)

    # Calculate total bytes
    numel = 1
    for s in shape:
        numel *= s
    dtype_size = _dtype_to_bytes(dtype)
    total_bytes = numel * dtype_size

    if _is_test_mode or not _HAS_BACKEND:
        # Fallback: regular allocation
        tensor = paddle.zeros(shape, dtype=dtype)
        return tensor

    if device is None:
        from paddle.base import core
        device = paddle.CUDAPlace(core.get_cuda_current_device_id())

    # Get device id
    if isinstance(device, paddle.CUDAPlace):
        device_id = device.get_device_id()
    elif isinstance(device, int):
        device_id = device
    else:
        device_id = paddle.device.cuda.current_device()

    group_name = "default"  # must match _ensure_group_info
    allocator = _get_allocator()
    raw_dense_tensor = allocator.alloc(total_bytes, device_id, group_name)

    # raw_dense_tensor is a phi::DenseTensor (UINT8, shape=[total_bytes])
    # Create a paddle.Tensor sharing the same data, then view as target dtype/shape
    output = paddle.Tensor()
    output.get_tensor()._share_data_nocheck_with(raw_dense_tensor)
    # Now output is UINT8 with total_bytes elements; reinterpret as target dtype
    output = output.view(dtype).reshape(shape)
    return output


def is_symm_mem_tensor(tensor: paddle.Tensor) -> bool:
    """
    Check if a tensor is backed by symmetric memory.

    Args:
        tensor: The tensor to check

    Returns:
        bool: True if the tensor is backed by symmetric memory
    """
    if not _HAS_BACKEND:
        return False
    try:
        allocator = _get_allocator()
        # Need to pass the underlying DenseTensor to C++
        dense_tensor = tensor.get_tensor() if hasattr(tensor, 'get_tensor') else tensor
        return allocator.is_symm_mem_tensor(dense_tensor)
    except Exception:
        return False


def rendezvous(tensor: paddle.Tensor, group=None) -> Any:
    """
    Perform a collective rendezvous to establish symmetric memory association.

    All ranks must call this function with their local symmetric memory tensor.
    After rendezvous, each rank can access all other ranks' buffers.

    Args:
        tensor: A tensor allocated via symmetric_memory.empty()
        group: The process group (default: WORLD)

    Returns:
        SymmetricMemory handle with get_buffer(), barrier(), etc.
    """
    if _is_test_mode:
        return _FallbackSymmetricMemory(tensor)

    if not _HAS_BACKEND:
        raise RuntimeError("Symmetric memory backend not available")

    group_name = _ensure_group_info(group)
    allocator = _get_allocator()
    # Pass the underlying DenseTensor to C++
    dense_tensor = tensor.get_tensor() if hasattr(tensor, 'get_tensor') else tensor
    handle = allocator.rendezvous(dense_tensor)
    if handle is None:
        raise RuntimeError(
            "rendezvous failed: tensor is not backed by symmetric memory. "
            "Use symmetric_memory.empty() to allocate the tensor."
        )
    return _SymmetricMemoryWrapper(handle)


class _SymmetricMemoryWrapper:
    """Python wrapper around C++ SymmetricMemory handle.

    Converts DenseTensor returns to proper paddle.Tensor objects.
    """

    def __init__(self, handle):
        self._handle = handle

    @property
    def rank(self):
        return self._handle.rank

    @property
    def world_size(self):
        return self._handle.world_size

    @property
    def buffer_size(self):
        return self._handle.buffer_size

    @property
    def signal_pad_size(self):
        return self._handle.signal_pad_size

    def get_buffer(self, rank, sizes, dtype, storage_offset=0):
        """Get a tensor view of a peer's buffer."""
        from paddle.base import core
        # Convert paddle dtype to core.DataType if needed
        if isinstance(dtype, type) or hasattr(dtype, 'name'):
            # Already a DataType enum (paddle.float32, etc.)
            dt = dtype
        else:
            dt = paddle.float32
        dense = self._handle.get_buffer(rank, sizes, dt, storage_offset)
        # Wrap DenseTensor as paddle.Tensor
        output = paddle.Tensor()
        output.get_tensor()._share_data_nocheck_with(dense)
        return output

    def get_signal_pad(self, rank, sizes=None, dtype=None, storage_offset=0):
        """Get a tensor view of a peer's signal pad."""
        if sizes is None:
            sizes = []
        if dtype is None:
            dtype = paddle.int32  # default signal pad dtype
        dense = self._handle.get_signal_pad(rank, sizes, dtype, storage_offset)
        output = paddle.Tensor()
        output.get_tensor()._share_data_nocheck_with(dense)
        return output

    def barrier(self, channel=0, timeout_ms=0):
        """Perform a barrier across all ranks."""
        self._handle.barrier(channel, timeout_ms)

    def put_signal(self, dst_rank, channel=0, timeout_ms=0):
        """Signal a peer rank."""
        self._handle.put_signal(dst_rank, channel, timeout_ms)

    def wait_signal(self, src_rank, channel=0, timeout_ms=0):
        """Wait for a signal from a peer rank."""
        self._handle.wait_signal(src_rank, channel, timeout_ms)


def set_backend(name: str) -> None:
    """Set the symmetric memory backend name."""
    global _backend_name
    _backend_name = name


def get_backend(device=None) -> str:
    """Get the current symmetric memory backend name."""
    return _backend_name


def set_signal_pad_size(size: int) -> None:
    """
    Set the signal pad size for future allocations.

    Args:
        size: Size in bytes for the signal pad
    """
    global _signal_pad_size
    _signal_pad_size = size
    if _HAS_BACKEND:
        _SymmetricMemoryAllocator.set_signal_pad_size(size)


def get_signal_pad_size() -> int:
    """
    Get the current signal pad size.

    Returns:
        int: The signal pad size in bytes
    """
    if _HAS_BACKEND:
        return _SymmetricMemoryAllocator.get_signal_pad_size()
    return _signal_pad_size


def stream_write_value32(tensor: paddle.Tensor, offset: int, val: int) -> None:
    """
    Write a 32-bit value to a tensor using CUDA stream write.

    Args:
        tensor: Target uint32 tensor
        offset: Offset in elements
        val: 32-bit value to write
    """
    if _HAS_BACKEND:
        _SymmetricMemoryAllocator.stream_write_value32(tensor, offset, val)


def memset32(tensor: paddle.Tensor, offset: int, val: int, count: int) -> None:
    """
    Set count elements of a uint32 tensor to a value.

    Args:
        tensor: Target uint32 tensor (must be flat and contiguous)
        offset: Offset in elements
        val: 32-bit value to set
        count: Number of elements to set
    """
    if _HAS_BACKEND:
        _SymmetricMemoryAllocator.memset32(tensor, offset, val, count)


def has_multicast_support(device_type: str = "gpu", device_idx: int = 0) -> bool:
    """
    Check if the device supports multicast (NVLink SHARP).

    Returns:
        bool: False for now (multicast not yet implemented)
    """
    return False


# --- Collective operations (fallback implementations) ---

def _fused_all_gather_matmul_fallback(
    A_shard, Bs, gather_dim=0, group_name="0", return_A=True
):
    """
    Fallback implementation: all_gather(A_shard) then matmul with each B.
    """
    group = dist.collective._get_default_group()
    world_size = dist.get_world_size()

    # All-gather A_shard along gather_dim
    gather_list = []
    for _ in range(world_size):
        gather_list.append(paddle.empty_like(A_shard))
    dist.all_gather(gather_list, A_shard, group=group)
    A = paddle.concat(gather_list, axis=gather_dim)

    # Matmul with each B
    mm_outputs = []
    for B in Bs:
        mm_outputs.append(paddle.matmul(A, B))

    if return_A:
        return A, mm_outputs
    return A, mm_outputs


def _fused_matmul_reduce_scatter_fallback(
    A, B, reduce_op="avg", scatter_dim=0, group_name="0"
):
    """
    Fallback implementation: matmul(A, B) then reduce_scatter.
    """
    group = dist.collective._get_default_group()
    world_size = dist.get_world_size()

    C = paddle.matmul(A, B)

    # reduce_scatter
    output_shape = list(C.shape)
    output_shape[scatter_dim] //= world_size
    output = paddle.empty(output_shape, dtype=C.dtype)

    # Split and reduce
    chunks = paddle.split(C, world_size, axis=scatter_dim)
    dist.reduce_scatter(output, list(chunks), group=group)

    if reduce_op == "avg":
        output = output / world_size

    return output


def restride_A_shard_for_fused_all_gather_matmul(t, gather_dim):
    """
    Restride tensor for optimal layout in fused all_gather + matmul.
    Makes the tensor contiguous when moved along gather_dim.
    """
    return t.moveaxis(gather_dim, 0).contiguous().moveaxis(0, gather_dim)


def restride_A_for_fused_matmul_reduce_scatter(t, scatter_dim):
    """
    Restride tensor for optimal layout in fused matmul + reduce_scatter.
    Makes the tensor contiguous when moved along scatter_dim.
    """
    return t.moveaxis(scatter_dim, 0).contiguous().moveaxis(0, scatter_dim)


# --- Symmetric memory collective operations ---

def one_shot_all_reduce(handle, tensor, reduce_op="sum"):
    """
    Single-pass all-reduce using symmetric memory P2P reads.

    Each rank reads all peers' buffers and reduces them locally.
    The tensor must already be stored in symmetric memory.

    Args:
        handle: SymmetricMemory handle from rendezvous
        tensor: The local tensor (must be in symmetric memory)
        reduce_op: "sum" or "avg"

    Returns:
        paddle.Tensor: Reduced result
    """
    rank = handle.rank
    world_size = handle.world_size
    shape = list(tensor.shape)
    dtype = tensor.dtype
    numel = 1
    for s in shape:
        numel *= s

    # Barrier to ensure all ranks have written their data
    handle.barrier()

    # Read all peers and reduce
    result = paddle.zeros(shape, dtype=dtype)
    for peer in range(world_size):
        buf = handle.get_buffer(peer, [numel], dtype)
        result = result + buf.reshape(shape)

    if reduce_op == "avg":
        result = result / float(world_size)

    handle.barrier()
    return result


def two_shot_all_reduce(handle, tensor, reduce_op="sum"):
    """
    Two-pass all-reduce using symmetric memory (reduce-scatter + all-gather).

    More bandwidth-efficient for large tensors as each rank only
    processes 1/world_size of the data in each phase.

    Args:
        handle: SymmetricMemory handle from rendezvous
        tensor: The local tensor (must be in symmetric memory)
        reduce_op: "sum" or "avg"

    Returns:
        paddle.Tensor: Reduced result
    """
    rank = handle.rank
    world_size = handle.world_size
    shape = list(tensor.shape)
    dtype = tensor.dtype
    numel = 1
    for s in shape:
        numel *= s

    assert numel % world_size == 0, \
        f"Tensor numel ({numel}) must be divisible by world_size ({world_size})"
    chunk_size = numel // world_size

    # Barrier
    handle.barrier()

    # Phase 1: Reduce-scatter
    # Each rank reduces its assigned chunk from all peers
    my_chunk = paddle.zeros([chunk_size], dtype=dtype)
    for peer in range(world_size):
        buf = handle.get_buffer(peer, [numel], dtype)
        peer_chunk = buf[rank * chunk_size: (rank + 1) * chunk_size]
        my_chunk = my_chunk + peer_chunk

    if reduce_op == "avg":
        my_chunk = my_chunk / float(world_size)

    # Write reduced chunk back to our symmetric buffer
    flat_tensor = tensor.reshape([-1])
    flat_tensor[rank * chunk_size: (rank + 1) * chunk_size] = my_chunk

    handle.barrier()

    # Phase 2: All-gather the reduced chunks
    result = paddle.zeros([numel], dtype=dtype)
    for peer in range(world_size):
        buf = handle.get_buffer(peer, [numel], dtype)
        peer_chunk = buf[peer * chunk_size: (peer + 1) * chunk_size]
        result[peer * chunk_size: (peer + 1) * chunk_size] = peer_chunk

    handle.barrier()
    return result.reshape(shape)


def symm_reduce_scatter(handle, tensor, reduce_op="sum", scatter_dim=0):
    """
    Reduce-scatter using symmetric memory P2P reads.

    Args:
        handle: SymmetricMemory handle from rendezvous
        tensor: Input tensor, first dim must be divisible by world_size
        reduce_op: "sum" or "avg"
        scatter_dim: Dimension to scatter along

    Returns:
        paddle.Tensor: Scatter-reduced result (1/world_size of original)
    """
    rank = handle.rank
    world_size = handle.world_size
    shape = list(tensor.shape)
    dtype = tensor.dtype

    assert shape[scatter_dim] % world_size == 0, \
        f"Dim {scatter_dim} size ({shape[scatter_dim]}) must be divisible by world_size ({world_size})"

    chunk_size = shape[scatter_dim] // world_size
    numel = 1
    for s in shape:
        numel *= s

    handle.barrier()

    # Each rank reduces its chunk from all peers
    # Using moveaxis to bring scatter_dim to front for easy chunking
    out_shape = list(shape)
    out_shape[scatter_dim] = chunk_size
    result = paddle.zeros(out_shape, dtype=dtype)

    for peer in range(world_size):
        buf = handle.get_buffer(peer, [numel], dtype)
        peer_tensor = buf.reshape(shape)
        # Extract our chunk along scatter_dim
        slices = [slice(None)] * len(shape)
        slices[scatter_dim] = slice(rank * chunk_size, (rank + 1) * chunk_size)
        peer_chunk = peer_tensor[tuple(slices)]
        result = result + peer_chunk

    if reduce_op == "avg":
        result = result / float(world_size)

    handle.barrier()
    return result


def symm_all_gather(handle, tensor, gather_dim=0):
    """
    All-gather using symmetric memory P2P reads.

    Args:
        handle: SymmetricMemory handle from rendezvous
        tensor: Input tensor (local shard)
        gather_dim: Dimension to gather along

    Returns:
        paddle.Tensor: Gathered result (world_size * original along gather_dim)
    """
    rank = handle.rank
    world_size = handle.world_size
    shape = list(tensor.shape)
    dtype = tensor.dtype
    numel = 1
    for s in shape:
        numel *= s

    handle.barrier()

    gathered = []
    for peer in range(world_size):
        buf = handle.get_buffer(peer, [numel], dtype)
        gathered.append(buf.reshape(shape).clone())

    result = paddle.concat(gathered, axis=gather_dim)
    handle.barrier()
    return result


# --- Fallback SymmetricMemory handle for test mode ---

class _FallbackSymmetricMemory:
    """Fallback handle when real symmetric memory is not available."""

    def __init__(self, tensor):
        self._tensor = tensor
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def buffer_size(self):
        numel = self._tensor.numel()
        if hasattr(numel, 'item'):
            numel = numel.item()
        return int(numel) * self._tensor.element_size()

    @property
    def signal_pad_size(self):
        return get_signal_pad_size()

    def get_buffer(self, rank, sizes, dtype, storage_offset=0):
        return self._tensor.reshape(sizes).cast(dtype)

    def get_signal_pad(self, rank, sizes=None, dtype=None, storage_offset=0):
        if dtype is None:
            dtype = paddle.int32  # Paddle doesn't support uint32 directly
        numel = self.signal_pad_size // 4
        if sizes:
            numel = 1
            for s in sizes:
                numel *= s
        return paddle.zeros([numel], dtype="int32")

    def barrier(self, channel=0, timeout_ms=0):
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass  # In fallback mode, barrier is best-effort

    def put_signal(self, dst_rank, channel=0, timeout_ms=0):
        pass

    def wait_signal(self, src_rank, channel=0, timeout_ms=0):
        pass


# --- Utility ---

def _dtype_to_bytes(dtype) -> int:
    """Get byte size of a paddle dtype."""
    dtype_map = {
        paddle.float16: 2,
        paddle.bfloat16: 2,
        paddle.float32: 4,
        paddle.float64: 8,
        paddle.int8: 1,
        paddle.int16: 2,
        paddle.int32: 4,
        paddle.int64: 8,
        paddle.uint8: 1,
        paddle.float8_e4m3fn: 1,
        paddle.float8_e5m2: 1,
    }
    return dtype_map.get(dtype, 4)


# --- FP8 Fused Operations ---

def _fused_all_gather_scaled_matmul(
    A_shard,
    Bs,
    scales_Bs,
    gather_dim=0,
    group_name="default",
    biases=None,
    result_scales=None,
    out_dtypes=None,
):
    """
    Fused all_gather + scaled FP8 matmul.

    Gathers A_shard across ranks, then performs scaled matmul with each B.
    Supports FP8 inputs with per-tensor scale factors.

    Args:
        A_shard: Local shard of input matrix (FP8 or float)
        Bs: List of weight matrices (FP8)
        scales_Bs: List of scale factors for each B
        gather_dim: Dimension to gather along
        group_name: Process group name
        biases: Optional list of bias tensors
        result_scales: Optional list of output scale factors
        out_dtypes: Optional list of output dtypes

    Returns:
        (A_gathered, list_of_matmul_results)
    """
    group = dist.collective._get_default_group()
    world_size = dist.get_world_size()

    # All-gather A_shard (handle FP8 by gathering as uint8 since NCCL doesn't support FP8)
    original_dtype = A_shard.dtype
    if original_dtype in (paddle.float8_e4m3fn, paddle.float8_e5m2):
        original_shape = A_shard.shape
        A_shard_bytes = A_shard.view(paddle.uint8)
        gather_list = []
        for _ in range(world_size):
            gather_list.append(paddle.empty_like(A_shard_bytes))
        dist.all_gather(gather_list, A_shard_bytes, group=group)
        A_bytes = paddle.concat(gather_list, axis=gather_dim)
        A = A_bytes.view(original_dtype)
        # Restore shape: gathered along gather_dim
        gathered_shape = list(original_shape)
        gathered_shape[gather_dim] *= world_size
        A = A.reshape(gathered_shape)
    else:
        gather_list = []
        for _ in range(world_size):
            gather_list.append(paddle.empty_like(A_shard))
        dist.all_gather(gather_list, A_shard, group=group)
        A = paddle.concat(gather_list, axis=gather_dim)

    # Scaled matmul with each B
    mm_outputs = []
    for i, B in enumerate(Bs):
        scale_b = scales_Bs[i] if scales_Bs else 1.0
        bias = biases[i] if biases else None
        out_dtype = out_dtypes[i] if out_dtypes else 'bfloat16'
        result_scale = result_scales[i] if result_scales else 1.0

        if isinstance(scale_b, paddle.Tensor):
            scale_val = float(scale_b.item()) if scale_b.numel() == 1 else float(scale_b)
        else:
            scale_val = float(scale_b)

        if isinstance(result_scale, paddle.Tensor):
            result_scale_val = float(result_scale.item()) if result_scale.numel() == 1 else float(result_scale)
        else:
            result_scale_val = float(result_scale)

        combined_scale = scale_val * result_scale_val

        # Use FP8 fused gemm if both inputs are FP8
        if A.dtype in (paddle.float8_e4m3fn, paddle.float8_e5m2) and \
           B.dtype in (paddle.float8_e4m3fn, paddle.float8_e5m2):
            result = paddle.linalg.fp8_fp8_half_gemm_fused(
                A, B, False, False, bias, combined_scale, out_dtype, 'identity'
            )
        else:
            # Cast to float for matmul, then scale
            A_f = A.cast(paddle.float32) if A.dtype in (paddle.float8_e4m3fn, paddle.float8_e5m2) else A
            B_f = B.cast(paddle.float32) if B.dtype in (paddle.float8_e4m3fn, paddle.float8_e5m2) else B
            result = paddle.matmul(A_f, B_f) * combined_scale
            if bias is not None:
                result = result + bias
            if out_dtype == 'bfloat16':
                result = result.cast(paddle.bfloat16)
            elif out_dtype == 'float16':
                result = result.cast(paddle.float16)

        mm_outputs.append(result)

    return A, mm_outputs


def _fused_scaled_matmul_reduce_scatter(
    A,
    B,
    scale_B,
    reduce_op="sum",
    scatter_dim=0,
    group_name="default",
    bias=None,
    result_scale=None,
    out_dtype=None,
):
    """
    Fused scaled FP8 matmul + reduce_scatter.

    Performs scaled matmul(A, B), then reduce_scatters the result.
    Supports FP8 inputs with per-tensor scale factors.

    Args:
        A: Input matrix (FP8 or float)
        B: Weight matrix (FP8)
        scale_B: Scale factor for B
        reduce_op: Reduction operation ("sum" or "avg")
        scatter_dim: Dimension to scatter along
        group_name: Process group name
        bias: Optional bias tensor
        result_scale: Optional output scale factor
        out_dtype: Output dtype

    Returns:
        Reduced and scattered result tensor
    """
    group = dist.collective._get_default_group()
    world_size = dist.get_world_size()

    if out_dtype is None:
        out_dtype = 'bfloat16'

    if isinstance(scale_B, paddle.Tensor):
        scale_val = float(scale_B.item()) if scale_B.numel() == 1 else float(scale_B)
    else:
        scale_val = float(scale_B)

    if result_scale is not None:
        if isinstance(result_scale, paddle.Tensor):
            result_scale_val = float(result_scale.item()) if result_scale.numel() == 1 else float(result_scale)
        else:
            result_scale_val = float(result_scale)
    else:
        result_scale_val = 1.0

    combined_scale = scale_val * result_scale_val

    # Scaled matmul
    if A.dtype in (paddle.float8_e4m3fn, paddle.float8_e5m2) and \
       B.dtype in (paddle.float8_e4m3fn, paddle.float8_e5m2):
        C = paddle.linalg.fp8_fp8_half_gemm_fused(
            A, B, False, False, bias, combined_scale, out_dtype, 'identity'
        )
    else:
        A_f = A.cast(paddle.float32) if A.dtype in (paddle.float8_e4m3fn, paddle.float8_e5m2) else A
        B_f = B.cast(paddle.float32) if B.dtype in (paddle.float8_e4m3fn, paddle.float8_e5m2) else B
        C = paddle.matmul(A_f, B_f) * combined_scale
        if bias is not None:
            C = C + bias
        if out_dtype == 'bfloat16':
            C = C.cast(paddle.bfloat16)
        elif out_dtype == 'float16':
            C = C.cast(paddle.float16)

    # Reduce-scatter
    output_shape = list(C.shape)
    output_shape[scatter_dim] //= world_size
    output = paddle.empty(output_shape, dtype=C.dtype)

    chunks = paddle.split(C, world_size, axis=scatter_dim)
    dist.reduce_scatter(output, list(chunks), group=group)

    if reduce_op == "avg":
        output = output / world_size

    return output
