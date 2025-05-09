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

from __future__ import annotations

import queue

import paddle
from paddle.incubate.tensor.manipulation import (
    async_offload_with_offset,
    async_reload,
    create_async_load,
)


class RROOBuffer:
    """A cpu buffer for asynchronous offloading and reloading of tensors."""

    def __init__(
        self, numel: int, dtype: str, buffer_id: int, pool_key: tuple[int, str]
    ) -> None:
        self.data = paddle.empty([numel], dtype=dtype).pin_memory()
        self.buffer_id = buffer_id
        self.pool_key = pool_key
        self.is_used = False

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RROOBuffer):
            return (self.buffer_id == other.buffer_id) and (
                self.pool_key == other.pool_key
            )
        return False

    def __hash__(self) -> int:
        return hash((self.buffer_id, self.pool_key))


class RROOBufferPool:
    """A pool of RROOBuffers for memory management."""

    def __init__(
        self,
        numel: int,
        dtype: str,
        pool_key: tuple[int, str],
        initial_pool_size: int = 5,
    ) -> None:
        self.numel = numel
        self.dtype = dtype
        self.pool_key = pool_key

        self.all_buffers = {
            RROOBuffer(numel, dtype, buffer_id, pool_key)
            for buffer_id in range(initial_pool_size)
        }
        self.available_buffers = set(self.all_buffers)

    def get_buffer(self) -> RROOBuffer:
        """Get an available buffer from the pool, creating a new one if none are available."""
        if 0 == len(self.available_buffers):
            new_buffer = RROOBuffer(
                self.numel, self.dtype, len(self.all_buffers), self.pool_key
            )
            self.all_buffers.add(new_buffer)
            self.available_buffers.add(new_buffer)
        buffer = self.available_buffers.pop()
        buffer.is_used = True
        return buffer

    def release_buffer(self, buffer: RROOBuffer) -> None:
        """Release a buffer back to the pool."""
        assert isinstance(
            buffer, RROOBuffer
        ), f"Expect a RROOBuffer, but got {type(buffer)}."
        assert buffer in self.all_buffers, "Buffer is not in this Pool."
        self.available_buffers.add(buffer)
        buffer.is_used = False

    def is_all_memory_free(self) -> bool:
        """Check if all buffers in the pool are available."""
        return len(self.available_buffers) == len(self.all_buffers)


class RROOBufferPoolManager:
    """Manager for multiple RROOBufferPools."""

    def __init__(self) -> None:
        self.pools: dict[tuple[int, str], RROOBufferPool] = {}

    def get_buffer(self, numel: int, dtype: str) -> RROOBuffer:
        """Get a buffer from the appropriate pool based on size and type."""
        pool_key = (numel, dtype)
        if pool_key not in self.pools:
            self.pools[pool_key] = RROOBufferPool(numel, dtype, pool_key)
        return self.pools[pool_key].get_buffer()

    def release_buffer(self, buffer: RROOBuffer) -> None:
        """Release a buffer back to its pool."""
        self.pools[buffer.pool_key].release_buffer(buffer)

    def is_all_memory_free(self) -> bool:
        """Check if all buffers in all pools are available."""
        for _, pool in self.pools.items():
            if not pool.is_all_memory_free():
                return False
        return True


class RROOTensorWrapper:
    """Wrapper for a tensor that handles offloading and reloading operations."""

    def __init__(
        self,
        cuda_data: paddle.Tensor,
        cpu_buffer: RROOBuffer | None = None,
        split_factor: int | None = None,
    ) -> None:
        self.cuda_data = cuda_data
        self.cpu_buffer = cpu_buffer
        self.shape = self.cuda_data.shape

        self.split_id = 0  # Indicates which split is next to process
        self.split_factor = split_factor  # Total number of splits
        self.loader = create_async_load()

    def get_cuda_datas_to_release(self) -> paddle.Tensor:
        """Get the CUDA tensor that can be released after offloading."""
        assert self.split_factor is not None
        return self.cuda_data

    def pop_cpu_buffers_to_release(self) -> RROOBuffer:
        """Get and clear the CPU buffer that can be released after reloading."""
        assert self.split_factor is not None
        out = self.cpu_buffer
        self.cpu_buffer = None
        return out

    def offload(self) -> tuple[paddle.Tensor, bool]:
        """Offload a portion of the tensor to CPU memory."""
        assert self.split_factor is not None
        assert self.cpu_buffer is not None and self.cpu_buffer.is_used

        numel = nonblocking_numel(self.cuda_data)
        assert numel == nonblocking_numel(self.cpu_buffer)

        # Calculate base chunk size and remainder elements that need distribution
        base_size = numel // self.split_factor
        remainder = numel % self.split_factor

        # Calculate actual chunk size for current split (earlier splits handle one extra element)
        if self.split_id < remainder:
            current_size = base_size + 1
            offset = (base_size + 1) * self.split_id
        else:
            current_size = base_size
            offset = (base_size + 1) * remainder + base_size * (
                self.split_id - remainder
            )

        paddle.framework.core.nvprof_nvtx_push("rroo offload")
        task = async_offload_with_offset(
            src_tensor=self.cuda_data.flatten(),
            dst_tensor=self.cpu_buffer.data,
            src_offset=offset,
            dst_offset=offset,
            offload_size=current_size,
            async_loader=self.loader,
        )
        paddle.framework.core.nvprof_nvtx_pop()
        self.split_id += 1
        offload_completed = self.split_id >= self.split_factor
        return task, offload_completed

    def reload(self) -> paddle.Tensor:
        """Reload the tensor from CPU memory to GPU memory."""
        assert self.split_factor is not None
        assert self.cpu_buffer is not None and self.cpu_buffer.is_used

        cuda_data, task = async_reload(self.cpu_buffer.data, self.loader)
        self.cuda_data = cuda_data.reshape(self.shape)
        return task


class RROOOffloadQueue:
    """Manager for offloading transactions."""

    def __init__(self) -> None:
        self.offload_transaction_queue: list[RROOTensorWrapper] = []
        self.task_list: list[paddle.Tensor] = []
        self.cuda_tensor_to_release: list[paddle.Tensor] = []

    def put(self, rroo_transaction: RROOTensorWrapper) -> None:
        self.offload_transaction_queue.append(rroo_transaction)

    def wait(self) -> None:
        """Wait for all offloading tasks to complete."""
        for task in self.task_list:
            task.cpu_wait()
        self.task_list = []

    def get_cuda_datas_to_release(self) -> list[paddle.Tensor]:
        """Get and clear the list of CUDA tensors that can be released."""
        out = self.cuda_tensor_to_release
        self.cuda_tensor_to_release = []
        return out

    def offload(self) -> RROOTensorWrapper | None:
        """Perform the next offload operation."""
        if 0 == len(self.offload_transaction_queue):
            return None

        task, offload_completed = self.offload_transaction_queue[0].offload()
        self.task_list.append(task)
        if offload_completed:
            tensor_wrapper = self.offload_transaction_queue.pop(0)
            self.cuda_tensor_to_release.append(
                tensor_wrapper.get_cuda_datas_to_release()
            )
            return tensor_wrapper
        else:
            return None

    def empty(self) -> bool:
        """Check if all operations are complete."""
        if len(self.offload_transaction_queue) > 0:
            return False
        if len(self.task_list) > 0:
            return False
        if len(self.cuda_tensor_to_release) > 0:
            return False
        return True


class RROOReloadQueue:
    """Manager for reloading transactions."""

    def __init__(self) -> None:
        self.reload_transaction_queue: list[RROOTensorWrapper] = []
        self.task_list: list[paddle.Tensor] = []
        self.cpu_buffer_to_release: list[RROOBuffer] = []

    def put(self, rroo_transaction: RROOTensorWrapper) -> None:
        self.reload_transaction_queue.append(rroo_transaction)

    def wait(self) -> None:
        """Wait for all reloading tasks to complete."""
        for task in self.task_list:
            task.cpu_wait()
        self.task_list = []

    def pop_cpu_buffers_to_release(self) -> list[RROOBuffer]:
        """Get and clear the list of CPU buffers that can be released."""
        out = self.cpu_buffer_to_release
        self.cpu_buffer_to_release = []
        return out

    def reload(self) -> None:
        """Perform the next reload operation."""
        if 0 == len(self.reload_transaction_queue):
            return

        tensor_wrapper = self.reload_transaction_queue.pop(0)

        self.task_list.append(tensor_wrapper.reload())
        self.cpu_buffer_to_release.append(
            tensor_wrapper.pop_cpu_buffers_to_release()
        )

    def empty(self) -> bool:
        """Check if all operations are complete."""
        if len(self.reload_transaction_queue) > 0:
            return False
        if len(self.task_list) > 0:
            return False
        if len(self.cpu_buffer_to_release) > 0:
            return False
        return True


class RROOQueue:
    """Queue for managing tensor offloading and reloading operations."""

    def __init__(
        self, acc_num: int, split_factor: int = 1, do_rroo: bool = False
    ) -> None:
        self.activations_queue = queue.Queue()

        self.offload_transaction_manager = RROOOffloadQueue()
        self.reload_transaction_manager = RROOReloadQueue()

        self.acc_num = acc_num
        self.acc_id = 0
        self.split_factor = split_factor
        self.do_rroo = (
            do_rroo  # For all acc in a chunk, either all do RROO or none do
        )

    def empty(self) -> bool:
        """Check if the queue is empty and all operations are complete."""
        if self.activations_queue.qsize() > 0:
            return False
        if not self.offload_transaction_manager.empty():
            return False
        if not self.reload_transaction_manager.empty():
            return False
        if not 0 == self.acc_id:
            return False
        return True

    def put(self, cuda_data: paddle.Tensor) -> None:
        """Put a tensor into the queue, routing to appropriate method based on RROO flag."""
        put_method = (
            self.rroo_put
            if self.do_rroo and self.can_offload_current_acc()
            else self.simple_put
        )
        put_method(cuda_data)
        self.acc_id = (self.acc_id + 1) % self.acc_num

    def simple_put(self, cuda_data: paddle.Tensor) -> None:
        """Simple put without offloading."""
        self.activations_queue.put(RROOTensorWrapper(cuda_data))

    def rroo_put(self, cuda_data: paddle.Tensor) -> None:
        """Put with offloading using RROO (Round-Robin Offloading)."""
        cpu_buffer = get_rroo_buffer_pool_manager().get_buffer(
            nonblocking_numel(cuda_data), cuda_data.dtype
        )
        tensor_wrapper = RROOTensorWrapper(
            cuda_data, cpu_buffer, self.split_factor
        )
        self.offload_transaction_manager.put(tensor_wrapper)
        self.activations_queue.put(tensor_wrapper)

    def get(self) -> paddle.Tensor:
        """Get the next tensor from the queue."""
        assert self.activations_queue.qsize() > 0
        tensor_wrapper = self.activations_queue.get()
        assert (
            tensor_wrapper.cpu_buffer is None
            or not tensor_wrapper.cpu_buffer.is_used
        ), "CPU buffer leakage occurred. Maybe someone was not offloaded."
        return tensor_wrapper.cuda_data

    def wait_and_release(self) -> None:
        """Wait for all operations to complete and release resources."""
        self.offload_transaction_manager.wait()
        self.reload_transaction_manager.wait()

        # Release resources by their original allocators
        for (
            cuda_data
        ) in self.offload_transaction_manager.get_cuda_datas_to_release():
            cuda_data._clear()
        for (
            cpu_buffer
        ) in self.reload_transaction_manager.pop_cpu_buffers_to_release():
            get_rroo_buffer_pool_manager().release_buffer(cpu_buffer)

    def offload(self) -> None:
        """Start asynchronous offloading at the beginning of each forward step."""
        reload_rroo_transaction = self.offload_transaction_manager.offload()
        if reload_rroo_transaction is not None:
            self.reload_transaction_manager.put(reload_rroo_transaction)

    def reload(self) -> None:
        """Start reloading at the beginning of each backward step."""
        self.reload_transaction_manager.reload()

    def can_offload_current_acc(self) -> bool:
        """Determine if the current accumulation can be offloaded.

        Returns:
            bool: True if the current accumulation can be offloaded, False otherwise.
            - For single-split case: Can't offload first or last accumulation
            - For multi-split case: Can't offload last accumulation
        """
        is_last_acc = self.acc_id == (self.acc_num - 1)
        is_first_acc = self.acc_id == 0

        if self.split_factor == 1:
            return not (is_first_acc or is_last_acc)
        else:
            return not is_last_acc


class RROOQueueManager:
    """Manager for queues that handle offloading and reloading operations."""

    def init(self, chunk_num: int, acc_num: int) -> None:
        self.queues: list[RROOQueue] = []
        self.offload_dict: list[list[RROOQueue]] = [
            [] for _ in range(chunk_num)
        ]
        self.reload_dict: list[list[RROOQueue]] = [[] for _ in range(chunk_num)]
        self.chunk_num = chunk_num
        self.cur_chunk_id = 0  # Follows the pipeline framework's design, setting VPP id as object state
        self.acc_num = acc_num

    def calc_rroo_infos(
        self, split_factor: int
    ) -> tuple[bool, list[int] | None, list[int] | None]:
        """
        Calculate RROO (Round-Robin Offloading) information.

        Returns:
            - bool: Whether RROO can be performed on current queue
            - list: Chunk IDs where offloading should occur
            - list: Chunk IDs where reloading should occur
        """
        if split_factor < 1:
            return False, None, None

        # Given total chunks, current chunk id and split factor, determine offload/reload chunks
        tgt_chunk_id = split_factor * self.cur_chunk_id
        if tgt_chunk_id + split_factor > self.chunk_num:
            return False, None, None
        offload_chunk_ids = list(
            range(tgt_chunk_id, tgt_chunk_id + split_factor)
        )
        reload_chunk_ids = [tgt_chunk_id]
        return True, offload_chunk_ids, reload_chunk_ids

    def set_cur_chunk_id(self, cur_chunk_id: int) -> None:
        self.cur_chunk_id = cur_chunk_id

    def create_rroo_queue(self, split_factor: int) -> RROOQueue:
        """Create a new RROOQueue with predetermined offload/reload chunks."""
        do_rroo, offload_chunk_ids, reload_chunk_ids = self.calc_rroo_infos(
            split_factor
        )

        rroo_queue = RROOQueue(self.acc_num, split_factor, do_rroo)

        self.queues.append(rroo_queue)
        if do_rroo:
            for offload_chunk_id in offload_chunk_ids:
                self.offload_dict[offload_chunk_id].append(rroo_queue)
            for reload_chunk_id in reload_chunk_ids:
                self.reload_dict[reload_chunk_id].append(rroo_queue)
        return rroo_queue

    def offload(self) -> None:
        """Perform offloading for current chunk."""
        for q in self.offload_dict[self.cur_chunk_id]:
            q.offload()

    def reload(self) -> None:
        """Perform reloading for current chunk."""
        for q in self.reload_dict[self.cur_chunk_id]:
            q.reload()

    def wait_and_release(self) -> None:
        """Wait for all operations to complete and release resources."""
        for q in self.queues:
            q.wait_and_release()

    def empty(self) -> bool:
        """Check if all queues are empty."""
        for q in self.queues:
            if not q.empty():
                return False
        return True


def nonblocking_numel(tensor: paddle.Tensor | RROOBuffer) -> int:
    """Calculate the total number of elements in a tensor or buffer without blocking."""
    assert isinstance(tensor, (paddle.Tensor, RROOBuffer))
    numel = 1
    for s in tensor.shape:
        numel *= s
    return numel


# Global manager instance
_rroo_buffer_pool_manager = None
_rroo_queue_manager = None


def get_rroo_buffer_pool_manager():
    global _rroo_buffer_pool_manager
    if _rroo_buffer_pool_manager is None:
        _rroo_buffer_pool_manager = RROOBufferPoolManager()
    return _rroo_buffer_pool_manager


def get_rroo_queue_manager():
    global _rroo_queue_manager
    if _rroo_queue_manager is None:
        _rroo_queue_manager = RROOQueueManager()
    return _rroo_queue_manager
