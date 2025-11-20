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

import types
from abc import ABC, abstractmethod

import paddle
import paddle.distributed as dist
from paddle.distributed.fleet.utils.log_util import logger

GROUPED_BATCH_SIZE = 2000


class AbstractCommunicator(ABC):
    @abstractmethod
    def communicate(self, comm_tasks, state, context):
        pass


def get_target_tensor(target_state_dict, read_item):
    use_dist = paddle.distributed.get_world_size() > 1
    if any(isinstance(k, tuple) for k in target_state_dict):
        key = (read_item.tensor_name, read_item.dst_global_offset)
    else:
        key = read_item.tensor_name

    tensor = target_state_dict[key]
    return tensor._local_value() if use_dist and tensor.is_dist() else tensor


def slice_tensor(tensor, slice_begin, slice_shape):
    if not slice_shape:
        assert not tensor.shape, (
            "Only 0-dimensional tensor supports empty slice_shape."
        )
        return tensor

    slice_end = [
        start + length for start, length in zip(slice_begin, slice_shape)
    ]
    axes = list(range(tensor.ndim))
    return paddle.slice(tensor, axes=axes, starts=slice_begin, ends=slice_end)


class SendRecvCommunicator(AbstractCommunicator):
    """
    Communicator that uses send/recv operations for data transfer.

    The process is broken down into batches to manage memory and communication overhead.
    """

    def communicate(self, comm_tasks, state, context):
        cur_rank = context['rank']
        process_group = context['process_group']
        use_group = context['use_group']

        source_state_dict = state['source_state_dict']
        target_state_dict = state['target_state_dict']

        all_received_slices = {}
        total_items = sum(len(items) for items in comm_tasks.values())
        processed_items = 0

        for batch_data in self._process_batches(
            comm_tasks, cur_rank, source_state_dict
        ):
            self._execute_p2p_ops(batch_data, cur_rank, use_group=use_group)

            for item, tensor in batch_data.source_slices.items():
                if item not in batch_data.local_copy_tasks:
                    tensor._clear()

            all_received_slices.update(batch_data.target_slices)

            processed_items += len(batch_data.read_items)
            progress = processed_items / total_items * 100
            logger.info(
                f"Batch communication completed. Progress: {processed_items}/{total_items} ({progress:.1f}%)."
            )

        self._assign_received_data(all_received_slices, target_state_dict)

        if use_group:
            paddle.distributed.barrier(process_group)
        logger.info("All communication tasks completed successfully.")

    def _process_batches(self, comm_tasks, cur_rank, source_state_dict):
        total_items = sum(len(items) for items in comm_tasks.values())
        item_count = 0

        batch_read_items = []
        batch_source_slices = {}
        batch_target_slices = {}
        batch_local_copy_tasks = set()

        for tensor_name, read_items in comm_tasks.items():
            tensors_to_clear = set()
            for item in read_items:
                item_count += 1
                batch_read_items.append(item)
                if cur_rank == item.src_rank:
                    src_tensor = source_state_dict[item.file_name][
                        item.tensor_name
                    ]
                    src_slice = (
                        slice_tensor(
                            src_tensor, item.src_local_offset, item.slice_shape
                        )
                        .cuda()
                        .clone()
                    )
                    batch_source_slices[item] = src_slice
                    tensors_to_clear.add(src_tensor)
                if cur_rank in item.dst_rank:
                    if cur_rank == item.src_rank:
                        batch_local_copy_tasks.add(item)
                        batch_target_slices[item] = batch_source_slices[item]
                    else:
                        dst_slice = paddle.zeros(
                            item.slice_shape, dtype=item.dtype
                        )
                        batch_target_slices[item] = dst_slice

                if (len(batch_read_items) % GROUPED_BATCH_SIZE == 0) or (
                    item_count == total_items
                ):
                    batch_data = types.SimpleNamespace(
                        read_items=batch_read_items,
                        source_slices=batch_source_slices,
                        target_slices=batch_target_slices,
                        local_copy_tasks=batch_local_copy_tasks,
                    )
                    yield batch_data
                    batch_read_items = []
                    batch_source_slices = {}
                    batch_target_slices = {}
                    batch_local_copy_tasks = set()

            for tensor in tensors_to_clear:
                tensor._clear_to_zero_allocation()

    def _execute_p2p_ops(self, batch_data, cur_rank, use_group):
        p2p_ops = []
        for item in batch_data.read_items:
            if item.src_rank == cur_rank:
                for rank in item.dst_rank:
                    if rank != cur_rank:
                        send_tensor = batch_data.source_slices[item]
                        if use_group:
                            p2p_ops.append(
                                dist.P2POp(dist.isend, send_tensor, rank)
                            )
                        else:
                            dist.send(send_tensor, rank)

            if cur_rank in item.dst_rank and item.src_rank != cur_rank:
                recv_tensor = batch_data.target_slices[item]
                if use_group:
                    p2p_ops.append(
                        dist.P2POp(dist.irecv, recv_tensor, item.src_rank)
                    )
                else:
                    dist.recv(recv_tensor, item.src_rank)

        if use_group and p2p_ops:
            logger.info(
                f"Starting batched send/recv for {len(p2p_ops)} P2P operations."
            )
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()
            logger.info("Batched send/recv finished.")

    def _assign_received_data(self, all_received_slices, target_state_dict):
        for item, received_slice in all_received_slices.items():
            dest_tensor = get_target_tensor(target_state_dict, item)
            if not dest_tensor._is_initialized():
                buffer = paddle.zeros_like(dest_tensor)
                buffer._share_buffer_to(dest_tensor)

            dest_slice = slice_tensor(
                dest_tensor, item.dst_local_offset, item.slice_shape
            )

            if dest_slice.place != received_slice.place:
                received_slice = received_slice.to(dest_slice.place)

            paddle.assign(received_slice, dest_slice)
