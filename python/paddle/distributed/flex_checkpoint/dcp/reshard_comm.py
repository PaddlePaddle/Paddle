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
from collections import defaultdict
from dataclasses import replace

import paddle
import paddle.distributed as dist
from paddle.distributed.collective import Group
from paddle.distributed.fleet.utils.log_util import logger

from .resharder import ReadItem
from .utils import (
    get_target_tensor,
    slice_tensor,
)

GROUPED_BATCH_SIZE = 10


class CommunicatorFactory:
    registry = {}

    @classmethod
    def register(cls, method, creator):
        cls.registry[method] = creator

    @classmethod
    def create(cls, comm_method, **kwargs):
        if comm_method not in cls.registry:
            raise ValueError(
                f"Unknown communication method '{comm_method}'. "
                f"Available: {list(cls.registry.keys())}"
            )
        return cls.registry[comm_method](**kwargs)


class AbstractCommunicator(ABC):
    @staticmethod
    def schedule_read_items(
        read_items: list[ReadItem],
    ) -> dict[str, list[ReadItem]]:
        order_rules = lambda read_item: (
            read_item.tensor_name,
            read_item.src_rank,
            read_item.src_global_offset,
            read_item.dst_rank,
            read_item.dst_local_offset,
            read_item.dst_global_offset
            if read_item.dst_global_offset is not None
            else (),
            read_item.src_local_offset,
            read_item.slice_shape,
            read_item.file_name,
            read_item.dtype,
        )
        # Step 1: Group by tensor_name
        tensor_groups = defaultdict(list)
        for item in read_items:
            tensor_groups[item.tensor_name].append(item)

        scheduled_items = defaultdict(list)

        # Step 2: For each tensor_name group, further group by all attributes except dst_rank
        for tensor_name, items in tensor_groups.items():
            grouped_items = defaultdict(list)
            for item in items:
                key = (
                    item.src_global_offset,
                    item.dst_global_offset,
                    item.src_rank,
                    item.dst_local_offset,
                    item.src_local_offset,
                    item.slice_shape,
                    item.file_name,
                    item.dtype,
                )
                grouped_items[key].append(item)

            # Step 3: Combine items with the same key into a single ReadItem with all dst_ranks
            for key, grouped_item in grouped_items.items():
                combined_dst_rank = []
                for item in grouped_item:
                    combined_dst_rank.extend(item.dst_rank)
                combined_dst_rank = sorted(
                    set(combined_dst_rank)
                )  # Remove duplicates

                # Create a new ReadItem with combined dst_ranks
                scheduled_item = ReadItem(
                    tensor_name=tensor_name,
                    src_global_offset=key[0],
                    dst_global_offset=key[1],
                    dst_rank=tuple(combined_dst_rank),
                    src_rank=key[2],
                    dst_local_offset=key[3],
                    src_local_offset=key[4],
                    slice_shape=key[5],
                    file_name=key[6],
                    dtype=key[7],
                )
                scheduled_items[tensor_name].append(scheduled_item)
        for key, items in scheduled_items.items():
            scheduled_items[key] = sorted(items, key=order_rules)

        return dict(sorted(scheduled_items.items()))

    @staticmethod
    def split_read_items(
        read_items: list[ReadItem],
    ) -> (list[ReadItem], list[ReadItem]):
        local_read_items = []
        comm_read_items = []

        for item in read_items:
            assert len(item.dst_rank) == 1, (
                "Before read_items is split, each ReadItem describes a communication task between one rank and another."
            )
            if item.src_rank == item.dst_rank[0]:
                local_read_items.append(item)
            else:
                comm_read_items.append(item)

        return local_read_items, comm_read_items

    @staticmethod
    def process_local_copy_tasks(
        local_tasks, cur_rank, source_state_dict, target_state_dict
    ):
        """
        Complete local copy tasks.
        """
        logger.debug(
            f"Rank {cur_rank} starting local copy for {len(local_tasks)} tasks."
        )
        for task in local_tasks:
            if task.src_rank != cur_rank:
                continue

            src_tensor = source_state_dict[task.file_name][task.tensor_name]
            dst_tensor = get_target_tensor(target_state_dict, task)
            src_chunk_tensor = slice_tensor(
                src_tensor, task.src_local_offset, task.slice_shape
            )

            dst_chunk_tensor = slice_tensor(
                dst_tensor, task.dst_local_offset, task.slice_shape
            )
            if src_chunk_tensor.place == dst_chunk_tensor.place:
                paddle.assign(src_chunk_tensor, dst_chunk_tensor)
                logger.debug(f"Local copy (same device) for task {task}.")
            else:
                tmp = (
                    src_chunk_tensor.cuda()
                    if dst_chunk_tensor.place.is_gpu_place()
                    else src_chunk_tensor.cpu()
                )
                paddle.assign(tmp, dst_chunk_tensor)
                del tmp
                logger.debug(f"Local copy (cross device) for task {task}.")

    @abstractmethod
    def communicate(self, read_items, state, context):
        pass


class BroadcastCommunicator(AbstractCommunicator):
    """
    Communicator that uses broadcast operation for data transfer.
    """

    def communicate(self, read_items, state, context):
        cur_rank = context['rank']
        process_group = context['process_group']

        source_state_dict = state['source_state_dict']
        target_state_dict = state['target_state_dict']

        local_read_items, comm_read_items = (
            BroadcastCommunicator.split_read_items(read_items)
        )

        logger.info(f"Generated {len(comm_read_items)} communication tasks.")
        logger.info(f"Generated {len(local_read_items)} local tasks.")

        BroadcastCommunicator.process_local_copy_tasks(
            local_read_items,
            cur_rank,
            source_state_dict,
            target_state_dict,
        )

        logger.info(
            f"Rank {cur_rank} finished local copy and entered communication phase."
        )

        comm_tasks = BroadcastCommunicator.schedule_read_items(comm_read_items)
        cnt = 0
        total_task_len = len(comm_tasks)
        for tensor_name, read_items in comm_tasks.items():
            cnt += 1
            if cnt % 500 == 0 or cnt == total_task_len:
                logger.info(
                    f"{cnt}/{total_task_len} tasks have been sent/received successfully!"
                )

            source_tensors = {}
            destination_tensors = {}
            for item in read_items:
                logger.debug(f"Beginning to send/recv task {item}.")
                if item.src_rank == cur_rank:
                    src_tensor = source_state_dict[item.file_name][
                        item.tensor_name
                    ]
                    if not src_tensor.place.is_gpu_place():
                        src_tensor = src_tensor.cuda()
                    source_tensors[(tensor_name, item.file_name)] = src_tensor
                elif cur_rank in item.dst_rank:
                    dst_tensor = get_target_tensor(target_state_dict, item)
                    if not dst_tensor.place.is_gpu_place():
                        gpu_dst_tensor = dst_tensor.cuda()
                        gpu_dst_tensor.need_cross_device_copy = True
                        gpu_dst_tensor.target_tensor = dst_tensor
                        destination_tensors[
                            (tensor_name, cur_rank, item.dst_global_offset)
                        ] = gpu_dst_tensor
                    else:
                        gpu_dst_tensor = dst_tensor
                        gpu_dst_tensor.target_tensor = dst_tensor
                        destination_tensors[
                            (tensor_name, cur_rank, item.dst_global_offset)
                        ] = dst_tensor

            for item in read_items:
                logger.debug(f"Beginning to send/recv task {item}.")
                if item.src_rank == cur_rank:
                    src_tensor = source_tensors[(tensor_name, item.file_name)]
                    src_chunk_tensor = slice_tensor(
                        src_tensor, item.src_local_offset, item.slice_shape
                    )
                    buffer_tensor = src_chunk_tensor.contiguous()
                elif cur_rank in item.dst_rank:
                    dst_tensor = destination_tensors[
                        (tensor_name, cur_rank, item.dst_global_offset)
                    ]
                    dst_chunk_tensor = slice_tensor(
                        dst_tensor, item.dst_local_offset, item.slice_shape
                    )
                    buffer_tensor = paddle.zeros_like(dst_chunk_tensor)
                    paddle.assign(dst_chunk_tensor, buffer_tensor)

                else:
                    buffer_tensor = paddle.zeros(item.slice_shape, item.dtype)
                paddle.distributed.broadcast(
                    buffer_tensor, src=item.src_rank, group=process_group
                )
                if cur_rank in item.dst_rank:
                    paddle.assign(buffer_tensor, dst_chunk_tensor)
                del buffer_tensor

            for dst_tensor in destination_tensors.values():
                if getattr(dst_tensor, 'need_cross_device_copy', False):
                    target_tensor = dst_tensor.target_tensor
                    delattr(dst_tensor, "target_tensor")
                    target_tensor.copy_(dst_tensor)
                else:
                    target_tensor = dst_tensor.target_tensor
                    delattr(dst_tensor, "target_tensor")
                    paddle.assign(dst_tensor, target_tensor)
                del dst_tensor

            del source_tensors
            paddle.distributed.barrier(process_group)

        logger.info("All communication tasks completed.")


class MultiGroupBroadcastCommunicator(AbstractCommunicator):
    """
    Communicator that uses broadcast for data transfer across multiple communication groups.
    """

    def __init__(self, worker_groups):
        if worker_groups is None:
            raise ValueError(
                "worker_groups must be specified when using multi_group_broadcast."
            )
        self.worker_groups = worker_groups

    @staticmethod
    def schedule_read_items(
        comm_read_items: list[ReadItem],
        worker_groups: list[Group],
    ) -> list[list[ReadItem]]:
        group_members = {}
        name_to_groups = {}
        read_items = []

        order_rules = lambda read_item: (
            read_item.tensor_name,
            read_item.src_rank,
            read_item.src_global_offset,
            read_item.dst_rank,
            read_item.dst_local_offset,
            read_item.dst_global_offset
            if read_item.dst_global_offset is not None
            else (),
            read_item.src_local_offset,
            read_item.slice_shape,
            read_item.file_name,
            read_item.dtype,
        )

        def _find_min_group(need_ranks, group_members, name_to_groups):
            min_group = None
            min_size = None
            for name, ranks in group_members.items():
                if need_ranks <= ranks:
                    if (min_size is None) or (len(ranks) < min_size):
                        min_size = len(ranks)
                        min_group = name_to_groups[name]
            assert min_group is not None, f"No group found for {need_ranks}!"
            return min_group

        for group in worker_groups:
            if len(group.ranks) <= 1:
                continue
            group_members[group.name] = set(group.ranks)
            name_to_groups[group.name] = group

        for read_item in comm_read_items:
            need_ranks = need_ranks = {*read_item.dst_rank, read_item.src_rank}
            group = _find_min_group(
                need_ranks,
                group_members,
                name_to_groups,
            )
            read_items.append(replace(read_item, comm_group=group))

        read_items = sorted(read_items, key=order_rules)

        def _build_group_conflict(group_members: dict[str, set]):
            member_to_groups = defaultdict(set)
            for g, members in group_members.items():
                for m in members:
                    member_to_groups[m].add(g)
            group_conflict = defaultdict(set)
            for group_set in member_to_groups.values():
                for g1 in group_set:
                    for g2 in group_set:
                        if g1 != g2:
                            group_conflict[g1].add(g2)
            return group_conflict

        def _dsatur_coloring(group_conflict: dict[str, set]) -> dict[str, int]:
            import heapq

            all_groups = sorted(group_conflict.keys())
            sorted_conflict = {g: sorted(group_conflict[g]) for g in all_groups}

            color_map = {}
            neighbor_colors = {g: set() for g in all_groups}
            uncolored = set(all_groups)

            degree = {g: len(sorted_conflict[g]) for g in all_groups}

            heap = []
            for g in all_groups:
                heapq.heappush(heap, (0, -degree[g], g))
            saturation = dict.fromkeys(all_groups, 0)

            while uncolored:
                while True:
                    _, _, node = heapq.heappop(heap)
                    if node in uncolored:
                        break
                used = neighbor_colors[node]
                color = 0
                while color in used:
                    color += 1
                color_map[node] = color
                uncolored.remove(node)
                for neighbor in sorted_conflict[node]:
                    if neighbor in uncolored:
                        if color not in neighbor_colors[neighbor]:
                            neighbor_colors[neighbor].add(color)
                            saturation[neighbor] += 1
                            heapq.heappush(
                                heap,
                                (
                                    -saturation[neighbor],
                                    -degree[neighbor],
                                    neighbor,
                                ),
                            )
            return color_map

        def _assign_batches(tasks, group_color_map):
            batches = defaultdict(list)
            for t in tasks:
                g = t.comm_group.name
                batches[group_color_map[g]].append(t)
            return [
                sorted(batches[c], key=order_rules) for c in sorted(batches)
            ]

        group_conflict = _build_group_conflict(group_members)
        group_color_map = _dsatur_coloring(group_conflict)
        results = _assign_batches(read_items, group_color_map)
        return results

    def communicate(self, read_items, state, context):
        cur_rank = context['rank']
        process_group = context['process_group']
        worker_groups = self.worker_groups

        source_state_dict = state['source_state_dict']
        target_state_dict = state['target_state_dict']

        local_read_items, comm_read_items = (
            MultiGroupBroadcastCommunicator.split_read_items(read_items)
        )

        logger.info(f"Generated {len(comm_read_items)} communication tasks.")
        logger.info(f"Generated {len(local_read_items)} local tasks.")

        MultiGroupBroadcastCommunicator.process_local_copy_tasks(
            local_read_items,
            cur_rank,
            source_state_dict,
            target_state_dict,
        )
        results = MultiGroupBroadcastCommunicator.schedule_read_items(
            comm_read_items, worker_groups
        )
        logger.info(
            f"Communication task scheduling completed, {len(results)}  batches in total."
        )
        for read_items in results:
            source_tensors = {}
            destination_tensors = {}
            for item in read_items:
                tensor_name = item.tensor_name
                if item.src_rank == cur_rank:
                    src_tensor = source_state_dict[item.file_name][tensor_name]
                    if not src_tensor.place.is_gpu_place():
                        src_tensor = src_tensor.cuda()
                    source_tensors[(tensor_name, item.file_name)] = src_tensor
                elif cur_rank in item.dst_rank:
                    dst_tensor = get_target_tensor(target_state_dict, item)
                    if not dst_tensor.place.is_gpu_place():
                        gpu_dst_tensor = dst_tensor.cuda()
                        gpu_dst_tensor.need_cross_device_copy = True
                        gpu_dst_tensor.target_tensor = dst_tensor
                        destination_tensors[
                            (tensor_name, cur_rank, item.dst_global_offset)
                        ] = gpu_dst_tensor
                    else:
                        gpu_dst_tensor = dst_tensor
                        gpu_dst_tensor.target_tensor = dst_tensor
                        destination_tensors[
                            (tensor_name, cur_rank, item.dst_global_offset)
                        ] = dst_tensor

            for item in read_items:
                logger.debug(f"Beginning to send/recv task {item}.")
                tensor_name = item.tensor_name
                if item.src_rank == cur_rank:
                    src_tensor = source_tensors[(tensor_name, item.file_name)]
                    src_chunk_tensor = slice_tensor(
                        src_tensor, item.src_local_offset, item.slice_shape
                    )
                    buffer_tensor = src_chunk_tensor.contiguous()
                elif cur_rank in item.dst_rank:
                    dst_tensor = destination_tensors[
                        (tensor_name, cur_rank, item.dst_global_offset)
                    ]
                    dst_chunk_tensor = slice_tensor(
                        dst_tensor, item.dst_local_offset, item.slice_shape
                    )
                    buffer_tensor = paddle.zeros_like(dst_chunk_tensor)
                    paddle.assign(dst_chunk_tensor, buffer_tensor)

                elif cur_rank in item.comm_group.ranks:
                    buffer_tensor = paddle.zeros(item.slice_shape, item.dtype)
                else:
                    buffer_tensor = None

                if cur_rank in item.comm_group.ranks:
                    paddle.distributed.broadcast(
                        buffer_tensor, src=item.src_rank, group=item.comm_group
                    )

                if cur_rank in item.dst_rank:
                    paddle.assign(buffer_tensor, dst_chunk_tensor)
                del buffer_tensor

            for dst_tensor in destination_tensors.values():
                if getattr(dst_tensor, 'need_cross_device_copy', False):
                    target_tensor = dst_tensor.target_tensor
                    delattr(dst_tensor, "target_tensor")
                    target_tensor.copy_(dst_tensor)
                else:
                    target_tensor = dst_tensor.target_tensor
                    delattr(dst_tensor, "target_tensor")
                    paddle.assign(dst_tensor, target_tensor)
                del dst_tensor

            del source_tensors

        paddle.distributed.barrier(process_group)
        logger.info("All communication tasks completed.")


class SendRecvCommunicator(AbstractCommunicator):
    """
    Communicator that uses send/recv operations for data transfer.

    The process is broken down into batches to manage memory and communication overhead.
    """

    def __init__(self, use_group):
        self.use_group = use_group

    @staticmethod
    def schedule_read_items(
        read_items: list[ReadItem],
    ) -> dict[str, list[ReadItem]]:
        order_rules = lambda read_item: (
            read_item.tensor_name,
            read_item.src_rank,
            read_item.src_global_offset,
            read_item.dst_rank,
            read_item.dst_local_offset,
            read_item.dst_global_offset
            if read_item.dst_global_offset is not None
            else (),
            read_item.src_local_offset,
            read_item.slice_shape,
            read_item.file_name,
            read_item.dtype,
        )

        tensor_groups = defaultdict(list)
        for item in read_items:
            tensor_groups[item.tensor_name].append(item)

        return dict(sorted(tensor_groups.items()))

    def communicate(self, read_items, state, context):
        comm_tasks = SendRecvCommunicator.schedule_read_items(read_items)
        cur_rank = context['rank']
        process_group = context['process_group']

        source_state_dict = state['source_state_dict']
        target_state_dict = state['target_state_dict']

        total_items = sum(len(items) for items in comm_tasks.values())
        processed_items = 0

        for batch_data in self._process_batches(
            comm_tasks, cur_rank, source_state_dict
        ):
            received_slices = {}
            self._execute_p2p_ops(
                batch_data, cur_rank, use_group=self.use_group
            )

            for item, tensor in batch_data.source_slices.items():
                if item not in batch_data.local_copy_tasks:
                    tensor._clear()

            received_slices.update(batch_data.target_slices)

            processed_items += len(batch_data.read_items)
            progress = processed_items / total_items * 100
            logger.info(
                f"Batch communication completed. Progress: {processed_items}/{total_items} ({progress:.1f}%)."
            )

            self._assign_received_data(received_slices, target_state_dict)

            for received_slice in received_slices.values():
                received_slice._clear()

            del received_slices

        if self.use_group:
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

                if ((item_count % GROUPED_BATCH_SIZE) == 0) or (
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

    def _assign_received_data(self, received_slices, target_state_dict):
        for item, received_slice in received_slices.items():
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


CommunicatorFactory.register(
    "multi_group_broadcast",
    lambda worker_groups: MultiGroupBroadcastCommunicator(worker_groups),
)
CommunicatorFactory.register(
    "send_recv", lambda **kwargs: SendRecvCommunicator(use_group=False)
)
CommunicatorFactory.register(
    "grouped_send_recv", lambda **kwargs: SendRecvCommunicator(use_group=True)
)
CommunicatorFactory.register(
    "broadcast", lambda **kwargs: BroadcastCommunicator()
)
