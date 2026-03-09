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

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.fleet.utils.log_util import logger

from .metadata import LocalTensorIndex, LocalTensorMetadata
from .sharded_weight import (
    ShardedWeight,
)
from .utils import (
    compute_local_shape_and_global_offset,
    get_target_tensor,
    slice_tensor,
)

if TYPE_CHECKING:
    from paddle.distributed.collective import Group

    from .reshard_comm import AbstractCommunicator

PATH_TO_CHECKPOINT_FILES: dict[str, tuple[list, list]] = {}


@dataclass(frozen=True)
class ReadItem:
    """
    A communication operation for a Tensor between ranks.

    Attributes:
        tensor_name (str): Name of the tensor.
        src_global_offset (tuple[int]): Global offset in the source tensor.
        dst_global_offset (tuple[int] | None): Global offset in the destination tensor.
        dst_rank (list[int]): Destination ranks.
        src_rank (int): Source rank.
        dst_local_offset (tuple[int]): Local offset in the destination tensor partition.
        src_local_offset (tuple[int]): Local offset in the source tensor partition.
        slice_shape (tuple[int]): Shape of the slice to transfer.
        file_name (str): The name of the file from which the source tensor is read on the source rank.
        dtype (str): Data type of the tensor.
    """

    tensor_name: str
    src_global_offset: tuple[int]
    dst_global_offset: tuple[int] | None
    dst_rank: tuple[int]
    src_rank: int
    dst_local_offset: tuple[int]
    src_local_offset: tuple[int]
    slice_shape: tuple[int]
    file_name: str
    dtype: str
    comm_group: Group | None = None


@dataclass(frozen=True)
class ExtendReadItem(ReadItem):
    global_shape: tuple[int] | None = None


class OperationType(Enum):
    GLOBAL_BROADCAST = auto()
    BROADCAST_ALLGATHER = auto()


class AllGatherType(Enum):
    WITH_PADDING = auto()
    NO_PADDING = auto()


INTERNAL_PADDING_TENSOR_NAME = "__internal_padding_tensor_name__"


def get_load_infos(metadata_list, local_load_files, process_group, use_dist):
    load_info = {}
    cur_rank = paddle.distributed.get_rank()
    for metadata in metadata_list:
        for local_tensor_index, file_name in metadata.storage_metadata.items():
            if file_name in local_load_files:
                load_info[local_tensor_index] = (
                    cur_rank,
                    file_name,
                )
    load_info_list = []
    if use_dist:
        paddle.distributed.all_gather_object(
            load_info_list, load_info, process_group
        )
    else:
        load_info_list.append(load_info)
    load_infos = {}
    for load_info in load_info_list:
        for local_tensor_index, (rank, file_name) in load_info.items():
            assert local_tensor_index not in load_infos
            load_infos[local_tensor_index] = (rank, file_name)
    return load_infos


def compute_overlap(
    cur_chunk_metadata: LocalTensorMetadata,
    storage_local_tensor_metadata: LocalTensorMetadata,
):
    cur_offsets = []
    storage_offsets = []
    lengths = []
    for cur_len, cur_offset, storage_len, storage_offset in zip(
        cur_chunk_metadata.local_shape,
        cur_chunk_metadata.global_offset,
        storage_local_tensor_metadata.local_shape,
        storage_local_tensor_metadata.global_offset,
    ):
        begin_offset = max(cur_offset, storage_offset)
        end_offset = min(cur_offset + cur_len, storage_offset + storage_len)
        if begin_offset == cur_offset:
            cur_offsets.append(0)
            storage_offsets.append(begin_offset - storage_offset)
        elif begin_offset == storage_offset:
            cur_offsets.append(begin_offset - cur_offset)
            storage_offsets.append(0)
        else:
            raise ValueError(
                f"Invalid begin_offset:{begin_offset}, cur_offset:{cur_offset}, storage_offset:{storage_offset}"
            )
        lengths.append(end_offset - begin_offset)
        assert lengths[-1] >= 0, (
            f"Invalid length:{lengths[-1]}, end_offset:{end_offset}, begin_offset:{begin_offset}"
        )
    return cur_offsets, storage_offsets, lengths


def not_overlap(
    cur_chunk_metadata: LocalTensorMetadata,
    storage_local_tensor_metadata: LocalTensorMetadata,
):
    for cur_len, cur_offset, storage_len, storage_offset in zip(
        cur_chunk_metadata.local_shape,
        cur_chunk_metadata.global_offset,
        storage_local_tensor_metadata.local_shape,
        storage_local_tensor_metadata.global_offset,
    ):
        if (
            cur_offset >= (storage_offset + storage_len)
            or (cur_offset + cur_len) <= storage_offset
        ):
            return True
    return False


def build_storage_state_dict_metadata(metadata_list):
    counts = {}
    for md in metadata_list:
        items = md.state_dict_metadata.items()
        for k, lst in items:
            counts[k] = counts.get(k, 0) + len(lst)

    result = {k: [None] * n for k, n in counts.items()}
    offset = dict.fromkeys(counts, 0)

    for md in metadata_list:
        items = md.state_dict_metadata.items()
        for k, lst in items:
            o = offset[k]
            n = len(lst)
            result[k][o : o + n] = lst
            offset[k] = o + n

    return result


def get_read_items(
    metadata_list, state_dict, process_group, use_dist, load_infos
):
    storage_state_dict_metadata = {}
    storage_state_dict_metadata = build_storage_state_dict_metadata(
        metadata_list
    )

    read_items = []
    global_shape = None
    for tensor_key, val in state_dict.items():
        tensor_name = None
        if isinstance(val, paddle.Tensor):
            if val.is_dist():
                # when val is scalar, the shape is []
                (
                    local_shape,
                    global_offset,
                ) = (
                    compute_local_shape_and_global_offset(
                        val.shape,
                        val.process_mesh,
                        val.placements,
                    )
                    if len(val.shape) > 0
                    else ((), ())
                )
                global_shape = tuple(val.shape)
                if local_shape is None or global_offset is None:
                    continue
            else:
                local_shape = tuple(val.shape)
                global_offset = (
                    tuple([0] * len(val.shape)) if len(val.shape) > 0 else ()
                )
                global_shape = local_shape
            dtype = str(val.dtype).split(".")[1]
            tensor_name = tensor_key
        elif isinstance(val, ShardedWeight):
            local_shape, global_offset = (
                (val.local_shape, val.global_offset)
                if len(val.global_shape) > 0
                else ((), ())
            )
            dtype = str(val.local_tensor.dtype).split(".")[1]
            tensor_name = (
                tensor_key[0] if isinstance(tensor_key, tuple) else tensor_key
            )
        else:
            raise ValueError(
                f"Only support paddle.Tensor., val type:{type(val)}"
            )

        cur_chunk_metadata = LocalTensorMetadata(
            global_offset, local_shape, dtype, global_shape
        )

        for storage_local_tensor_metadata in storage_state_dict_metadata[
            tensor_name
        ]:
            if not_overlap(cur_chunk_metadata, storage_local_tensor_metadata):
                continue
            cur_offsets, storage_offsets, lengths = compute_overlap(
                cur_chunk_metadata, storage_local_tensor_metadata
            )
            storage_local_tensor_index = LocalTensorIndex(
                tensor_name,
                tuple(storage_local_tensor_metadata.global_offset),
                local_shape=tuple(storage_local_tensor_metadata.local_shape),
            )
            src_rank, file_name = load_infos[storage_local_tensor_index]
            read_items.append(
                ReadItem(
                    tensor_name=tensor_name,
                    src_global_offset=tuple(
                        storage_local_tensor_metadata.global_offset
                    ),
                    dst_global_offset=global_offset,
                    dst_rank=(paddle.distributed.get_rank(),),
                    src_rank=src_rank,
                    dst_local_offset=tuple(cur_offsets),
                    src_local_offset=tuple(storage_offsets),
                    slice_shape=tuple(lengths),
                    file_name=file_name,
                    dtype=storage_local_tensor_metadata.dtype,
                ),
            )

    global_read_items = []
    tmp = []
    if use_dist:
        paddle.distributed.all_gather_object(tmp, read_items, process_group)
    else:
        tmp.append(read_items)
    for items in tmp:
        for item in items:
            global_read_items.append(item)
    return global_read_items


class StateDictResharder:
    def __init__(
        self,
        target_state_dict,
        source_state_dict,
        metadata_list,
        communicator: AbstractCommunicator,
        process_group=None,
        offload=False,
        use_dist=True,
    ):
        self.target_state_dict = target_state_dict
        self.source_state_dict = source_state_dict
        self.metadata_list = metadata_list
        self.communicator = communicator
        self.process_group = process_group
        self.offload = offload
        self.use_dist = use_dist

    def preprocess(self):
        if self.offload:
            for file_name, state_dict in self.source_state_dict.items():
                self.source_state_dict[file_name] = {
                    k: paddle.to_tensor(v, place=paddle.CPUPlace())
                    if isinstance(v, np.ndarray)
                    else v
                    for k, v in state_dict.items()
                }
        local_load_files = list(self.source_state_dict.keys())
        load_infos = get_load_infos(
            self.metadata_list,
            local_load_files,
            self.process_group,
            self.use_dist,
        )
        read_items = get_read_items(
            self.metadata_list,
            self.target_state_dict,
            self.process_group,
            self.use_dist,
            load_infos,
        )

        processed_target_state_dict = {
            k: v.local_tensor if isinstance(v, ShardedWeight) else v
            for k, v in self.target_state_dict.items()
        }
        has_tuple_key = any(
            isinstance(k, tuple) for k in processed_target_state_dict
        )
        has_non_tuple_key = any(
            not isinstance(k, tuple) for k in processed_target_state_dict
        )
        assert not (has_tuple_key and has_non_tuple_key), (
            "target_state_dict contains a mix of tuple and non-tuple keys."
        )
        return processed_target_state_dict, read_items

    def local_reshard(self, read_items, processed_target_state_dict):
        for read_item in read_items:
            src_tensor = self.source_state_dict[read_item.file_name][
                read_item.tensor_name
            ]
            src_chunk_tensor = slice_tensor(
                src_tensor, read_item.src_local_offset, read_item.slice_shape
            ).contiguous()
            dst_tensor = get_target_tensor(
                processed_target_state_dict, read_item
            )
            dst_chunk_tensor = slice_tensor(
                dst_tensor, read_item.dst_local_offset, read_item.slice_shape
            )
            if src_chunk_tensor.place != dst_chunk_tensor.place:
                src_chunk_tensor = src_chunk_tensor.to(dst_chunk_tensor.place)
            paddle.assign(src_chunk_tensor, dst_chunk_tensor)

    def reshard(self):
        cur_rank = paddle.distributed.get_rank()
        processed_target_state_dict, read_items = self.preprocess()

        logger.info(
            f"ReadItem generation completed, with a total of {len(read_items)}."
        )
        if not read_items:
            return processed_target_state_dict

        context = {
            'rank': cur_rank,
            'process_group': self.process_group,
        }

        state = {
            'source_state_dict': self.source_state_dict,
            'target_state_dict': processed_target_state_dict,
        }

        if self.use_dist:
            self.communicator.communicate(read_items, state, context)
        else:
            self.local_reshard(read_items, processed_target_state_dict)

        del self.source_state_dict
        return processed_target_state_dict


def assign_sharded_weight(src, dst):
    assert src.global_shape == dst.global_shape, (
        "Global shapes must be the same"
    )
    ndim = len(src.global_shape)
    starts, ends = [], []
    dst_starts, dst_ends = [], []

    for i in range(ndim):
        src_begin = src.global_offset[i]
        src_end = src_begin + src.local_shape[i]
        dst_begin = dst.global_offset[i]
        dst_end = dst_begin + dst.local_shape[i]

        overlap_begin = max(src_begin, dst_begin)
        overlap_end = min(src_end, dst_end)
        if overlap_end <= overlap_begin:
            return
        starts.append(overlap_begin - src_begin)
        ends.append(overlap_end - src_begin)
        dst_starts.append(overlap_begin - dst_begin)
        dst_ends.append(overlap_end - dst_begin)

    src_slice = paddle.slice(
        src.local_tensor, axes=list(range(ndim)), starts=starts, ends=ends
    )
    dst_slice = paddle.slice(
        dst.local_tensor,
        axes=list(range(ndim)),
        starts=dst_starts,
        ends=dst_ends,
    )
    paddle.assign(src_slice, dst_slice)


class ThreeDCommGroupStateResharder:
    def __init__(
        self,
        target_state_dict,
        source_state_dict,
        metadata_list,
        h_group,
        v_group,
        p_group,
        memory_growth_threshold: int = 8 * (2**30),  # 8GB
        offload=False,
    ):
        self.target_state_dict = target_state_dict
        self.source_state_dict = source_state_dict
        assert len(metadata_list) == 1, "Only support one metadata now!"
        self.metadata = metadata_list[0]
        self.h_group = h_group
        self.v_group = v_group
        for group, name in [
            (self.h_group, "horizontal"),
            (self.v_group, "vertical"),
        ]:
            assert group.nranks > 1, (
                f"The number of ranks in the {name} communication group must be greater than 1, "
                f"but actually it is {group.nranks}. Please check this communication group: {group}!"
            )
        self.p_group = p_group
        self.using_2d_comm_group = (not self.p_group) or (
            self.p_group.nranks == 1
        )
        self.memory_growth_threshold = memory_growth_threshold
        self.offload = offload
        self.using_tuple_key = True
        self.preprocess()

    def preprocess(self):
        if self.offload:
            for file_name, state_dict in self.source_state_dict.items():
                self.source_state_dict[file_name] = {
                    k: paddle.to_tensor(v, place=paddle.CPUPlace())
                    if isinstance(v, np.ndarray)
                    else v
                    for k, v in state_dict.items()
                }

            for file_name, state_dict in self.source_state_dict.items():
                for tensor_name, tensor in state_dict.items():
                    if tensor.dtype == paddle.float32:
                        state_dict[tensor_name] = tensor.cuda().pin_memory()
                    else:
                        state_dict[tensor_name] = tensor.cuda()

        self.local_load_files = list(self.source_state_dict.keys())

        has_tuple_key = any(
            isinstance(k, tuple) for k in self.target_state_dict
        )
        has_non_tuple_key = any(
            not isinstance(k, tuple) for k in self.target_state_dict
        )
        assert not (has_tuple_key and has_non_tuple_key), (
            "target_state_dict contains a mix of tuple and non-tuple keys."
        )
        assert all(
            isinstance(v, ShardedWeight)
            for _, v in self.target_state_dict.items()
        ), "All sharded weights must be ShardedWeight type."

        self.using_tuple_key = has_tuple_key

        self.grouped_target_state_dict = defaultdict(list)
        for key, sharded_weight in self.target_state_dict.items():
            if self.using_tuple_key:
                self.grouped_target_state_dict[key[0]].append(sharded_weight)
            else:
                self.grouped_target_state_dict[key].append(sharded_weight)

        self.cur_rank = paddle.distributed.get_rank()

        self._build_cross_section_topology()
        self.get_read_items()
        self.schedule_read_items()
        self.aggregate_global_read_items()

    def all_gather_cross_section_fn(self, info):
        h_group = self.h_group
        v_group = self.v_group

        h_obj_list = []
        paddle.distributed.all_gather_object(h_obj_list, info, h_group)

        v_obj_list = []
        paddle.distributed.all_gather_object(v_obj_list, h_obj_list, v_group)

        gathered_info = [x for sublist in v_obj_list for x in sublist]
        return gathered_info

    def _build_cross_section_topology(self):
        h_ranks = []
        self.topology = []
        paddle.distributed.all_gather_object(
            h_ranks, self.cur_rank, self.h_group
        )
        paddle.distributed.all_gather_object(
            self.topology, h_ranks, self.v_group
        )

        if not self.using_2d_comm_group:
            p_ranks = []
            paddle.distributed.all_gather_object(
                p_ranks, self.cur_rank, self.p_group
            )
        else:
            p_ranks = [self.cur_rank]

        self.parallel_index = {rank: i for i, rank in enumerate(p_ranks)}
        self.p_ranks = p_ranks
        self.cur_parallel_index = self.parallel_index[self.cur_rank]

        self.vertical_ranks = [set(col) for col in zip(*self.topology)]
        self.horizontal_index = {
            rank: i
            for i, ranks in enumerate(self.vertical_ranks)
            for rank in ranks
        }
        self.vertical_index = {
            rank: i for i, row in enumerate(self.topology) for rank in row
        }

        self.cur_horizontal_index = self.horizontal_index[self.cur_rank]
        self.h_group_size = self.h_group.nranks
        self.v_group_size = self.v_group.nranks

    # NOTE(xingmingyyj) : maybe not need this function
    def dedup_read_items(self, global_read_items):
        group = defaultdict(list)
        for item in global_read_items:
            key = (item.tensor_name, item.src_global_offset, item.slice_shape)
            group[key].append(item)
        result = []
        for key, items in group.items():
            min_item = min(items, key=lambda x: x.src_rank)
            result.append(min_item)
        return result

    def get_read_items(
        self,
        all_gather_args=None,
    ):
        current_rank = paddle.distributed.get_rank()
        state_dict_metadata = self.metadata.state_dict_metadata
        storage_metadata = self.metadata.storage_metadata

        shard_infos = {}
        for local_tensor_index, file_name in storage_metadata.items():
            tensor_key = local_tensor_index.tensor_key
            local_tensor_metadata = state_dict_metadata[tensor_key]
            assert len(local_tensor_metadata) != 0, (
                f"No metadata found for tensor with name {tensor_key} in file {file_name}"
            )
            global_shape = local_tensor_metadata[0].global_shape
            key = (tensor_key, file_name)
            shard_info = (
                global_shape,
                local_tensor_index.local_shape,
                local_tensor_index.global_offset,
            )
            shard_infos[key] = shard_info

        local_read_plan = []
        for read_file, state_dict in self.source_state_dict.items():
            for tensor_name, tensor in state_dict.items():
                global_shape, local_shape, global_offset = shard_infos[
                    (tensor_name, read_file)
                ]
                dtype = str(tensor.dtype).split(".")[1]
                assert tuple(tensor.shape) == tuple(local_shape), (
                    f"Shape mismatch in  tensor name {tensor_name} in file {read_file}, expected shape {local_shape}, but got {tuple(tensor.shape)}"
                )
                common_attrs = {
                    "tensor_name": tensor_name,
                    "src_rank": current_rank,
                    "src_global_offset": tuple(global_offset),
                    "dst_global_offset": tuple(global_offset),
                    "src_local_offset": (0,) * len(local_shape),
                    "dst_local_offset": (0,) * len(local_shape),
                    "slice_shape": tuple(local_shape),
                    "global_shape": tuple(global_shape),
                    "file_name": read_file,
                    "dtype": dtype,
                    "dst_rank": None,
                    "comm_group": None,
                }
                local_read_plan.append(ExtendReadItem(**common_attrs))

        gathered_plans_per_rank = self.all_gather_cross_section_fn(
            local_read_plan
        )

        global_read_plan_per_section = [
            item for plan in gathered_plans_per_rank for item in plan
        ]

        self.read_items = self.dedup_read_items(global_read_plan_per_section)

    def schedule_read_items(self):
        vertical_ranks = self.vertical_ranks
        global_broadcast_read_items = []
        bucket_read_items = defaultdict(list)
        for item in self.read_items:
            cur_dtype = item.dtype
            cur_shape = item.slice_shape
            element_size = paddle.core.size_of_dtype(getattr(paddle, cur_dtype))
            memory_growth = (
                element_size * math.prod(cur_shape) * len(vertical_ranks)
            )
            if memory_growth > self.memory_growth_threshold:
                global_broadcast_read_items.append(item)
                continue
            else:
                key = (cur_shape, cur_dtype)
                bucket_read_items[key].append(item)

        bucket_read_items_t = sorted(
            bucket_read_items.items(),
            key=lambda x: (
                x[0][0],
                x[0][1],
            ),
        )

        bucket_read_items = dict(bucket_read_items_t)

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

        for k, v in bucket_read_items.items():
            bucket_read_items[k] = sorted(v, key=order_rules)

        batch_read_items = []
        for (cur_shape, cur_dtype), items in list(bucket_read_items.items()):
            if len(items) < self.h_group_size:
                continue

            while len(items) >= self.h_group_size:
                cur_batch_read_items = [None] * len(vertical_ranks)
                cnt = 0
                used_indices = set()

                for i, item in enumerate(items):
                    if i in used_indices:
                        continue
                    src_rank = item.src_rank
                    h_index = self.horizontal_index[src_rank]
                    if cur_batch_read_items[h_index] is None:
                        cur_batch_read_items[h_index] = item
                        used_indices.add(i)
                        cnt += 1
                        if cnt == len(vertical_ranks):
                            break

                if all(i is not None for i in cur_batch_read_items):
                    batch_read_items.append(
                        (cur_batch_read_items, AllGatherType.NO_PADDING)
                    )
                    items = [
                        item
                        for i, item in enumerate(items)
                        if i not in used_indices
                    ]
                    bucket_read_items[(cur_shape, cur_dtype)] = items
                else:
                    break

        while len(bucket_read_items) != 0:
            cur_batch_read_items = [None] * len(vertical_ranks)
            cur_batch_dtype = None
            used_indices = defaultdict(set)
            cnt = 0

            for (cur_shape, cur_dtype), items in bucket_read_items.items():
                cur_batch_dtype = cur_dtype
                break

            for (cur_shape, cur_dtype), items in bucket_read_items.items():
                if cur_dtype != cur_batch_dtype:
                    continue
                for i, item in enumerate(items):
                    src_rank = item.src_rank
                    h_index = self.horizontal_index[src_rank]
                    if cur_batch_read_items[h_index] is None:
                        cur_batch_read_items[h_index] = item
                        used_indices[(cur_shape, cur_dtype)].add(i)
                        cnt += 1
                        if cnt == len(vertical_ranks):
                            break

            need_remove = []
            for key, items in list(bucket_read_items.items()):
                remaining_items = [
                    item
                    for i, item in enumerate(items)
                    if i not in used_indices[key]
                ]
                if len(remaining_items) == 0:
                    need_remove.append(key)
                else:
                    bucket_read_items[key] = remaining_items

            for key in need_remove:
                del bucket_read_items[key]

            for i, item in enumerate(cur_batch_read_items):
                if item is None:
                    src_rank = min(vertical_ranks[i])
                    common_attrs = {
                        "tensor_name": INTERNAL_PADDING_TENSOR_NAME,
                        "src_rank": src_rank,
                        "src_global_offset": (0,),
                        "dst_global_offset": (0,),
                        "src_local_offset": (0,),
                        "dst_local_offset": (0,),
                        "slice_shape": (1,),
                        "global_shape": (1,),
                        "file_name": "padding_vfile",
                        "dtype": cur_batch_dtype,
                        "comm_group": None,
                    }

                    padding_read_item = ExtendReadItem(
                        dst_rank=None, **common_attrs
                    )
                    cur_batch_read_items[i] = padding_read_item
            batch_read_items.append(
                (cur_batch_read_items, AllGatherType.WITH_PADDING)
            )

        self.global_broadcast_read_items = global_broadcast_read_items
        self.batch_read_items = batch_read_items

    def aggregate_global_read_items(self):
        if self.using_2d_comm_group:
            self.aggregated_global_broadcast_read_items = (
                self.global_broadcast_read_items
            )
            self.aggregated_batch_read_items = [
                [batch_items] for batch_items in self.batch_read_items
            ]
            return
        aggregated_global_broadcast_read_items = []
        aggregated_batch_read_items = []

        dist.all_gather_object(
            aggregated_global_broadcast_read_items,
            self.global_broadcast_read_items,
            self.p_group,
        )
        dist.all_gather_object(
            aggregated_batch_read_items,
            self.batch_read_items,
            self.p_group,
        )
        self.aggregated_global_broadcast_read_items = [
            item
            for sublist in aggregated_global_broadcast_read_items
            for item in sublist
        ]
        self.aggregated_batch_read_items = []  # [[[batch1],[batch2],,,,],]
        max_tasks = max(
            [len(sublist) for sublist in aggregated_batch_read_items]
        )
        for i in range(max_tasks):
            task_batches = []
            for batch_read_items in aggregated_batch_read_items:
                if len(batch_read_items) != 0:
                    task_batches.append(batch_read_items.pop(0))
                else:
                    task_batches.append(([], None))
            self.aggregated_batch_read_items.append(task_batches)

    def _process_one_batch_broadcast_in_section(self, batch_items):
        """Performs V-Broadcast + H-AllGather for one batch of items."""
        read_items, allgather_type = batch_items
        if len(read_items) == 0:
            return []

        read_item = read_items[self.cur_horizontal_index]
        if self.cur_rank == read_item.src_rank:
            buffer = (
                paddle.empty(read_item.slice_shape, read_item.dtype)
                if read_item.tensor_name == INTERNAL_PADDING_TENSOR_NAME
                else self.source_state_dict[read_item.file_name][
                    read_item.tensor_name
                ]
            )
            if not isinstance(buffer.place, paddle.CUDAPlace):
                buffer = buffer.cuda()
        else:
            buffer = paddle.empty(read_item.slice_shape, dtype=read_item.dtype)
        paddle.distributed.broadcast(
            buffer, src=read_item.src_rank, group=self.v_group
        )
        tensor_list = []
        if allgather_type == AllGatherType.WITH_PADDING:
            max_numel = max(math.prod(item.slice_shape) for item in read_items)
            if math.prod(buffer.shape) == max_numel:
                buffer = buffer.reshape(
                    [
                        max_numel,
                    ]
                )
            else:
                numel = buffer.numel()
                padded_buffer = paddle.zeros([max_numel], dtype=buffer.dtype)
                padded_buffer[:numel] = paddle.reshape(buffer, [-1])
                buffer._clear()
                buffer = padded_buffer
            paddle.distributed.all_gather(
                tensor_list, buffer, group=self.h_group
            )
            unpadded_tensor_list = []
            for idx, padded_tensor in enumerate(tensor_list):
                read_item = read_items[idx]
                numel = math.prod(read_item.slice_shape)
                unpadded_tensor = (
                    padded_tensor[:numel].clone().reshape(read_item.slice_shape)
                )
                unpadded_tensor_list.append(unpadded_tensor)
                padded_tensor._clear()
            tensor_list = unpadded_tensor_list
        else:
            paddle.distributed.all_gather(
                tensor_list, buffer, group=self.h_group
            )

        # NOTE(xingmingyyj) Release the GPU memory occupied by source_state_dict in advance.
        buffer._clear()

        return tensor_list

    def broadcast_cross_p_group_and_assign(self, tensor_list, task_batches):
        batch_read_items, allgather_type = task_batches[self.cur_parallel_index]
        need_remove_indices = set()
        for idx, read_item in enumerate(batch_read_items):
            if read_item.tensor_name == INTERNAL_PADDING_TENSOR_NAME:
                need_remove_indices.add(idx)

        for idx in sorted(need_remove_indices, reverse=True):
            del tensor_list[idx]

        filtered_read_items = []
        for idx, (batch_read_items, allgather_type) in enumerate(task_batches):
            src_rank = self.p_ranks[idx]
            for read_item in batch_read_items:
                if read_item.tensor_name != INTERNAL_PADDING_TENSOR_NAME:
                    replcaed_read_item = replace(read_item, src_rank=src_rank)
                    filtered_read_items.append(replcaed_read_item)

        cnt = 0
        for idx, read_item in enumerate(filtered_read_items):
            if not self.using_2d_comm_group:
                if read_item.src_rank == self.cur_rank:
                    buffer = tensor_list[cnt]
                    cnt += 1
                else:
                    buffer = paddle.empty(
                        read_item.slice_shape, dtype=read_item.dtype
                    )

                paddle.distributed.broadcast(
                    buffer, src=read_item.src_rank, group=self.p_group
                )
            else:
                buffer = tensor_list[cnt]
                cnt += 1

            received_sharded_weight = ShardedWeight(
                key=read_item.tensor_name,
                local_tensor=buffer,
                local_shape=read_item.slice_shape,
                global_shape=read_item.global_shape,
                global_offset=read_item.src_global_offset,
            )

            for target_sharded_weight in self.grouped_target_state_dict[
                read_item.tensor_name
            ]:
                if not target_sharded_weight.local_tensor._is_initialized():
                    buffer = paddle.zeros_like(
                        target_sharded_weight.local_tensor
                    )
                    buffer._share_buffer_to(target_sharded_weight.local_tensor)

                src_tensor = received_sharded_weight.local_tensor
                tgt_place = target_sharded_weight.local_tensor.place

                if src_tensor.place != tgt_place:
                    src_tensor = src_tensor.to(tgt_place)

                received_sharded_weight.local_tensor = src_tensor

                assign_sharded_weight(
                    src=received_sharded_weight,
                    dst=target_sharded_weight,
                )

            buffer._clear()
            del received_sharded_weight

    def broadcast_cross_global_group_and_assign(self):
        global_broadcast_read_items = (
            self.aggregated_global_broadcast_read_items
        )
        total_items = len(global_broadcast_read_items)
        for idx, read_item in enumerate(global_broadcast_read_items, start=1):
            if idx % 10 == 0 or idx == total_items:
                logger.info(
                    f"Broadcasting item {idx}/{total_items}: {read_item.tensor_name}"
                )
            if self.cur_rank == read_item.src_rank:
                buffer = self.source_state_dict[read_item.file_name][
                    read_item.tensor_name
                ]
                if not isinstance(buffer.place, paddle.CUDAPlace):
                    buffer = buffer.cuda()
            else:
                buffer = paddle.empty(
                    read_item.slice_shape, dtype=read_item.dtype
                )
            # NOTE(xingmingyyj): using global group to broadcast
            paddle.distributed.broadcast(
                buffer, src=read_item.src_rank, group=None
            )
            received_sharded_weight = ShardedWeight(
                key=read_item.tensor_name,
                local_tensor=buffer,
                local_shape=read_item.slice_shape,
                global_shape=read_item.global_shape,
                global_offset=read_item.src_global_offset,
            )

            for target_sharded_weight in self.grouped_target_state_dict[
                read_item.tensor_name
            ]:
                assign_sharded_weight(
                    src=received_sharded_weight,
                    dst=target_sharded_weight,
                )

            buffer._clear()
            del received_sharded_weight

    def reshard(self):
        total = len(self.aggregated_batch_read_items)
        logger.info(
            "[ThreeDCommGroupStateResharder] Begin resharding using batch broadcasting..."
        )
        for idx, task_batches in enumerate(
            self.aggregated_batch_read_items, start=1
        ):
            tensor_list = self._process_one_batch_broadcast_in_section(
                task_batches[self.cur_parallel_index]
            )
            self.broadcast_cross_p_group_and_assign(tensor_list, task_batches)
            if idx % 10 == 0 or idx == total:
                logger.info(
                    f"Resharding batches: {idx}/{total} ({idx * 100 // total}%)"
                )
        logger.info(
            "[ThreeDCommGroupStateResharder] End resharding using batch broadcasting..."
        )
        logger.info(
            "[ThreeDCommGroupStateResharder] Begin resharding using global broadcasting..."
        )
        self.broadcast_cross_global_group_and_assign()
        logger.info(
            "[ThreeDCommGroupStateResharder] End resharding using global broadcasting..."
        )
        logger.info("[ThreeDCommGroupStateResharder] Resharding finished.")
