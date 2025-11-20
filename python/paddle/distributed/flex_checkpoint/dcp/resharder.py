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

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import paddle
from paddle.distributed.fleet.utils.log_util import logger

from .metadata import LocalTensorIndex, LocalTensorMetadata
from .sharded_weight import (
    ShardedWeight,
)
from .utils import (
    compute_local_shape_and_global_offset,
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


def schedule_read_items(
    comm_read_items: list[ReadItem],
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
    for item in comm_read_items:
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
        use_group=True,
    ):
        self.target_state_dict = target_state_dict
        self.source_state_dict = source_state_dict
        self.metadata_list = metadata_list
        self.communicator = communicator
        self.process_group = process_group
        self.offload = offload
        self.use_dist = use_dist
        self.use_group = use_group

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

    def reshard(self):
        cur_rank = paddle.distributed.get_rank()
        processed_target_state_dict, read_items = self.preprocess()

        logger.info(
            f"ReadItem generation completed, with a total of {len(read_items)}."
        )
        comm_tasks = schedule_read_items(read_items)
        if not read_items:
            return processed_target_state_dict

        context = {
            'rank': cur_rank,
            'process_group': self.process_group,
            'use_group': self.use_group,
        }
        state = {
            'source_state_dict': self.source_state_dict,
            'target_state_dict': processed_target_state_dict,
        }

        self.communicator.communicate(comm_tasks, state, context)

        del self.source_state_dict
        logger.info("All communication tasks completed.")
        return processed_target_state_dict
