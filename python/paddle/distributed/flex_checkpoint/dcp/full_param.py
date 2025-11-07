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
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import paddle

from ..aoa.aoa_engine import AOAEngine
from .load_state_dict import (
    ReadItem,
)
from .sharded_weight import (
    ShardedWeight,
    ShardedWeightDesc,
)
from .utils import (
    assign_sharded_slice,
    build_global_state_shard_info,
    recover_shard_tensor_from_shards,
)

if TYPE_CHECKING:
    from paddle.distributed.collective import Group
    from paddle.nn import Layer


SUPPORTED_DTYPES = ['float16', 'float32', 'bfloat16']


def infer_real_dtype(desc) -> str:
    found_dtypes = []
    for slice_ref in desc.slices:
        key, sl_src, sl_dst, pp_list = slice_ref
        if pp_list is None or len(pp_list) == 0:
            continue
        last_supported = None
        for item in reversed(pp_list):
            if item in SUPPORTED_DTYPES:
                last_supported = item
                break
        if last_supported:
            found_dtypes.append(last_supported)
    if not found_dtypes:
        return desc.dtype

    dtype_set = set(found_dtypes)
    if len(dtype_set) > 1:
        raise ValueError(
            f"Found multiple different dtypes from slices: {dtype_set}"
        )
    return found_dtypes[0]


@dataclass(frozen=True)
class ExtendReadItem(ReadItem):
    target_tensor_names: tuple[str] | None = None
    global_shape: tuple[int] | None = None


def dedup_read_items(global_read_items, world_size):
    group = defaultdict(list)
    for item in global_read_items:
        key = (item.tensor_name, item.src_global_offset, item.slice_shape)
        group[key].append(item)
    result = []
    for key, items in group.items():
        min_item = min(items, key=lambda x: x.src_rank)
        src_rank = min_item.src_rank
        result.append(replace(min_item, dst_rank=(src_rank,)))
        other_ranks = tuple(i for i in range(world_size) if i != src_rank)
        result.append(replace(min_item, dst_rank=other_ranks))
    return result


def get_read_items(
    source_sharded_state_dict: dict[str, ShardedWeight],
    source_to_target_names,
    world_size: int,
    process_group: Group | None = None,
):
    current_rank = paddle.distributed.get_rank()
    rank_vfile = f"{current_rank}.vdistcp"

    local_read_plan = []
    self_rank_tuple = (current_rank,)
    remote_ranks_tuple = tuple(
        r for r in range(world_size) if r != current_rank
    )

    for tensor_name, shard_info in source_sharded_state_dict.items():
        common_attrs = {
            "tensor_name": tensor_name,
            "src_rank": current_rank,
            "src_global_offset": tuple(shard_info.global_offset),
            "dst_global_offset": tuple(shard_info.global_offset),
            "src_local_offset": (0,) * len(shard_info.local_shape),
            "dst_local_offset": (0,) * len(shard_info.local_shape),
            "slice_shape": tuple(shard_info.local_shape),
            "global_shape": tuple(shard_info.global_shape),
            "target_tensor_names": tuple(source_to_target_names[tensor_name]),
            "file_name": rank_vfile,
            "dtype": str(shard_info.local_tensor.dtype).split(".")[1],
            "comm_group": None,
        }

        read_for_self = ExtendReadItem(dst_rank=self_rank_tuple, **common_attrs)
        local_read_plan.append(read_for_self)

        if remote_ranks_tuple:
            read_for_others = ExtendReadItem(
                dst_rank=remote_ranks_tuple, **common_attrs
            )
            local_read_plan.append(read_for_others)

    gathered_plans_per_rank = []
    paddle.distributed.all_gather_object(
        gathered_plans_per_rank, local_read_plan, process_group
    )

    global_read_plan = [
        item for plan in gathered_plans_per_rank for item in plan
    ]

    final_read_plan = dedup_read_items(global_read_plan, world_size)

    return final_read_plan


def group_read_items_by_tensor_name(global_read_items):
    groups = defaultdict(list)
    for item in global_read_items:
        groups[item.tensor_name].append(item)
    return groups


def sort_groups_for_early_release(groups, source_to_target_names):
    def count_fn(name):
        return len(source_to_target_names.get(name, []))

    sorted_items = sorted(groups.items(), key=lambda x: -count_fn(x[0]))
    return dict(sorted_items)


def build_reference_map(groups: dict[str, list[ExtendReadItem]]):
    ref_map = defaultdict(set)
    for _, items in groups.items():
        for item in items:
            for tgt in item.target_tensor_names:
                ref_map[tgt].add(item)
    return ref_map


class TensorBuffer:
    def __init__(self, buffer_size: int = 128, dtype: str = 'bfloat16'):
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.current_size = 0
        self.tensors = []
        self._buffer = paddle.empty(
            shape=[self.buffer_size],
            dtype=self.dtype,
        )

    def append(self, tensor: paddle.Tensor) -> bool:
        if tensor.dtype != self._buffer.dtype:
            raise TypeError(
                f"dtype mismatch: buffer is {self._buffer.dtype}, tensor is {tensor.dtype}"
            )
        numel = tensor.numel()
        if self.current_size + numel > self.buffer_size:
            return False

        self.tensors.append(tensor)

        start = self.current_size
        end = start + numel
        buffer_slice = paddle.slice(
            self._buffer, axes=[0], starts=[start], ends=[end]
        )
        paddle.assign(tensor.flatten(), buffer_slice)
        self.current_size += numel
        return True

    def recover(self) -> list:
        tensors = []
        offset = 0
        for tensor in self.tensors:
            numel = tensor.numel()
            tensor_slice = paddle.slice(
                self._buffer, axes=[0], starts=[offset], ends=[offset + numel]
            )
            paddle.assign(tensor_slice, tensor.flatten())
            tensors.append(tensor)
            offset += numel
        return tensors

    def get_buffer(self) -> paddle.Tensor:
        cur_buffer = paddle.slice(
            self._buffer, axes=[0], starts=[0], ends=[self.current_size]
        )
        return cur_buffer

    def clear(self):
        self.current_size = 0
        self.tensors = []

    def destroy(self):
        self._buffer._clear()


def is_identity_mapping(shard_mappings):
    if len(shard_mappings) != 1:
        return False
    mapping = shard_mappings[0]
    src = mapping.source_slice
    dst = mapping.target_slice
    return (
        src.key == dst.key
        and src.local_shape == dst.local_shape
        and src.global_shape == dst.global_shape
        and src.global_offset == dst.global_offset
        and src.dtype == dst.dtype
        and mapping.postprocess_list is None
    )


def full_param(
    model: Layer,
    aoa_config: dict[str, list[str]] | None = None,
    process_group: Group | None = None,
):
    cur_rank = paddle.distributed.get_rank()
    world_size = paddle.distributed.get_world_size()
    use_dist = True if world_size > 1 else False

    source_sharded_state_dict = model.sharded_state_dict()
    source_state_shard_info = build_global_state_shard_info(
        source_sharded_state_dict, process_group
    )

    aoa_config = aoa_config if aoa_config is not None else {}

    aoa_engine = AOAEngine(
        aoa_config=aoa_config,
        source_state_shard_info=source_state_shard_info,
        destination_state_shard_info=None,
    )

    destination_sharded_weight_desc = {}
    for k, v in aoa_engine.output_vars.items():
        dtype = infer_real_dtype(v)
        destination_sharded_weight_desc[k] = ShardedWeightDesc(
            key=k,
            local_shape=v.shape,
            global_shape=v.shape,
            global_offset=(0,) * len(v.shape),
            dtype=dtype,
        )

    destination_sharded_mappings = {}
    for k, v in destination_sharded_weight_desc.items():
        shard_mappings = aoa_engine.find_shard_sources(v)
        destination_sharded_mappings[k] = shard_mappings

    if not use_dist:
        for k, v in source_sharded_state_dict.items():
            assert v.local_shape == v.global_shape, (
                "On a single card, each parameter must not be sharded."
            )

        for k, shard_mappings in destination_sharded_mappings.items():
            if is_identity_mapping(shard_mappings):
                src_key = shard_mappings[0].source_slice.key
                yield k, source_sharded_state_dict[src_key].local_tensor
            else:
                desc = destination_sharded_weight_desc[k]
                local_tensor = paddle.empty(desc.local_shape, dtype=desc.dtype)
                cur_sharded_tensor = ShardedWeight(
                    key=desc.key,
                    local_tensor=local_tensor,
                    local_shape=desc.local_shape,
                    global_shape=desc.global_shape,
                    global_offset=desc.global_offset,
                )
                for mapping in shard_mappings:
                    src_desc = mapping.source_slice
                    dst_desc = mapping.target_slice
                    source_sharded_tensor = source_sharded_state_dict[
                        src_desc.key
                    ]
                    assign_sharded_slice(
                        src_desc,
                        source_sharded_tensor,
                        dst_desc,
                        cur_sharded_tensor,
                        postprocess_list=mapping.postprocess_list,
                    )
                yield k, cur_sharded_tensor.local_tensor

    else:
        source_to_target_names = defaultdict(set)
        for k, mapping in destination_sharded_mappings.items():
            for m in mapping:
                source_to_target_names[m.source_slice.key].add(k)

        read_items = get_read_items(
            source_sharded_state_dict=source_sharded_state_dict,
            source_to_target_names=source_to_target_names,
            world_size=world_size,
            process_group=process_group,
        )

        grouped_read_items = group_read_items_by_tensor_name(read_items)
        grouped_read_items = sort_groups_for_early_release(
            grouped_read_items, source_to_target_names
        )
        ref_map = build_reference_map(grouped_read_items)
        read_items = []
        for _, items in grouped_read_items.items():
            read_items.extend(items)

        buffer_size = max(
            256 * 1024 * 1024,
            max(
                (math.prod(item.slice_shape) for item in read_items), default=0
            ),
        )

        tensor_buffer = TensorBuffer(buffer_size=buffer_size)

        sharded_desc_to_tensor = {}

        ref_count = deepcopy(source_to_target_names)

        while len(read_items) != 0:
            read_items_comm_bf16 = []
            read_items_comm_other = []
            read_items_local = []
            cur_batch_full_tensors = {}
            first_item = read_items[0]
            cur_src_rank = first_item.src_rank
            for item in read_items:
                if (
                    len(item.dst_rank) == 1
                    and item.dst_rank[0] == item.src_rank
                ):
                    if item.src_rank == cur_rank:
                        shard_desc = ShardedWeightDesc(
                            key=item.tensor_name,
                            local_shape=item.slice_shape,
                            global_shape=item.global_shape,
                            global_offset=item.src_global_offset,
                            dtype=item.dtype,
                        )
                        cur_tensor = source_sharded_state_dict[
                            item.tensor_name
                        ].local_tensor.clone()

                        assert tuple(cur_tensor.shape) == item.slice_shape
                        sharded_desc_to_tensor[shard_desc] = cur_tensor
                    read_items_local.append(item)

                elif item.src_rank == cur_src_rank and item.dtype == 'bfloat16':
                    if item.src_rank == cur_rank:
                        tensor_name = item.tensor_name
                        assert tensor_name in source_sharded_state_dict
                        local_tensor = source_sharded_state_dict[
                            tensor_name
                        ].local_tensor.clone()
                        assert tuple(local_tensor.shape) == item.slice_shape
                        if not tensor_buffer.append(local_tensor):
                            break
                    else:
                        tmp_tensor = paddle.empty(
                            item.slice_shape, dtype=item.dtype
                        )
                        if not tensor_buffer.append(tmp_tensor):
                            tmp_tensor._clear()
                            break
                    read_items_comm_bf16.append(item)
                elif item.src_rank == cur_src_rank and item.dtype != 'bfloat16':
                    if item.src_rank == cur_rank:
                        tensor_name = item.tensor_name
                        assert tensor_name in source_sharded_state_dict
                        local_tensor = source_sharded_state_dict[
                            tensor_name
                        ].local_tensor.clone()
                    else:
                        local_tensor = paddle.empty(
                            item.slice_shape, dtype=item.dtype
                        )
                    paddle.distributed.broadcast(
                        local_tensor, src=cur_src_rank, group=process_group
                    )
                    shard_desc = ShardedWeightDesc(
                        key=item.tensor_name,
                        local_shape=item.slice_shape,
                        global_shape=item.global_shape,
                        global_offset=item.src_global_offset,
                        dtype=item.dtype,
                    )
                    sharded_desc_to_tensor[shard_desc] = local_tensor
                    read_items_comm_other.append(item)

            if tensor_buffer.current_size > 0:
                paddle.distributed.broadcast(
                    tensor_buffer.get_buffer(),
                    src=cur_src_rank,
                    group=process_group,
                )

                tensors = tensor_buffer.recover()
                tensor_buffer.clear()

                for idx, item in enumerate(read_items_comm_bf16):
                    shard_desc = ShardedWeightDesc(
                        key=item.tensor_name,
                        local_shape=item.slice_shape,
                        global_shape=item.global_shape,
                        global_offset=item.src_global_offset,
                        dtype=item.dtype,
                    )

                    sharded_desc_to_tensor[shard_desc] = tensors[idx]

            cur_batch_read_items = (
                read_items_comm_bf16 + read_items_comm_other + read_items_local
            )
            ready_tensor_names = []
            for item in cur_batch_read_items:
                for name in item.target_tensor_names:
                    ref_map[name].remove(item)
                    if len(ref_map[name]) == 0:
                        ready_tensor_names.append(name)

            for name in ready_tensor_names:
                del ref_map[name]

            for item in cur_batch_read_items:
                read_items.remove(item)

            need_clear_tensor_names = []

            for name in ready_tensor_names:
                target_sharded_weight_desc = destination_sharded_weight_desc[
                    name
                ]
                local_tensor = paddle.empty(
                    target_sharded_weight_desc.local_shape,
                    dtype=target_sharded_weight_desc.dtype,
                )
                cur_sharded_tensor = ShardedWeight(
                    key=target_sharded_weight_desc.key,
                    local_tensor=local_tensor,
                    local_shape=target_sharded_weight_desc.local_shape,
                    global_shape=target_sharded_weight_desc.global_shape,
                    global_offset=target_sharded_weight_desc.global_offset,
                )
                mappings = destination_sharded_mappings[name]
                for mapping in mappings:
                    src_desc = mapping.source_slice
                    dst_desc = mapping.target_slice
                    src_shard = ShardedWeight(
                        key=src_desc.key,
                        local_tensor=paddle.zeros(
                            src_desc.local_shape, dtype=src_desc.dtype
                        ),
                        local_shape=src_desc.local_shape,
                        global_shape=src_desc.global_shape,
                        global_offset=src_desc.global_offset,
                    )

                    sharded_weights = []

                    for desc, local_tensor in sharded_desc_to_tensor.items():
                        if desc.key != src_desc.key:
                            continue
                        cur_shard = ShardedWeight(
                            key=src_desc.key,
                            local_tensor=local_tensor,
                            local_shape=desc.local_shape,
                            global_shape=desc.global_shape,
                            global_offset=desc.global_offset,
                        )
                        sharded_weights.append(cur_shard)

                    recover_shard_tensor_from_shards(sharded_weights, src_shard)

                    assign_sharded_slice(
                        src_desc,
                        src_shard,
                        dst_desc,
                        cur_sharded_tensor,
                        postprocess_list=mapping.postprocess_list,
                    )

                    src_shard.local_tensor._clear()

                cur_batch_full_tensors[name] = cur_sharded_tensor.local_tensor

                need_clear_tensor_names = []
                del_keys = []

                for source_name in list(ref_count.keys()):
                    target_names = ref_count[source_name]
                    if name in target_names:
                        target_names.remove(name)
                        if len(target_names) == 0:
                            del_keys.append(source_name)
                            need_clear_tensor_names.append(source_name)

                for k in del_keys:
                    del ref_count[k]

            to_delete = []

            for src_desc in sharded_desc_to_tensor:
                if src_desc.key in need_clear_tensor_names:
                    local_tensor = sharded_desc_to_tensor[src_desc]
                    local_tensor._clear()
                    to_delete.append(src_desc)

            for src_desc in to_delete:
                del sharded_desc_to_tensor[src_desc]

            if len(read_items) == 0:
                tensor_buffer.clear()
                tensor_buffer.destroy()
            for name, tensor in cur_batch_full_tensors.items():
                yield name, tensor
