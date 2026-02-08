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

import abc
import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
)

import paddle

from ..aoa.aoa_engine import SUPPORTED_DTYPES, AOAEngine
from .resharder import (
    ReadItem,
)
from .sharded_weight import (
    ShardedWeight,
    ShardedWeightDesc,
)
from .utils import (
    assign_sharded_slice,
    build_shard_desc,
    merge_shard_info_list,
    recover_shard_tensor_from_shards,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from paddle.distributed.collective import Group

    from .sharded_weight import ShardedStateDict


INTERNAL_PADDING_TENSOR_NAME = "__internal_padding_tensor_name__"


@dataclass(frozen=True)
class ExtendReadItem(ReadItem):
    target_tensor_names: tuple[str] | None = None
    global_shape: tuple[int] | None = None


class BaseAssembler(abc.ABC):
    """
    Abstract base class for assembling full parameters from sharded states.

    This class encapsulates the common logic for:
        1.  Analyzing source and destination tensor mappings (AOA).
        2.  Creating a plan to read/communicate necessary tensor shards.
        3.  Assembling final tensors once all their source shards are available.
        4.  Managing memory by cleaning up consumed shards.

    Subclasses must implement the `run` method, which defines the specific
    distributed communication strategy to fetch the tensor shards.
    """

    def __init__(
        self,
        sharded_state_dict: ShardedStateDict,
        aoa_config: dict[str, list[str]] | None = None,
        num_splits: int = 1,
        idx: int = 0,
    ):
        self.sharded_state_dict = sharded_state_dict
        self.aoa_config = aoa_config or {}
        self.num_splits = num_splits
        self.idx = idx

        self.cur_rank: int = paddle.distributed.get_rank()
        self.world_size: int = paddle.distributed.get_world_size()
        self.use_dist: bool = self.world_size > 1

        self.filtered_sharded_state_dict = {}
        self.aoa_engine = None
        self.destination_sharded_weight_desc: dict[str, ShardedWeightDesc] = {}
        self.destination_sharded_mappings = {}

        self.source_to_target_names: dict[str, set[str]] = defaultdict(set)
        self.source_consumers: dict[str, set[str]] = {}
        self.ref_map: dict[str, set] = {}
        self.read_items: list[ExtendReadItem] = []

        self.sharded_desc_to_tensor: dict[ShardedWeightDesc, paddle.Tensor] = {}

    def _prepare_metainfo(self, source_state_shard_info):
        """Builds destination descriptions and mappings using AOAEngine."""
        self.aoa_engine = AOAEngine(
            aoa_config=self.aoa_config,
            source_state_shard_info=source_state_shard_info,
            destination_state_shard_info=None,
        )

        output_vars = self.split_output_vars()

        for k, v in output_vars.items():
            dtype = self.infer_real_dtype(v)
            self.destination_sharded_weight_desc[k] = ShardedWeightDesc(
                key=k,
                local_shape=v.shape,
                global_shape=v.shape,
                global_offset=(0,) * len(v.shape),
                dtype=dtype,
            )

        for k, desc in self.destination_sharded_weight_desc.items():
            self.destination_sharded_mappings[k] = (
                self.aoa_engine.find_shard_sources(desc)
            )

        for tgt_name, mapping in self.destination_sharded_mappings.items():
            for m in mapping:
                self.source_to_target_names[m.source_slice.key].add(tgt_name)

        self.filtered_sharded_state_dict = {
            k: v
            for k, v in self.sharded_state_dict.items()
            if k in self.source_to_target_names
        }

        self.source_consumers = deepcopy(self.source_to_target_names)

    def split_output_vars(self):
        data_dict = self.aoa_engine.output_vars
        if self.num_splits < 1:
            raise ValueError('num_splits must be >= 1')
        if self.idx < 0 or self.idx >= self.num_splits:
            raise IndexError(f'idx must be in [0,{self.num_splits - 1}]')

        sorted_keys = sorted(data_dict.keys())
        total = len(sorted_keys)
        base = total // self.num_splits
        extra = total % self.num_splits

        if self.idx < extra:
            start = self.idx * (base + 1)
            end = start + (base + 1)
        else:
            start = extra * (base + 1) + (self.idx - extra) * base
            end = start + base

        selected_keys = sorted_keys[start:end]
        return {k: data_dict[k] for k in selected_keys}

    def _assemble_and_yield_ready_tensors(
        self, ready_tensor_names: list[str]
    ) -> Iterable[tuple[str, paddle.Tensor]]:
        """
        Assembles, yields, and cleans up tensors whose dependencies are all met.
        This logic is shared across different communication strategies.
        """
        if not ready_tensor_names:
            return

        for name in ready_tensor_names:
            target_desc = self.destination_sharded_weight_desc[name]
            local_tensor = paddle.empty(
                target_desc.local_shape, dtype=target_desc.dtype
            )
            cur_sharded_tensor = ShardedWeight(
                key=target_desc.key,
                local_tensor=local_tensor,
                local_shape=target_desc.local_shape,
                global_shape=target_desc.global_shape,
                global_offset=target_desc.global_offset,
            )

            for mapping in self.destination_sharded_mappings[name]:
                src_desc = mapping.source_slice
                dst_desc = mapping.target_slice

                src_shard_template = ShardedWeight(
                    key=src_desc.key,
                    local_tensor=paddle.zeros(
                        src_desc.local_shape, dtype=src_desc.dtype
                    ),
                    local_shape=src_desc.local_shape,
                    global_shape=src_desc.global_shape,
                    global_offset=src_desc.global_offset,
                )

                received_shards = []
                for desc, tensor in self.sharded_desc_to_tensor.items():
                    if desc.key == src_desc.key:
                        received_shards.append(
                            ShardedWeight(
                                key=desc.key,
                                local_tensor=tensor,
                                local_shape=desc.local_shape,
                                global_shape=desc.global_shape,
                                global_offset=desc.global_offset,
                            )
                        )

                recover_shard_tensor_from_shards(
                    received_shards, src_shard_template
                )

                assign_sharded_slice(
                    src_desc=src_desc,
                    src_shard=src_shard_template,
                    dst_desc=dst_desc,
                    dst_shard=cur_sharded_tensor,
                    postprocess_list=mapping.postprocess_list,
                )
                src_shard_template.local_tensor._clear()

            yield name, cur_sharded_tensor.local_tensor

        need_clear_source_names = self._update_consumer_counts(
            ready_tensor_names
        )

        self._cleanup_consumed_shards(need_clear_source_names)

    def _update_consumer_counts(
        self, ready_tensor_names: list[str]
    ) -> list[str]:
        """Decrement consumer counts and return source names that can be cleared."""
        need_clear_source_names = []
        del_keys = []
        for source_name, target_names in self.source_consumers.items():
            target_names.difference_update(ready_tensor_names)
            if not target_names:
                del_keys.append(source_name)
                need_clear_source_names.append(source_name)

        for k in del_keys:
            del self.source_consumers[k]

        return need_clear_source_names

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

    def _cleanup_consumed_shards(self, source_names_to_clear: list[str]):
        """Delete cached tensors corresponding to the given source names."""
        if not source_names_to_clear:
            return

        to_delete_descs = []
        for desc, tensor in self.sharded_desc_to_tensor.items():
            if desc.key in source_names_to_clear:
                tensor._clear()
                to_delete_descs.append(desc)

        for desc in to_delete_descs:
            del self.sharded_desc_to_tensor[desc]

    @abc.abstractmethod
    def prepare(self):
        """Subclasses must implement this to build their specific read plan."""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self) -> Generator[tuple[str, paddle.Tensor], None, None]:
        """
        The main entry point. Subclasses must implement their communication
        loop and yield final tensors.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def all_gather_fn(self, info, **kwargs):
        raise NotImplementedError

    def infer_real_dtype(self, desc) -> str:
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

    def build_global_state_shard_info(self, **all_gather_args):
        state_shard_info = defaultdict(list)
        for key, val in self.sharded_state_dict.items():
            desc = build_shard_desc(val)
            state_shard_info[key].append(desc)

        use_dist = True if paddle.distributed.get_world_size() > 1 else False

        if use_dist:
            gathered_info = self.all_gather_fn(
                dict(state_shard_info), **all_gather_args
            )
        else:
            gathered_info = [dict(state_shard_info)]

        return merge_shard_info_list(gathered_info)

    def get_read_items(
        self,
        all_gather_args=None,
    ):
        current_rank = paddle.distributed.get_rank()
        rank_vfile = f"{current_rank}.vdistcp"

        local_read_plan = []
        for tensor_name, shard_info in self.filtered_sharded_state_dict.items():
            common_attrs = {
                "tensor_name": tensor_name,
                "src_rank": current_rank,
                "src_global_offset": tuple(shard_info.global_offset),
                "dst_global_offset": tuple(shard_info.global_offset),
                "src_local_offset": (0,) * len(shard_info.local_shape),
                "dst_local_offset": (0,) * len(shard_info.local_shape),
                "slice_shape": tuple(shard_info.local_shape),
                "global_shape": tuple(shard_info.global_shape),
                "target_tensor_names": tuple(
                    self.source_to_target_names[tensor_name]
                ),
                "file_name": rank_vfile,
                "dtype": str(shard_info.local_tensor.dtype).split(".")[1],
                "dst_rank": None,
                "comm_group": None,
            }
            local_read_plan.append(ExtendReadItem(**common_attrs))
        gathered_plans_per_rank = self.all_gather_fn(
            local_read_plan, **(all_gather_args or {})
        )

        global_read_plan = [
            item for plan in gathered_plans_per_rank for item in plan
        ]

        return self.dedup_read_items(global_read_plan)

    def group_read_items_by_tensor_name(self, global_read_items):
        groups = defaultdict(list)
        for item in global_read_items:
            groups[item.tensor_name].append(item)
        return groups

    def sort_groups_for_early_release(self, groups, source_to_target_names):
        def count_fn(name):
            return len(source_to_target_names.get(name, []))

        sorted_items = sorted(groups.items(), key=lambda x: -count_fn(x[0]))
        return dict(sorted_items)

    def build_reference_map(self, groups: dict[str, set[ExtendReadItem]]):
        ref_map = defaultdict(set)
        for _, items in groups.items():
            for item in items:
                for tgt in item.target_tensor_names:
                    ref_map[tgt].add(item)
        return ref_map

    def _build_read_plan(self, all_gather_args):
        """Creates an optimized, sorted list of read operations."""
        read_items = self.get_read_items(
            all_gather_args=all_gather_args,
        )
        grouped = self.group_read_items_by_tensor_name(read_items)
        grouped = self.sort_groups_for_early_release(
            grouped, self.source_to_target_names
        )
        self.ref_map = self.build_reference_map(grouped)

        self.read_items = [
            item for _, items in grouped.items() for item in items
        ]

    def __iter__(self):
        return self.run()


class SingleCommGroupFullParamAssembler(BaseAssembler):
    """
    Implements the assembly logic from the original full_param function.
    This version handles both single-card and distributed scenarios.
    In the distributed case, it uses a broadcast-based communication strategy.
    """

    def __init__(
        self,
        sharded_state_dict: ShardedStateDict,
        aoa_config: dict[str, list[str]] | None = None,
        process_group: Group | None = None,
        num_splits: int = 1,
        idx: int = 0,
    ):
        super().__init__(sharded_state_dict, aoa_config, num_splits, idx)
        self.process_group = process_group

    def all_gather_fn(self, info, **kwargs):
        process_group = kwargs.get('process_group', self.process_group)
        gathered_info = []
        paddle.distributed.all_gather_object(gathered_info, info, process_group)
        return gathered_info

    def is_identity_mapping(self, shard_mappings):
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

    def prepare(self):
        """Prepare metadata and build the read plan."""
        source_state_shard_info = self.build_global_state_shard_info(
            process_group=self.process_group
        )

        self._prepare_metainfo(source_state_shard_info)

        if self.use_dist:
            self._build_read_plan(
                all_gather_args={"process_group": self.process_group}
            )

    def run(self) -> Generator[tuple[str, paddle.Tensor], None, None]:
        """Main execution generator."""
        self.prepare()
        if not self.use_dist:
            yield from self._run_single_card()
        else:
            yield from self._run_distributed()

    def _run_single_card(
        self,
    ) -> Generator[tuple[str, paddle.Tensor], None, None]:
        """Simple assembly path for a single GPU."""
        for k, v in self.filtered_sharded_state_dict.items():
            assert v.local_shape == v.global_shape, (
                "Single card params must not be sharded.But now the key is {k}, the local_shape is {v.local_shape}, the global_shape is {v.global_shape}."
            )

        for k, shard_mappings in self.destination_sharded_mappings.items():
            if self.is_identity_mapping(shard_mappings):
                src_key = shard_mappings[0].source_slice.key
                yield (
                    k,
                    self.filtered_sharded_state_dict[
                        src_key
                    ].local_tensor.clone(),
                )
            else:
                desc = self.destination_sharded_weight_desc[k]
                cur_sharded_tensor = ShardedWeight(
                    key=desc.key,
                    local_tensor=paddle.empty(
                        desc.local_shape, dtype=desc.dtype
                    ),
                    local_shape=desc.local_shape,
                    global_shape=desc.global_shape,
                    global_offset=desc.global_offset,
                )
                for mapping in shard_mappings:
                    source_tensor = self.filtered_sharded_state_dict[
                        mapping.source_slice.key
                    ]
                    assign_sharded_slice(
                        src_desc=mapping.source_slice,
                        src_shard=source_tensor,
                        dst_desc=mapping.target_slice,
                        dst_shard=cur_sharded_tensor,
                        postprocess_list=mapping.postprocess_list,
                    )
                yield k, cur_sharded_tensor.local_tensor

    def _run_distributed(
        self,
    ) -> Generator[tuple[str, paddle.Tensor], None, None]:
        """Distributed assembly using broadcast and packed buffers."""
        for item in self.read_items:
            cur_src_rank = item.src_rank

            if self.cur_rank == cur_src_rank:
                local_tensor = self.filtered_sharded_state_dict[
                    item.tensor_name
                ].local_tensor.clone()
            else:
                local_tensor = paddle.empty(item.slice_shape, dtype=item.dtype)

            on_cpu = local_tensor.place.is_cpu_place()
            if on_cpu:
                local_tensor = local_tensor.cuda()
            paddle.distributed.broadcast(
                local_tensor, src=cur_src_rank, group=self.process_group
            )
            if on_cpu:
                local_tensor = local_tensor.cpu()

            shard_desc = ShardedWeightDesc(
                key=item.tensor_name,
                local_shape=item.slice_shape,
                global_shape=item.global_shape,
                global_offset=item.src_global_offset,
                dtype=item.dtype,
            )
            self.sharded_desc_to_tensor[shard_desc] = local_tensor

            ready_tensor_names = []
            for name in item.target_tensor_names:
                self.ref_map[name].remove(item)
                if len(self.ref_map[name]) == 0:
                    ready_tensor_names.append(name)
                    del self.ref_map[name]

            yield from self._assemble_and_yield_ready_tensors(
                ready_tensor_names
            )


class OperationType(Enum):
    GLOBAL_BROADCAST = 1
    BROADCAST_ALLGATHER = 2


class HVCommGroupFullParamAssembler(BaseAssembler):
    """
    Implements the assembly logic using a 2D-mesh communication strategy.

    This strategy involves a broadcast along the vertical axis of the process
    mesh, followed by an all-gather along the horizontal axis.
    """

    def __init__(
        self,
        sharded_state_dict: ShardedStateDict,
        horizontal_group: Group,
        vertical_group: Group,
        aoa_config: dict[str, list[str]] | None = None,
        num_splits: int = 1,
        idx: int = 0,
        memory_growth_threshold: int = 8 * (2**30),  # 8GB
    ):
        super().__init__(sharded_state_dict, aoa_config, num_splits, idx)
        self.h_group = horizontal_group
        self.v_group = vertical_group

        self.topology: list[list[int]] = []
        self.vertical_ranks: list[set[int]] = []
        self.horizontal_index: dict[int, int] = {}
        self.vertical_index: dict[int, int] = {}
        self.cur_horizontal_index: int = -1
        self.memory_growth_threshold = memory_growth_threshold

    def all_gather_fn(self, info, **kwargs):
        h_group = kwargs.get('h_group', self.h_group)
        v_group = kwargs.get('v_group', self.v_group)

        h_obj_list = []
        paddle.distributed.all_gather_object(h_obj_list, info, h_group)

        v_obj_list = []
        paddle.distributed.all_gather_object(v_obj_list, h_obj_list, v_group)

        gathered_info = [x for sublist in v_obj_list for x in sublist]
        return gathered_info

    def prepare(self):
        """Build topology, prepare metadata, and build the read plan."""
        assert self.use_dist, (
            "FullParamAssembler only supports distributed training."
        )
        self._build_topology()

        source_state_shard_info = self.build_global_state_shard_info(
            h_group=self.h_group, v_grou=self.v_group
        )
        self._prepare_metainfo(source_state_shard_info)
        self._build_read_plan(
            all_gather_args={'h_group': self.h_group, 'v_group': self.v_group}
        )

    def _build_topology(self):
        h_ranks = []
        paddle.distributed.all_gather_object(
            h_ranks, self.cur_rank, self.h_group
        )
        paddle.distributed.all_gather_object(
            self.topology, h_ranks, self.v_group
        )
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

    def run(self) -> Generator[tuple[str, paddle.Tensor], None, None]:
        """Main execution generator using 2D-mesh communication."""
        self.prepare()

        while len(self.read_items) > 0:
            ready_tensor_names = self._process_one_batch()

            yield from self._assemble_and_yield_ready_tensors(
                ready_tensor_names
            )

    def get_batch_read_items(self):
        read_items = self.read_items
        vertical_ranks = self.vertical_ranks
        horizontal_index = self.horizontal_index

        bathch_read_items = [None] * len(vertical_ranks)
        read_item_index = [None] * len(vertical_ranks)
        cnt = 0
        cur_shape = None
        cur_dtype = None
        for i, item in enumerate(read_items):
            src_rank = item.src_rank
            h_index = horizontal_index[src_rank]
            if bathch_read_items[h_index] is None and cnt == 0:
                bathch_read_items[h_index] = item
                read_item_index[h_index] = i
                cnt += 1
                cur_dtype = item.dtype
                cur_shape = item.slice_shape
                element_size = paddle.core.size_of_dtype(
                    getattr(paddle, cur_dtype)
                )
                memory_growth = (
                    element_size * math.prod(cur_shape) * len(vertical_ranks)
                )
                if memory_growth > self.memory_growth_threshold:
                    return (
                        bathch_read_items,
                        read_item_index,
                        OperationType.GLOBAL_BROADCAST,
                    )
                if cnt == len(vertical_ranks):
                    return (
                        bathch_read_items,
                        read_item_index,
                        OperationType.GLOBAL_BROADCAST,
                    )

            if bathch_read_items[h_index] is None and cnt != 0:
                if item.slice_shape == cur_shape and item.dtype == cur_dtype:
                    bathch_read_items[h_index] = item
                    read_item_index[h_index] = i
                    cnt += 1
                    if cnt == len(vertical_ranks):
                        return (
                            bathch_read_items,
                            read_item_index,
                            OperationType.BROADCAST_ALLGATHER,
                        )

        assert cur_shape is not None
        assert cur_dtype is not None

        for i, item in enumerate(bathch_read_items):
            if item is None:
                src_rank = min(vertical_ranks[i])
                common_attrs = {
                    "tensor_name": INTERNAL_PADDING_TENSOR_NAME,
                    "src_rank": src_rank,
                    "src_global_offset": (0,) * len(cur_shape),
                    "dst_global_offset": (0,) * len(cur_shape),
                    "src_local_offset": (0,) * len(cur_shape),
                    "dst_local_offset": (0,) * len(cur_shape),
                    "slice_shape": cur_shape,
                    "global_shape": cur_shape,
                    "target_tensor_names": None,
                    "file_name": "padding_vfile",
                    "dtype": cur_dtype,
                    "comm_group": None,
                }

                padding_read_item = ExtendReadItem(
                    dst_rank=None, **common_attrs
                )
                bathch_read_items[i] = padding_read_item

        return (
            bathch_read_items,
            read_item_index,
            OperationType.BROADCAST_ALLGATHER,
        )

    def _process_one_batch(self) -> list[str]:
        """Performs V-Broadcast + H-AllGather for one batch of items."""

        batch_items, batch_indices, op_type = self.get_batch_read_items()

        if op_type == OperationType.BROADCAST_ALLGATHER:
            read_item = batch_items[self.cur_horizontal_index]
        else:
            values = [x for x in batch_items if x is not None]
            if len(values) == 1:
                read_item = values[0]
            else:
                raise ValueError(
                    "When the comm op is GLOBAL_BROADCAST, read_items should be of length 1!"
                )
            batch_items = [read_item]

        if self.cur_rank == read_item.src_rank:
            buffer = (
                paddle.empty(read_item.slice_shape, read_item.dtype)
                if read_item.tensor_name == INTERNAL_PADDING_TENSOR_NAME
                else self.filtered_sharded_state_dict[
                    read_item.tensor_name
                ].local_tensor.clone()
            )
        else:
            buffer = paddle.empty(read_item.slice_shape, dtype=read_item.dtype)

        if op_type == OperationType.BROADCAST_ALLGATHER:
            paddle.distributed.broadcast(
                buffer, src=read_item.src_rank, group=self.v_group
            )
            tensor_list = []
            paddle.distributed.all_gather(
                tensor_list, buffer, group=self.h_group
            )
        else:
            src_rank = read_item.src_rank
            v_ranks = sorted(
                self.vertical_ranks[self.horizontal_index[src_rank]]
            )
            if self.cur_rank in v_ranks:
                paddle.distributed.broadcast(
                    buffer, src=src_rank, group=self.v_group
                )
            src_rank = v_ranks[self.vertical_index[self.cur_rank]]
            paddle.distributed.broadcast(
                buffer, src=src_rank, group=self.h_group
            )
            tensor_list = [buffer]

        for idx, item in enumerate(batch_items):
            if item.tensor_name != INTERNAL_PADDING_TENSOR_NAME:
                shard_desc = ShardedWeightDesc(
                    key=item.tensor_name,
                    local_shape=item.slice_shape,
                    global_shape=item.global_shape,
                    global_offset=item.src_global_offset,
                    dtype=item.dtype,
                )
                self.sharded_desc_to_tensor[shard_desc] = tensor_list[idx]

        ready_tensor_names = []
        for item in batch_items:
            if item.target_tensor_names:
                for name in item.target_tensor_names:
                    self.ref_map[name].remove(item)
                    if not self.ref_map[name]:
                        ready_tensor_names.append(name)
                        del self.ref_map[name]

        for index in sorted(
            [i for i in batch_indices if i is not None], reverse=True
        ):
            del self.read_items[index]

        return ready_tensor_names


@paddle.no_grad()
def full_param(
    sharded_state_dict: ShardedStateDict,
    aoa_config: dict[str, list[str]] | None = None,
    **kwargs,
):
    h_group = kwargs.pop("h_group", None)
    v_group = kwargs.pop("v_group", None)
    process_group = kwargs.pop("process_group", None)
    num_splits = kwargs.pop("num_splits", 1)
    memory_growth_threshold = kwargs.pop("memory_growth_threshold", 8 * (2**30))
    idx = kwargs.pop("shard_idx", 0)
    assert (h_group and v_group) or not (h_group or v_group), (
        "Both horizontal and vertical groups must be provided when using FullParamAssembler."
    )
    if h_group and v_group:
        return HVCommGroupFullParamAssembler(
            sharded_state_dict,
            h_group,
            v_group,
            aoa_config,
            num_splits,
            idx,
            memory_growth_threshold,
        )
    else:
        return SingleCommGroupFullParamAssembler(
            sharded_state_dict, aoa_config, process_group
        )
