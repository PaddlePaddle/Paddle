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

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .lexer import Lexer
from .parser import Parser

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class ShardedWeightDesc:
    key: str
    local_shape: tuple[int, ...]
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]


_ShardInfo = dict[str, list[ShardedWeightDesc]]

SliceRef = tuple[str, tuple[slice, ...], tuple[slice, ...]]


class TensorDesc:
    def __init__(self, slices: list[SliceRef], shape: tuple[int]):
        self.slices = slices
        self.shape = shape

    def __repr__(self):
        s = []
        for key, sl_src, sl_dst in self.slices:
            s.append(f"{key}{sl_src} -> self{sl_dst}")
        return f"Tensor(shape={self.shape}, slices={s})"


@dataclass(frozen=True)
class ShardMappingEntry:
    target_slice: ShardedWeightDesc
    source_slice: ShardedWeightDesc
    postprocess_list: list[str] | None = None


ShardMapping = list[ShardMappingEntry]


class AoAShardInfoContext:
    def __init__(
        self,
        source_state_shard_info: _ShardInfo,
        destination_state_shard_info: _ShardInfo,
    ) -> None:
        self.source_state_shard_info = source_state_shard_info
        self.destination_state_shard_info = destination_state_shard_info

    def get_all_dst_state_keys(self) -> Iterable[str]:
        return self.destination_state_shard_info.keys()

    def get_all_src_state_keys(self) -> Iterable[str]:
        return self.source_state_shard_info.keys()

    def get_num_hidden_layers(
        self, name_with_layer_id: str, layer_id_macro_tag: str
    ) -> int:
        if layer_id_macro_tag not in name_with_layer_id:
            raise ValueError(
                f"layer_id_macro_tag '{layer_id_macro_tag}' not in name_with_layer_id '{name_with_layer_id}'"
            )
        prefix, suffix = name_with_layer_id.split(layer_id_macro_tag, 1)
        pattern = re.compile(rf"{re.escape(prefix)}(\d+){re.escape(suffix)}")
        max_layer = 0
        for key in self.get_all_dst_state_keys():
            match = pattern.fullmatch(key)
            if match:
                layer_num = int(match.group(1))
                max_layer = max(max_layer, layer_num)
        return max_layer + 1

    def get_src_state_shard_num(self, src_state_key: str) -> int:
        if src_state_key not in self.source_state_shard_info:
            raise KeyError(
                f"src_state_key '{src_state_key}' not in  source_state_shard_info"
            )
        return len(self.source_state_shard_info[src_state_key])

    def get_dst_state_shard_num(self, dst_state_key: str) -> int:
        if dst_state_key not in self.destination_state_shard_info:
            raise KeyError(
                f"dst_state_key '{dst_state_key}' not in destination_state_shard_info"
            )
        return len(self.destination_state_shard_info[dst_state_key])


class AoAEngine:
    def __init__(
        self,
        aoa_config: dict[str, list[str]],
        source_state_shard_info: _ShardInfo,
        destination_state_shard_info: _ShardInfo,
    ):
        self.aoa_config = aoa_config
        self.source_state_shard_info = source_state_shard_info
        self.destination_state_shard_info = destination_state_shard_info
        self.context = AoAShardInfoContext(
            source_state_shard_info, destination_state_shard_info
        )
        self.lexer = Lexer(self.context)
        self.parser = Parser(
            self.lexer.all_tokens(self.aoa_config["aoa_statements"])
        )
        self.statements = self.parser.parse_program()
        self.input_vars = self.build_input_vars()
        self.output_vars = {}
        self.need_remove_input_vars = set()
        self.need_remove_output_vars = set()
        self.need_transpose_output_vars = set()
        self.need_transpose_input_vars = {}

        self.shape_propagation()

    def make_input_tensor(self, key: str, shape: tuple[int]) -> TensorDesc:
        base_slice = tuple([slice(0, s) for s in shape])
        return TensorDesc([(key, base_slice, base_slice)], shape)

    def build_input_vars(self):
        input_vars = {}
        for key, shards in self.source_state_shard_info.items():
            global_shape = shards[0].global_shape
            input_vars[key] = self.make_input_tensor(key, global_shape)
        return input_vars

    def split(
        self, tensor: TensorDesc, axis: int, sizes: list[int]
    ) -> list[TensorDesc]:
        results = []
        start = 0
        for sz in sizes:
            sub_dst_slice = [slice(None)] * len(tensor.shape)
            sub_dst_slice[axis] = slice(0, sz)
            sub_slices = []
            for aidx, src_sl, dst_sl in tensor.slices:
                dst_start = (
                    dst_sl[axis].start if dst_sl[axis].start is not None else 0
                )
                dst_stop = (
                    dst_sl[axis].stop
                    if dst_sl[axis].stop is not None
                    else tensor.shape[axis]
                )
                inter_begin = max(start, dst_start)
                inter_end = min(start + sz, dst_stop)
                if inter_begin < inter_end:
                    src_axis_start = (
                        src_sl[axis].start
                        if src_sl[axis].start is not None
                        else 0
                    )
                    sub_src_sl = list(src_sl)
                    sub_dst_sl = list(dst_sl)
                    offset = inter_begin - dst_start
                    length = inter_end - inter_begin
                    sub_src_sl[axis] = slice(
                        src_axis_start + offset,
                        src_axis_start + offset + length,
                    )
                    sub_dst_sl[axis] = slice(
                        inter_begin - start, inter_begin - start + length
                    )
                    sub_slices.append(
                        (aidx, tuple(sub_src_sl), tuple(sub_dst_sl))
                    )
            new_shape = list(tensor.shape)
            new_shape[axis] = sz
            results.append(TensorDesc(sub_slices, tuple(new_shape)))
            start += sz
        return results

    def concat(self, tensors: list[TensorDesc], axis: int) -> TensorDesc:
        slices = []
        shape = list(tensors[0].shape)
        shape[axis] = sum(t.shape[axis] for t in tensors)
        curr = 0
        for t in tensors:
            for aidx, src_sl, dst_sl in t.slices:
                new_dst_sl = list(dst_sl)
                dst_start = (
                    dst_sl[axis].start if dst_sl[axis].start is not None else 0
                )
                dst_stop = (
                    dst_sl[axis].stop
                    if dst_sl[axis].stop is not None
                    else t.shape[axis]
                )
                length = dst_stop - dst_start
                new_dst_sl[axis] = slice(
                    dst_start + curr, dst_start + curr + length
                )
                slices.append((aidx, src_sl, tuple(new_dst_sl)))
            curr += t.shape[axis]
        return TensorDesc(slices, tuple(shape))

    def transpose(self, tensor: TensorDesc) -> TensorDesc:
        raise NotImplementedError

    def cast(self, tensor: TensorDesc) -> TensorDesc:
        raise NotImplementedError

    def shape_propagation(self):
        intermediate_vars = {}

        def _get_var_ref(var):
            if var.name in intermediate_vars:
                return intermediate_vars[var.name]
            elif var.name in self.input_vars:
                return self.input_vars[var.name]
            else:
                raise ValueError(f"{var.name} should be assigned before!")

        for stmt in self.statements:
            left_vars = stmt.left_vars
            right_vars = stmt.right_vars
            attrs = stmt.attrs

            if len(left_vars) > 1 or len(right_vars) > 1:
                if not (len(attrs) == 1 and attrs[0].key == "axis"):
                    raise ValueError(
                        "When split/concat, only support one attr named `axis`"
                    )
                axis = attrs[0].value

                if len(left_vars) == 1:
                    in_name = left_vars[0].name
                    in_ref = _get_var_ref(left_vars[0])
                    assert in_ref.shape[axis] % len(right_vars) == 0
                    sizes = [
                        in_ref.shape[axis] // len(right_vars)
                        for var in right_vars
                    ]
                    result = self.split(in_ref, axis, sizes)
                    for out_var, out_ref in zip(right_vars, result):
                        intermediate_vars[out_var.name] = out_ref
                        if (
                            out_var.name
                            in self.context.get_all_dst_state_keys()
                        ):
                            self.output_vars[out_var.name] = out_ref

                elif len(right_vars) == 1:
                    left_refs = [_get_var_ref(var) for var in left_vars]
                    result = self.concat(left_refs, axis)
                    out_name = right_vars[0].name
                    intermediate_vars[out_name] = result
                    if out_name in self.context.get_all_dst_state_keys():
                        self.output_vars[out_name] = result

                else:
                    raise SyntaxError(
                        f'Unexpected split/concat statement: {stmt}'
                    )

            elif len(left_vars) == 1 and len(right_vars) == 1:
                lvar, rvar = left_vars[0], right_vars[0]
                if rvar.name == "_":
                    self.need_remove_input_vars.add(lvar.name)
                elif lvar.name == "_":
                    self.need_remove_output_vars.add(rvar.name)
                else:
                    for attr in attrs:
                        if attr.key == "transpose":
                            raise NotImplementedError
                        elif attr.key == "dtype":
                            raise NotImplementedError
                        else:
                            raise ValueError(f"Unsupported attribute: {attr}")
                    intermediate_vars[lvar.name] = _get_var_ref(rvar)
                    if lvar.name in self.destination_vars:
                        self.output_vars[lvar.name] = intermediate_vars[
                            lvar.name
                        ]
            else:
                raise SyntaxError(f'Unexpected statement: {stmt}')

        for name in self.destination_state_shard_info.keys():
            if name not in self.output_vars:
                assert name in self.input_vars
                self.output_vars[name] = self.input_vars[name]

    def find_source_slices(
        self, key: str, local_slice: tuple[slice, ...]
    ) -> list[SliceRef]:
        assert key in self.output_vars
        tensor = self.output_vars[key]
        results = []
        assert len(local_slice) == len(tensor.shape)
        ndim = len(tensor.shape)

        def slice_intersect(a: slice, b: slice, dim_len: int):
            a_start, a_stop, a_step = a.indices(dim_len)
            b_start, b_stop, b_step = b.indices(dim_len)
            if a_step != 1 or b_step != 1:
                raise NotImplementedError("Only support step size of 1")
            start = max(a_start, b_start)
            stop = min(a_stop, b_stop)
            if start >= stop:
                return None
            return slice(start, stop, 1)

        for src_key, sl_src, sl_dst in tensor.slices:
            intersection = []
            for i in range(ndim):
                inter = slice_intersect(
                    local_slice[i], sl_dst[i], tensor.shape[i]
                )
                if inter is None:
                    break
                intersection.append(inter)
            else:
                # Compute corresponding src_slice for the intersection
                src_slice = []
                for i in range(ndim):
                    dst = sl_dst[i]
                    src = sl_src[i]
                    dim_len = tensor.shape[i]
                    dst_start, _, _ = dst.indices(dim_len)
                    src_start, _, _ = src.indices(dim_len)
                    inter_start, inter_stop, _ = intersection[i].indices(
                        dim_len
                    )
                    offset = inter_start - dst_start
                    src_inter_start = src_start + offset
                    src_inter_stop = src_inter_start + (
                        inter_stop - inter_start
                    )
                    src_slice.append(slice(src_inter_start, src_inter_stop, 1))
                results.append((src_key, tuple(src_slice), tuple(intersection)))
        return results

    def find_shard_sources(
        self,
        target: ShardedWeightDesc,
    ) -> ShardMapping:
        target_key = target.key
        target_local_shape = target.local_shape
        target_global_offset = target.global_offset
        target_global_shape = target.global_shape

        slices = tuple(
            slice(offset, offset + size, 1)
            for offset, size in zip(target_global_offset, target_local_shape)
        )

        results = self.find_source_slices(target_key, slices)

        shard_mappings = []

        for src_key, src_slices, local_slices in results:
            src_var = self.input_vars[src_key]
            src_global_shape = src_var.shape

            src_local_shape = tuple(slc.stop - slc.start for slc in src_slices)
            src_global_offset = tuple(slc.start for slc in src_slices)

            tgt_local_shape = tuple(
                slc.stop - slc.start for slc in local_slices
            )
            tgt_global_offset = tuple(slc.start for slc in local_slices)

            source_sharded_weight = ShardedWeightDesc(
                src_key, src_local_shape, src_global_shape, src_global_offset
            )
            target_sharded_weight = ShardedWeightDesc(
                target_key,
                tgt_local_shape,
                target_global_shape,
                tgt_global_offset,
            )

            postprocess_list = []

            shard_mappings.append(
                ShardMappingEntry(
                    target_sharded_weight,
                    source_sharded_weight,
                    postprocess_list,
                )
            )
        return shard_mappings
