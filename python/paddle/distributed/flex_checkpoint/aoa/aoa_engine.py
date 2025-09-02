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

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..dcp.sharded_weight import ShardedWeightDesc
from .lexer import Lexer
from .parser import Parser

if TYPE_CHECKING:
    from collections.abc import Iterable

_ShardInfo = dict[str, list[ShardedWeightDesc]]

# SliceRef := (key, src_slice, dst_slice, postprocess_list)
SliceRef = tuple[str, tuple[slice, ...], tuple[slice, ...], Optional[list[str]]]


class TensorDesc:
    def __init__(self, slices: list[SliceRef], shape: tuple[int]):
        self.slices = slices
        self.shape = shape

    def __repr__(self):
        s = []
        for key, sl_src, sl_dst, pp_list in self.slices:
            s.append(
                f"{key}{sl_src} -> self{sl_dst}, postprocess_list={pp_list}"
            )
        return f"Tensor(shape={self.shape}, slices={s})"


@dataclass(frozen=True)
class ShardMappingEntry:
    target_slice: ShardedWeightDesc
    source_slice: ShardedWeightDesc
    postprocess_list: list[str] | None = None


ShardMapping = list[ShardMappingEntry]


class AOAShardInfoContext:
    def __init__(
        self,
        source_state_shard_info: _ShardInfo,
        destination_state_shard_info: _ShardInfo,
    ) -> None:
        self.source_state_shard_info = source_state_shard_info
        self.destination_state_shard_info = destination_state_shard_info
        self.optim_state_name = [
            ".w_0",
            ".moment1_0",
            ".moment2_0",
            ".beta1_pow_acc_0",
            ".beta2_pow_acc_0",
        ]

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
        new_state_key = src_state_key
        for state_name in self.optim_state_name:
            if state_name in src_state_key:
                new_state_key = src_state_key.replace(state_name, "")
                break

        return len(self.source_state_shard_info[new_state_key])

    def get_dst_state_shard_num(self, dst_state_key: str) -> int:
        if dst_state_key not in self.destination_state_shard_info:
            raise KeyError(
                f"dst_state_key '{dst_state_key}' not in destination_state_shard_info"
            )

        new_state_key = dst_state_key
        for state_name in self.optim_state_name:
            if state_name in dst_state_key:
                new_state_key = dst_state_key.replace(state_name, "")
                break

        shard_infos = self.destination_state_shard_info[new_state_key]
        global_offset_set = set()
        for shard_info in shard_infos:
            global_offset_set.add(shard_info.global_offset)

        return len(global_offset_set)


class AOAEngine:
    def __init__(
        self,
        aoa_config: dict[str, list[str]],
        source_state_shard_info: _ShardInfo,
        destination_state_shard_info: _ShardInfo,
    ):
        self.aoa_config = aoa_config
        self.source_state_shard_info = source_state_shard_info
        self.destination_state_shard_info = destination_state_shard_info
        self.context = AOAShardInfoContext(
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
        self.need_add_output_vars = set()

        self.shape_propagation()

    def make_input_tensor(self, key: str, shape: tuple[int]) -> TensorDesc:
        base_slice = tuple([slice(0, s) for s in shape])
        return TensorDesc([(key, base_slice, base_slice, None)], shape)

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
            for aidx, src_sl, dst_sl, pp_list in tensor.slices:
                if pp_list is not None:
                    src_sl = postprocess_transpose(list(src_sl), pp_list)

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
                    if pp_list is not None:
                        sub_src_sl = postprocess_transpose(
                            list(sub_src_sl), pp_list, reverse=True
                        )
                        sub_slices.append(
                            (
                                aidx,
                                tuple(sub_src_sl),
                                tuple(sub_dst_sl),
                                pp_list.copy(),
                            )
                        )
                    else:
                        sub_slices.append(
                            (aidx, tuple(sub_src_sl), tuple(sub_dst_sl), None)
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
            for aidx, src_sl, dst_sl, pp_list in t.slices:
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
                if pp_list is not None:
                    slices.append(
                        (aidx, src_sl, tuple(new_dst_sl), pp_list.copy())
                    )
                else:
                    slices.append((aidx, src_sl, tuple(new_dst_sl), None))
            curr += t.shape[axis]
        return TensorDesc(slices, tuple(shape))

    def transpose(self, tensor: TensorDesc, permutation: str) -> TensorDesc:
        slices = []
        tensor_shape = transpose_list(
            tensor.shape, ast.literal_eval(permutation)
        )
        for aidx, src_sl, dst_sl, pp_list in tensor.slices:
            trans_dst_sl = transpose_list(dst_sl, ast.literal_eval(permutation))
            if pp_list is not None:
                new_pp_list = pp_list.copy()
                new_pp_list.append(permutation)
                slices.append((aidx, src_sl, trans_dst_sl, new_pp_list))
            else:
                slices.append((aidx, src_sl, trans_dst_sl, [permutation]))
        return TensorDesc(slices, tensor_shape)

    def cast(self, tensor: TensorDesc, dtype: str) -> TensorDesc:
        slices = []
        for aidx, src_sl, dst_sl, pp_list in tensor.slices:
            if pp_list is not None:
                new_pp_list = pp_list.copy()
                new_pp_list.append(dtype)
                slices.append((aidx, src_sl, dst_sl, new_pp_list))
            else:
                slices.append((aidx, src_sl, dst_sl, [dtype]))
        return TensorDesc(slices, tensor.shape)

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
                    self.need_add_output_vars.add(rvar.name)
                else:
                    if len(attrs) > 0:
                        for attr in attrs:
                            in_ref = _get_var_ref(lvar)
                            if attr.key == "permute":
                                if attr.value == "[]":
                                    ndim = len(in_ref.shape)
                                    perm = str(list(range(ndim - 1, -1, -1)))
                                else:
                                    perm = attr.value
                                result = self.transpose(in_ref, perm)
                            elif attr.key == "dtype":
                                result = self.cast(in_ref, attr.value)
                            elif attr.key == "axis":
                                pass
                            else:
                                raise ValueError(
                                    f"Unsupported attribute: {attr}"
                                )

                            intermediate_vars[rvar.name] = result
                            if (
                                rvar.name
                                in self.context.get_all_dst_state_keys()
                            ):
                                self.output_vars[rvar.name] = result
                    else:
                        in_ref = _get_var_ref(lvar)
                        intermediate_vars[rvar.name] = in_ref
                        if rvar.name in self.context.get_all_dst_state_keys():
                            self.output_vars[rvar.name] = in_ref

            else:
                raise SyntaxError(f'Unexpected statement: {stmt}')

        for name in self.destination_state_shard_info.keys():
            if name not in self.output_vars:
                if name in self.need_add_output_vars:
                    self.output_vars[name] = None
                else:
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

        def slice_intersect(a: slice, b: slice):
            start = max(a.start, b.start)
            stop = min(a.stop, b.stop)
            if start >= stop:
                return None
            return slice(start, stop, 1)

        for src_key, sl_src, sl_dst, pp_list in tensor.slices:
            intersection = []
            for i in range(ndim):
                inter = slice_intersect(local_slice[i], sl_dst[i])
                if inter is None:
                    break
                intersection.append(inter)
            else:
                # Compute corresponding src_slice for the intersection
                if pp_list is not None:
                    sl_src = postprocess_transpose(list(sl_src), pp_list)
                src_slice = []
                for i in range(ndim):
                    dst = sl_dst[i]
                    src = sl_src[i]
                    dst_start = dst.start
                    src_start = src.start
                    inter_start, inter_stop = (
                        intersection[i].start,
                        intersection[i].stop,
                    )
                    offset = inter_start - dst_start
                    src_inter_start = src_start + offset
                    src_inter_stop = src_inter_start + (
                        inter_stop - inter_start
                    )
                    src_slice.append(slice(src_inter_start, src_inter_stop, 1))
                if pp_list is not None:
                    src_slice = postprocess_transpose(
                        list(src_slice), pp_list, reverse=True
                    )
                    results.append(
                        (
                            src_key,
                            tuple(src_slice),
                            tuple(intersection),
                            pp_list.copy(),
                        ),
                    )
                else:
                    results.append(
                        (src_key, tuple(src_slice), tuple(intersection), None)
                    )
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

        for src_key, src_slices, local_slices, pp_list in results:
            src_var = self.input_vars[src_key]
            src_global_shape = src_var.shape

            src_local_shape = tuple(slc.stop - slc.start for slc in src_slices)
            src_global_offset = tuple(slc.start for slc in src_slices)

            tgt_local_shape = tuple(
                slc.stop - slc.start for slc in local_slices
            )
            tgt_global_offset = tuple(slc.start for slc in local_slices)

            source_sharded_weight = ShardedWeightDesc(
                src_key,
                src_local_shape,
                tuple(src_global_shape),
                src_global_offset,
            )
            target_sharded_weight = ShardedWeightDesc(
                target_key,
                tgt_local_shape,
                tuple(target_global_shape),
                tgt_global_offset,
            )

            if source_sharded_weight.key in self.need_remove_input_vars:
                mapping_entry = ShardMappingEntry(
                    target_sharded_weight,
                    source_sharded_weight,
                    [],
                )
                continue

            shard_mappings.append(
                ShardMappingEntry(
                    target_sharded_weight,
                    source_sharded_weight,
                    pp_list,
                )
            )
        return shard_mappings


def postprocess_transpose(
    li: list[tuple[slice, ...]] | tuple[tuple[slice, ...]],
    postprocess_list: list[str],
    reverse: bool = False,
) -> list[tuple[slice, ...]] | tuple[tuple[slice, ...]]:
    result = li
    if reverse:
        for pp in list(reversed(postprocess_list)):
            if pp.startswith("["):
                reversed_transpose = np.argsort(ast.literal_eval(pp)).tolist()
                result = transpose_list(result, reversed_transpose)
    else:
        for pp in postprocess_list:
            if pp.startswith("["):
                result = transpose_list(result, ast.literal_eval(pp))
    return result


def transpose_list(
    li: list[tuple[slice, ...]] | tuple[tuple[slice, ...]],
    permutation: list[int],
) -> list[tuple[slice, ...]] | tuple[tuple[slice, ...]]:
    trans_list = []
    for idx in permutation:
        trans_list.append(li[idx])
    if isinstance(li, tuple):
        return tuple(trans_list)
    else:
        return trans_list
