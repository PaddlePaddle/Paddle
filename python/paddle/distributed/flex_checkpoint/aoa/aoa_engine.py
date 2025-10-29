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
import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

from ..dcp.sharded_weight import ShardedWeightDesc
from .lexer import Lexer
from .parser import Parser

_ShardInfo = dict[str, list[ShardedWeightDesc]]

# SliceRef := (key, src_slice, dst_slice, postprocess_list)
SliceRef = tuple[str, tuple[slice, ...], tuple[slice, ...], Optional[list[str]]]


class TensorDesc:
    def __init__(
        self,
        slices: list[SliceRef],
        shape: tuple[int],
        in_degree: int = 0,
        out_degree: int = 0,
        dtype: str | None = None,
    ):
        self.slices = slices
        self.shape = shape
        self.in_degree = in_degree
        self.out_degree = out_degree
        self.dtype = dtype

    def __repr__(self):
        s = []
        for key, sl_src, sl_dst, pp_list in self.slices:
            s.append(
                f"{key}{sl_src} -> self{sl_dst}, postprocess_list={pp_list}"
            )
        return f"Tensor(shape={self.shape}, slices={s}, in_degree={self.in_degree}, out_degree={self.out_degree}, dtype={self.dtype})"


@dataclass(frozen=True)
class ShardMappingEntry:
    target_slice: ShardedWeightDesc
    source_slice: ShardedWeightDesc
    postprocess_list: list[str] | None = None


ShardMapping = list[ShardMappingEntry]

OPTIMIZER_STATE_NAME = [
    ".w_0",
    ".moment1_0",
    ".moment2_0",
    ".beta1_pow_acc_0",
    ".beta2_pow_acc_0",
]


def split_optimizer_state_key(key: str) -> tuple[str, str]:
    for opt_state_name in OPTIMIZER_STATE_NAME:
        if key.endswith(opt_state_name):
            return key[: -len(opt_state_name)], opt_state_name
    return key, None


class AOAShardInfoContext:
    def __init__(
        self,
        source_state_shard_info: _ShardInfo,
        destination_state_shard_info: _ShardInfo,
    ) -> None:
        self.source_state_shard_info = source_state_shard_info
        self.destination_state_shard_info = destination_state_shard_info
        self.left_var_to_right_var_mapping = {}
        self.right_var_from_left_var_mapping = {}

    def get_all_dst_state_keys(self):
        dst_state_keys = set()
        if self.destination_state_shard_info is None:
            return dst_state_keys
        for k in self.destination_state_shard_info.keys():
            model_state_key, _ = split_optimizer_state_key(k)
            dst_state_keys.add(model_state_key)
        return dst_state_keys

    def get_all_src_state_keys(self):
        src_state_keys = set()
        for k in self.source_state_shard_info.keys():
            model_state_key, _ = split_optimizer_state_key(k)
            src_state_keys.add(model_state_key)
        return src_state_keys

    def get_num_hidden_layers(
        self,
        name_with_layer_id: str,
        layer_id_macro_tag: str,
    ) -> int:
        if layer_id_macro_tag not in name_with_layer_id:
            raise ValueError(
                f"layer_id_macro_tag '{layer_id_macro_tag}' not in name_with_layer_id '{name_with_layer_id}'"
            )
        prefix, suffix = name_with_layer_id.split(layer_id_macro_tag, 1)
        pattern = re.compile(rf"{re.escape(prefix)}(\d+){re.escape(suffix)}")
        match_layer_id = set()
        for key in self.get_all_src_state_keys():
            match = pattern.fullmatch(key)
            if match:
                layer_num = int(match.group(1))
                match_layer_id.add(layer_num)
        return match_layer_id

    def get_src_state_shard_num(self, src_state_key: str) -> int:
        model_state_key, opt_state_name = split_optimizer_state_key(
            src_state_key
        )

        assert opt_state_name is None, (
            "AOA notions apply only to the model state, but are automatically propagated to the optimizer state."
        )

        # Only need to parse the model state key for optimizer state shard num, because the optimizer state slice info is completely consistent with the model state slice info.
        resolved_model_state_key = self.resolve_mapping_chain(
            model_state_key, reverse=True
        )

        state_keys = [
            resolved_model_state_key,
            f"{resolved_model_state_key}.w_0",
            f"{resolved_model_state_key}.moment1_0",
            f"{resolved_model_state_key}.moment2_0",
        ]

        shard_nums = {
            len(
                {
                    shard_info.global_offset
                    for shard_info in self.source_state_shard_info[key]
                }
            )
            for key in state_keys
            if key in self.source_state_shard_info
        }

        if not shard_nums:
            logger.warning(
                f"No shard information found for any of the keys: {state_keys}, return 1."
            )
            return 1
        if len(shard_nums) > 1:
            raise AssertionError(
                f"Inconsistent shard numbers among keys in source_sharded_state_dict: {shard_nums}."
            )
        return shard_nums.pop()

    def get_dst_state_shard_num(self, dst_state_key: str) -> int:
        if self.destination_state_shard_info is None:
            # Default `dst_state_shard_num=1` if `destination_state_shard_info` is missing.
            return 1
        model_state_key, opt_state_name = split_optimizer_state_key(
            dst_state_key
        )

        assert opt_state_name is None, (
            "AOA notions apply only to the model state, but are automatically propagated to the optimizer state."
        )

        # Only need to parse the model state key for optimizer state shard num, because the optimizer state slice info is completely consistent with the model state slice info.
        resolved_model_state_key = self.resolve_mapping_chain(
            model_state_key, reverse=False
        )

        state_keys = [
            resolved_model_state_key,
            f"{resolved_model_state_key}.w_0",
            f"{resolved_model_state_key}.moment1_0",
            f"{resolved_model_state_key}.moment2_0",
        ]

        shard_nums = {
            len(
                {
                    shard_info.global_offset
                    for shard_info in self.destination_state_shard_info[key]
                }
            )
            for key in state_keys
            if key in self.destination_state_shard_info
        }

        if not shard_nums:
            logger.warning(
                f"No shard information found for any of the keys: {state_keys}, return 1."
            )
            return 1
        if len(shard_nums) > 1:
            raise AssertionError(
                f"Inconsistent shard numbers among keys in destination_state_shard_info: {shard_nums}."
            )
        return shard_nums.pop()

    def resolve_mapping_chain(self, key: str, reverse: bool = False) -> str:
        """
        Recursively resolve the mapping chain, find the final leaf node

        Args:
            key: The key to be resolved
            reverse: False use left_var_to_right_var_mapping，True use right_var_from_left_var_mapping

        For example:
        - reverse=False: temp_var -> dst_key
        - reverse=True: temp_var -> src_key
        """
        visited = set()  # avoid infinite loop
        current_key = key

        if reverse:
            mapping_dict = self.right_var_from_left_var_mapping
        else:
            mapping_dict = self.left_var_to_right_var_mapping

        while current_key in mapping_dict:
            assert current_key not in visited, (
                "Infinite loop detected in resolve_mapping_chain,which means the start key is not src_key or the end key is not dst_key, the aoa_config is error"
            )
            visited.add(current_key)
            if reverse and current_key in self.get_all_src_state_keys():
                break
            elif not reverse and current_key in self.get_all_dst_state_keys():
                break

            mapped_vars = mapping_dict[current_key]
            if mapped_vars and len(mapped_vars) > 0:
                current_key = mapped_vars[0]
            else:
                break

        return current_key


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
            source_state_shard_info,
            destination_state_shard_info,
        )
        self.lexer = Lexer(self.context)
        self.parser = Parser(
            self.lexer.all_tokens(self.aoa_config.get("aoa_statements", []))
        )
        self.statements = self.parser.parse_program()
        self.input_vars = self.build_input_vars()
        self.output_vars = {}
        self.intermediate_vars = {}
        self.need_remove_input_vars = set()
        self.need_add_output_vars = set()

        self.shape_propagation()

    def make_input_tensor(
        self, key: str, shape: tuple[int], dtype: str
    ) -> TensorDesc:
        base_slice = tuple([slice(0, s) for s in shape])
        return TensorDesc(
            [(key, base_slice, base_slice, None)],
            shape,
            in_degree=0,
            out_degree=0,
            dtype=dtype,
        )

    def build_input_vars(self):
        input_vars = {}
        dtype = None
        for key, shards in sorted(self.source_state_shard_info.items()):
            global_shape = shards[0].global_shape
            model_state_key, opt_state_name = split_optimizer_state_key(key)
            if opt_state_name is None:
                dtype = shards[0].dtype
            if model_state_key in input_vars.keys() or opt_state_name in [
                ".beta1_pow_acc_0",
                ".beta2_pow_acc_0",
            ]:
                continue
            input_vars[model_state_key] = self.make_input_tensor(
                model_state_key, global_shape, dtype
            )
        return input_vars

    def split(
        self, tensor: TensorDesc, axis: int, sizes: list[int]
    ) -> list[TensorDesc]:
        results = []
        start = 0
        tensor.out_degree += len(sizes)
        dtype = tensor.dtype
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
            results.append(
                TensorDesc(
                    sub_slices,
                    tuple(new_shape),
                    in_degree=1,
                    out_degree=0,
                    dtype=dtype,
                )
            )
            start += sz
        return results

    def concat(self, tensors: list[TensorDesc], axis: int) -> TensorDesc:
        slices = []
        assert len(tensors) >= 1, (
            "When concatenating multiple tensors, there should be at least one!"
        )
        shape = list(tensors[0].shape)
        shape[axis] = sum(t.shape[axis] for t in tensors)
        dtype = tensors[0].dtype
        assert all(t.dtype == dtype for t in tensors), (
            "All tensors must have the same dtype!"
        )
        curr = 0
        for t in tensors:
            t.out_degree += 1
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
        return TensorDesc(
            slices,
            tuple(shape),
            in_degree=len(tensors),
            out_degree=0,
            dtype=dtype,
        )

    def transpose(self, tensor: TensorDesc, permutation: str) -> TensorDesc:
        slices = []
        tensor.out_degree += 1
        tensor_shape = transpose_list(
            tensor.shape, ast.literal_eval(permutation)
        )
        dtype = tensor.dtype
        for aidx, src_sl, dst_sl, pp_list in tensor.slices:
            trans_dst_sl = transpose_list(dst_sl, ast.literal_eval(permutation))
            if pp_list is not None:
                new_pp_list = pp_list.copy()
                new_pp_list.append(permutation)
                slices.append((aidx, src_sl, trans_dst_sl, new_pp_list))
            else:
                slices.append((aidx, src_sl, trans_dst_sl, [permutation]))
        return TensorDesc(
            slices, tensor_shape, in_degree=1, out_degree=0, dtype=dtype
        )

    def cast(self, tensor: TensorDesc, dtype: str) -> TensorDesc:
        slices = []
        tensor.out_degree += 1
        for aidx, src_sl, dst_sl, pp_list in tensor.slices:
            if pp_list is not None:
                new_pp_list = pp_list.copy()
                new_pp_list.append(dtype)
                slices.append((aidx, src_sl, dst_sl, new_pp_list))
            else:
                slices.append((aidx, src_sl, dst_sl, [dtype]))
        # For the cast operation, post_process is required. Therefore, the returned
        # Tensor's dtype here is the same as the input tensor's dtype, rather than the casted dtype.
        return TensorDesc(
            slices, tensor.shape, in_degree=1, out_degree=0, dtype=tensor.dtype
        )

    def identity(self, tensor: TensorDesc) -> TensorDesc:
        tensor.out_degree += 1
        return TensorDesc(
            tensor.slices,
            tensor.shape,
            in_degree=1,
            out_degree=0,
            dtype=tensor.dtype,
        )

    def shape_propagation(self):
        def _get_var_ref(var):
            if var.name in self.intermediate_vars:
                return self.intermediate_vars[var.name]
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
                        self.intermediate_vars[out_var.name] = out_ref
                        if (
                            out_var.name
                            in self.context.get_all_dst_state_keys()
                        ):
                            self.output_vars[out_var.name] = out_ref

                elif len(right_vars) == 1:
                    left_refs = [_get_var_ref(var) for var in left_vars]
                    result = self.concat(left_refs, axis)
                    out_name = right_vars[0].name
                    self.intermediate_vars[out_name] = result
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
                        assert len(attrs) == 1, "Only support one operator!"
                        attr = attrs[0]
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
                            result = in_ref
                        else:
                            raise ValueError(f"Unsupported attribute: {attr}")

                        self.intermediate_vars[rvar.name] = result
                        if rvar.name in self.context.get_all_dst_state_keys():
                            self.output_vars[rvar.name] = result
                    else:
                        # rename operation
                        in_ref = _get_var_ref(lvar)
                        result = self.identity(in_ref)
                        self.intermediate_vars[rvar.name] = result
                        if rvar.name in self.context.get_all_dst_state_keys():
                            self.output_vars[rvar.name] = result
            else:
                raise SyntaxError(f'Unexpected statement: {stmt}')
        if self.destination_state_shard_info is not None:
            for name in self.destination_state_shard_info:
                model_state_key, _ = split_optimizer_state_key(name)
                if model_state_key not in self.output_vars:
                    self.output_vars[model_state_key] = (
                        None
                        if model_state_key in self.need_add_output_vars
                        else self.input_vars[
                            model_state_key
                        ]  # Assertion implied by direct access
                    )
        else:
            # When destination_state_shard_info is not provided, the AOAEngine automatically derives it
            # from source_state_shard_info and aha_statements. In this case, all destination_states
            # remain unsharded (not partitioned).
            for name, ref_t in self.input_vars.items():
                if name not in self.output_vars and ref_t.out_degree == 0:
                    self.output_vars[name] = self.identity(ref_t)
            for name, ref_t in self.intermediate_vars.items():
                if name not in self.output_vars and ref_t.out_degree == 0:
                    self.output_vars[name] = self.identity(ref_t)

    def find_source_slices(
        self, key: str, local_slice: tuple[slice, ...]
    ) -> list[SliceRef]:
        assert key in self.output_vars
        tensor = self.output_vars[key]
        if tensor is None:
            return []
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
        target_key, opt_state_name = split_optimizer_state_key(target.key)
        target_local_shape = target.local_shape
        target_global_offset = target.global_offset
        target_global_shape = target.global_shape

        if opt_state_name in [".beta1_pow_acc_0", ".beta2_pow_acc_0"]:
            assert target_key in self.output_vars
            tensor = self.output_vars[target_key]
            target_local_shape = tensor.shape
            target_global_offset = (0,) * len(target_local_shape)
            target_global_shape = target_local_shape

        slices = tuple(
            slice(offset, offset + size, 1)
            for offset, size in zip(target_global_offset, target_local_shape)
        )

        results = self.find_source_slices(target_key, slices)

        shard_mappings = []

        target_key = (
            target_key + opt_state_name
            if opt_state_name is not None
            else target_key
        )

        src_keys = {
            result[0]
            for result in results
            if result[0] not in self.need_remove_input_vars
        }
        if opt_state_name in [".beta1_pow_acc_0", ".beta2_pow_acc_0"]:
            if len(src_keys) == 0:
                return shard_mappings
            elif len(src_keys) > 1:
                logger.warning(
                    f"{target_key} has multiple sources: {src_keys} (e.g., .beta1_pow_acc_0). Returning one arbitrarily."
                )
                src_key = next(iter(src_keys))
            else:
                src_key = next(iter(src_keys))
            return [
                ShardMappingEntry(
                    target,
                    ShardedWeightDesc(
                        src_key + opt_state_name,
                        target.local_shape,
                        target.global_shape,
                        target.global_offset,
                        target.dtype,
                    ),
                    None,
                )
            ]

        for src_key, src_slices, local_slices, pp_list in results:
            src_var = self.input_vars[src_key]
            target_model_state_key, target_opt_state_name = (
                split_optimizer_state_key(target.key)
            )
            if target_opt_state_name is None:
                if src_var.dtype != target.dtype:
                    assert pp_list is not None and target.dtype in str(
                        pp_list
                    ), (
                        "Direct assignment of Tensors with different types is prohibited in AOA. "
                        "If you want to achieve this functionality, please use the cast semantics provided by AOA."
                    )
            else:
                src_var.dtype = target.dtype

            src_global_shape = src_var.shape

            src_local_shape = tuple(slc.stop - slc.start for slc in src_slices)
            src_global_offset = tuple(slc.start for slc in src_slices)

            tgt_local_shape = tuple(
                slc.stop - slc.start for slc in local_slices
            )
            tgt_global_offset = tuple(slc.start for slc in local_slices)

            new_src_key = (
                src_key + opt_state_name
                if opt_state_name is not None
                else src_key
            )

            source_sharded_weight = ShardedWeightDesc(
                new_src_key,
                src_local_shape,
                tuple(src_global_shape),
                src_global_offset,
                src_var.dtype,
            )
            target_sharded_weight = ShardedWeightDesc(
                target_key,
                tgt_local_shape,
                tuple(target_global_shape),
                tgt_global_offset,
                target.dtype,
            )

            if src_key in self.need_remove_input_vars:
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
