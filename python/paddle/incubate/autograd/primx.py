# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import typing

import paddle
from paddle.base import framework
from paddle.base.core import ops_contain_none, prim_config
from paddle.incubate.autograd.utils import as_tensors

from .composite_rules import _composite
from .primreg import lookup_composite
from .utils import map_output_for_composite, prepare_python_api_arguments


def _lower_composite(
    block,
    filter_: typing.Callable[[framework.Operator], bool] = lambda x: True,
    start_idx=-1,
    backward_length=-1,
):
    """The operators in block wich satisfy the filter conditon will be decomposite into primitives."""

    def bind(args, to_bind, value_table):
        for i in range(len(args)):
            if isinstance(args[i], list):
                bind(args[i], to_bind, value_table)
            if not isinstance(args[i], paddle.base.framework.Variable):
                continue
            elif args[i] is not None and args[i].name in to_bind:
                args[i] = value_table[to_bind[args[i].name]]

    def bind_name(names, to_bind):
        return_list = []
        for name in names:
            if isinstance(name, list):
                return_list.append(bind_name(name, to_bind))
            else:
                return_list.append(to_bind[name] if name in to_bind else name)
        return return_list

    def expand_nested_list(xs):
        return_list = []
        for x in xs:
            if isinstance(x, list):
                return_list = return_list + expand_nested_list(x)
            else:
                return_list.append(x)
        return return_list

    if isinstance(block, paddle.base.framework.Block):
        logging.info("Atomize composite op to primitive ops begin.")

        # Step1: Do some preparatory work for lower
        lower_fn = _composite
        lookup_fn = lookup_composite

        value_table = {}
        to_bind = {}
        to_bind_rev = {}
        for var in block.desc.all_vars():
            value_table[var.name()] = block.var(var.name())

        ops_to_remove = []
        vars_to_remove = set()

        # if output var of composite rule is None, this means this var is not needed
        none_vars_to_remove = set()

        change = None

        # Only process required sliced block
        # If given start_idx, only ops[start_idx:] will be processed.
        # If given backward_length, only ops[:-backward_length] will be processed.
        # Note, start_idx and backward_length cannot be both given, because the length of non-processed part must be kept unchanged.
        length = len(block.ops)
        idx_list = range(length)
        assert (
            -1 <= backward_length <= length
        ), f'expect -1 <= backward_length <= {length}, but got backward_length: {backward_length}'
        assert (
            -1 <= start_idx <= length
        ), f'expect -1 <= start_idx <= {length}, but got start_idx: {start_idx}'
        assert not (
            backward_length > -1 and start_idx > -1
        ), f'got start_idx: {start_idx} and backward_length: {backward_length}'
        if backward_length > -1:
            idx_list = range(length - backward_length)
        if start_idx > -1:
            idx_list = range(start_idx, length)

        lower = lower_pre = False  # Flag of routing to lower or copy branch
        # Step2: Process all ops in the target block
        for op_idx in range(length):
            op = block.ops[op_idx]
            ops_to_remove.append(op_idx)
            op_name = op.type

            # NOTE: why need _sync_with_cpp here
            # _sync_wich_cpp after every copied operator is very slow.
            # However, _sync_wich_cpp only support continuous block currently.
            # The lowering transformation will generate program which is
            # crossed combination of copy block and lower block, such as
            # op1(copy) -> op2(copy) -> op3(lower) -> op4(lower) -> op5(copy) -> op6(copy)
            # It will cause _sync_wich_cpp error.
            # So, _sync_with_cpp will be executed only once after every continuous copy block.
            lower = (
                (lookup_fn(op_name) is not None)
                and filter_(op)
                and op_idx in idx_list
            )
            if not lower_pre and lower:
                block._sync_with_cpp()
            lower_pre = lower

            if lower:
                change = True
                prim_config["composite_ops_record"].add(op_name)
                input_args = prepare_python_api_arguments(op)
                bind(input_args, to_bind, value_table)

                orig_outs = expand_nested_list(map_output_for_composite(op))
                new_outs = expand_nested_list(
                    as_tensors(lower_fn(op, *input_args))
                )
                assert len(orig_outs) == len(new_outs), (
                    f'when replace origin op {op_name} with composite rule, num of origin outs should be equal to new outs, '
                    f'but len(orig_outs) = {len(orig_outs)} and len(new_outs) = {len(new_outs)}'
                )

                for orig_out, new_out in zip(
                    orig_outs,
                    new_outs,
                ):
                    if (orig_out is None or new_out is None) and (
                        op_name not in ops_contain_none
                    ):
                        raise ValueError(
                            f"op {op_name} should not contain any None value. original outs={orig_outs} and its composite rule outs={new_outs}"
                        )
                    if orig_out is None:
                        # to keep same as phi op definition, orig_out may receive None
                        continue
                    elif new_out is not None:
                        assert orig_out.dtype == new_out.dtype, (
                            f'when replace origin op {op_name} with composite rule, origin out dtype should be equal to new out dtype, '
                            f'but orig_out: {orig_out.name}.dtype={orig_out.dtype} and new_out: {new_out.name}.dtype={new_out.dtype}'
                        )
                        assert (
                            -1 not in new_out.shape
                        ), f'when replace origin op {op_name} with composite rule, composite out shape has -1.'
                        assert orig_out.shape == new_out.shape, (
                            f'when replace origin op {op_name} with composite rule, origin out shape should be equal to new out shape, '
                            f'but orig_out: {orig_out.name}.shape={orig_out.shape} and new_out: {new_out.name}.shape={new_out.shape}'
                        )
                        assert not (orig_out is None) ^ (
                            new_out is None
                        ), "orig_out and new_out should match."
                        vars_to_remove.add(new_out.name)
                        value_table[new_out.name] = new_out
                        to_bind[orig_out.name] = new_out.name
                        to_bind_rev[new_out.name] = orig_out.name
                    else:
                        none_vars_to_remove.add(orig_out.name)
            else:
                op_desc = block.desc.append_op()
                op_desc.copy_from(op.desc)

        block._sync_with_cpp()
        # Step3: Do some post-processing work
        for op_idx in reversed(ops_to_remove):
            block.desc._remove_op(op_idx, op_idx + 1)
            del block.ops[op_idx]
        block._sync_with_cpp()

        for op_idx in range(len(block.ops)):
            op = block.ops[op_idx]
            for in_name in op.input_arg_names:
                if in_name in to_bind_rev:
                    op._rename_input(in_name, to_bind_rev[in_name])

            for out_name in op.output_arg_names:
                if out_name in to_bind_rev:
                    op._rename_output(out_name, to_bind_rev[out_name])

        for var_name in sorted(vars_to_remove):
            assert (
                var_name in to_bind_rev
            ), f'var_name "{var_name}" is not in to_bind_rev.'
            if var_name != to_bind_rev[var_name]:
                block.desc._remove_var(var_name.encode())
                del block.vars[var_name]
        block._sync_with_cpp()

        for var_name in sorted(none_vars_to_remove):
            block.desc._remove_var(var_name.encode())
            del block.vars[var_name]
        block._sync_with_cpp()

        for op in block.ops:
            if op._has_kernel(op.desc.type()):
                op.desc.infer_var_type(block.desc)
                op.desc.infer_shape(block.desc)
        # composite ops may contain other composite ops, thus, call _lower_composite again.
        if change:
            _lower_composite(
                block,
                filter_,
                start_idx=start_idx,
                backward_length=backward_length,
            )
        return

    elif isinstance(block, typing.Sequence):
        for item in block:
            _lower_composite(
                item,
                filter_,
                start_idx=start_idx,
                backward_length=backward_length,
            )
        return
    else:
        raise TypeError
