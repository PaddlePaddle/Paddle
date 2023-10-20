# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import pir
from paddle.base.core import call_decomp, has_decomp
from paddle.base.libpaddle.pir import Block, Operation, Program
from paddle.framework import core

from . import register


def _build_tensor_tuple(xs):
    if isinstance(xs, pir.OpResult):
        return (xs,)
    elif isinstance(xs, typing.Sequence):
        return tuple(xs)
    return TypeError(f"Type {type(xs)} is not supported.")


def _analyse_decomp_results(orig_outs, decomp_outs):
    assert len(orig_outs) == len(decomp_outs)
    res = []
    for org_item, new_item in zip(orig_outs, decomp_outs):
        if isinstance(org_item, pir.OpResult):
            assert len(new_item) == 1 and isinstance(new_item[0], pir.OpResult)
            res.append(new_item[0])
        else:
            res.append(new_item)
    return res


def _prepare_python_api_arguments(op):
    """
    For standard api of operator, its inputs should keep consistent with organization of its inputs and attrs.

    Args:
    op (Operator): The target operator.
    """
    op_inputs = []
    for x in op.operands():
        op_input = x.source()
        upper_op = op_input.get_defining_op()
        if (
            isinstance(upper_op, Operation)
            and upper_op.name() == 'builtin.combine'
        ):
            op_input = [item.source() for item in upper_op.operands()]
        op_inputs.append(op_input)
    # The inputs of PIR op builtin.combine will be restored as list of tensor.
    if op.name() in ["builtin.combine"]:
        return (op_inputs,)

    op_attrs_dict = op.attrs()
    op_attrs_name = op.get_attr_names()
    op_attrs = [op_attrs_dict[x] for x in op_attrs_name]
    api_arguments = op_inputs + op_attrs
    return tuple(api_arguments)


def _check_op_results(op_name, orig_outs, new_outs, orig_vars, dst_vars):
    """
    Check whether the replaced outputs are consistent with origin outputs.

    Args:
    op_name (str): The name of operator.
    orig_outs (tuple): The outputs of original operator.
    new_outs (tuple): The outputs of replaced operator.
    orig_vars (dict): Origin variables of original block.
    dst_vars (list): Corresponding replaced variables of Origin variables.
    """
    assert len(orig_outs) == len(new_outs), (
        f'when replace origin op {op_name} with composite rule, num of origin outs should be equal to new outs, '
        f'but len(orig_outs) = {len(orig_outs)} and len(new_outs) = {len(new_outs)}'
    )

    for orig_out, new_out in zip(
        orig_outs,
        new_outs,
    ):
        if (orig_out is None or new_out is None) and (
            op_name not in core.ops_contain_none
        ):
            raise ValueError(
                f"op {op_name} should not contain any None value. original outs={orig_outs} and its composite rule outs={new_outs}"
            )
        if orig_out is None:
            # to keep same as phi op definition, orig_out may receive None
            continue
        elif new_out is not None:
            if orig_out in orig_vars.keys():
                dst_vars[orig_vars[orig_out]] = new_out
            orig_dtype = orig_out.dtype
            new_dtype = new_out.dtype
            orig_shape = orig_out.shape
            new_shape = new_out.shape
            assert orig_dtype == new_dtype, (
                f'when replace origin op {op_name} with composite rule, origin out dtype should be equal to new out dtype, '
                f'but orig_out dtype={orig_dtype} and new_out dtype={new_dtype}'
            )
            assert (
                -1 not in new_shape
            ), f'when replace origin op {op_name} with composite rule, composite out shape has -1.'
            assert orig_shape == new_shape, (
                f'when replace origin op {op_name} with composite rule, origin out shape should be equal to new out shape, '
                f'but orig_out shape={orig_shape} and new_out shape={new_shape}'
            )
            assert not (orig_out is None) ^ (
                new_out is None
            ), "orig_out and new_out should match."
        return


def decompose(
    program,
    src_vars,
    blacklist=frozenset(),
    whitelist=frozenset(),
):
    """
    Search nonbasic ops which have be registered composite rules and replace them with primitive ops.
    The operators in blacklist will be excluded from program when decomposed into primitives, and only the
    operators in whitelist will be decomposed. The priority of blacklist is higher than whitelist, it means
    an operator both in blacklist and whitelist will not be decomposed.

    The finally set that will be decomposed is:
        (block.ops & ops have decomposite rule & whitelist) - blacklist

    Note:
        All variables must be contained inside the given program.

    Args:
        program (Program): The program to be processed.
        src_vars (list[OpResult]): In program, once some operator is decomposed, its vars will be replaced by new ones. This argument means some vars will be used later and corresponding vars will be returned for later usage.
        blacklist (frozenset): The Operators that will be exclude when decomposed into primitives.
        whitelist (frozenset): Only the operators in whitelist will be decomposed into primitives.

    Returns:
        dst_vars (list): A list contains all vars which replace origin ones in src_vars.
    """
    if not core._is_fwd_prim_enabled():
        return src_vars
    if not isinstance(program, Program):
        raise TypeError(f"Expect type Program, but got type {type(program)}.")
    block = program.global_block()

    if not isinstance(blacklist, (set, frozenset)):
        raise TypeError(
            f'Expected type of blacklisst is set|frozenset, but got {type(blacklist)}.'
        )
    if not isinstance(whitelist, (set, frozenset)):
        raise TypeError(
            f'Expected type of whiltelist is set|frozenset, but got {type(whitelist)}.'
        )

    blacklist = core.prim_config["forward_blacklist"] | blacklist

    logging.debug("Decompose composite forward ops begin...")

    if len(blacklist) > 0 and len(whitelist) > 0:
        op_filter = (
            lambda x: x.name() in whitelist and x.name() not in blacklist
        )
    elif len(blacklist) > 0 and len(whitelist) == 0:
        op_filter = lambda x: x.name() not in blacklist
    elif len(blacklist) == 0 and len(whitelist) > 0:
        op_filter = lambda x: x.name() in whitelist
    else:
        op_filter = lambda x: True
    dst_vars = [None] * len(src_vars)
    dst_vars_dct = {}
    for idx, item in enumerate(src_vars):
        if not isinstance(item, pir.OpResult):
            raise TypeError(
                f"Each var in dst_vars should map corresponding var in src_vars, but got type {type(item)} in {src_vars}."
            )
        dst_vars_dct[item] = idx
    with pir.core.program_guard(program):
        _decompose_subgraph(
            block,
            dst_vars_dct,
            dst_vars,
            op_filter,
        )
    for idx, item in enumerate(dst_vars):
        if not isinstance(item, pir.OpResult):
            if item is None:
                dst_vars[idx] = src_vars[idx]
            else:
                raise TypeError(
                    f"Each var in dst_vars should map corresponding var in src_vars, but got type {type(item)} in {dst_vars}."
                )
    logging.debug(
        "Decompose composite forward ops finish: {}".format(
            core.prim_config["composite_ops_record"]
        )
    )
    return dst_vars


def _decompose_subgraph(block, orig_vars, dst_vars, op_filter):
    """
    The operators in block wich satisfy the filter conditon will be decomposed into primitives.

    Args:
        block (Block|Sequence[Block]): The blocks of program to be processed.
        op_filter (function): The filter to specify which ops to be processed.
        orig_vars (dict): Origin variables of original block.
        dst_vars (list): Corresponding replaced variables of Origin variables.
    """

    if isinstance(block, Block):
        ops_list = block.ops
        temp_op = None
        for idx, op in enumerate(ops_list):
            op_name = op.name()
            decom_rule = register.get_decomp_rule(op_name)
            has_sink_decomp_rule = has_decomp(op)
            lower = (decom_rule or has_sink_decomp_rule) and op_filter(op)

            if op.name() == "builtin.combine":
                temp_op = op

            if lower:
                core.prim_config["composite_ops_record"].add(op_name)
                if (
                    temp_op is not None
                    and ops_list[idx - 1].name() == "builtin.combine"
                ):
                    pir.set_insertion_point(temp_op)
                else:
                    pir.set_insertion_point(op)
                input_args = _prepare_python_api_arguments(op)
                orig_outs = op.results()
                if has_sink_decomp_rule:
                    decomp_outs = call_decomp(op)
                    new_outs = _analyse_decomp_results(orig_outs, decomp_outs)
                else:
                    new_outs = _build_tensor_tuple(decom_rule(*input_args))

                # Todo: To cover such case: some outputs are no longer needed after decomposition.
                _check_op_results(
                    op_name, orig_outs, new_outs, orig_vars, dst_vars
                )
                if op.name() in ("pd_op.unsqueeze", "pd_op.squeeze"):
                    orig_outs[0].replace_all_uses_with(new_outs[0])
                else:
                    op.replace_all_uses_with(new_outs)
                block.remove_op(op)

                if temp_op is not None:
                    remove_op = True
                    for item in temp_op.results():
                        if item.has_one_use():
                            remove_op = False
                            break
                    if remove_op:
                        block.remove_op(temp_op)
                    temp_op = None
        return

    elif isinstance(block, typing.Sequence):
        for item in block:
            _decompose_subgraph(item, orig_vars, dst_vars, op_filter)
        return
    raise TypeError(
        f"Expect type Block or Sequence of Block, but got type {type(block)}"
    )
