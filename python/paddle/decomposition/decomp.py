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
from paddle.autograd import ir_backward
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
    combine_op_name = "builtin.combine"
    inputs = []
    for x in op.operands():
        input = x.source()
        if input and input.initialized():
            prev_op = input.get_defining_op()
            if (
                isinstance(prev_op, Operation)
                and prev_op.name() == combine_op_name
            ):
                input = [item.source() for item in prev_op.operands()]
            inputs.append(input)
        else:
            # for optional input, such as scale for layer_norm op,
            # if it is not set, there will be an empty OpResult which is not initialized in ops.operands
            # therefore append None for it.
            inputs.append(None)

    # The inputs of PIR op builtin.combine will be restored as list of tensor.
    if op.name() == combine_op_name:
        return (inputs,)

    api_arguments = inputs + [op.attrs()[x] for x in op.get_attr_names()]
    return tuple(api_arguments)


def _check_op_results(
    op_name, orig_outs, new_outs, orig_vars=None, dst_vars=None
):
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
            if orig_vars is not None and dst_vars is not None:
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


def get_leaf_ops(block, global_outputs):
    '''
    This API checks which op contributes to the outputs of the entire computation graph,
    as well as determining the corresponding output index.

    Args:
        block (Block): the block of program to be processed.
        global_outputs (tuple(Value)): the outputs of the entire computation graph.

    Returns:
        related_ops (tuple(pir.Operation)): a tuple of op that contributes to the outputs of the entire graph.
        related_ops_output_indexes (tuple(tuple())) : a tuple records the mapping of tuple(the output index of the op,  the output index of the entire graph)
    '''
    if not isinstance(block, Block):
        raise TypeError(f"block should be Block, but got type {type(block)}")
    if not isinstance(global_outputs, list):
        raise TypeError("The type of global_outputs should be list")

    related_ops = []
    related_ops_output_indexes = []

    op_to_op_valid_result = {}
    for op in block.ops:
        op_valid_result = []
        for x in op.results():
            if x.initialized():
                op_valid_result.append(x)
        op_to_op_valid_result[op] = op_valid_result

    for global_output in global_outputs:
        for op in op_to_op_valid_result.keys():
            if global_output in op_to_op_valid_result[op]:
                if op not in related_ops:
                    related_ops.append(op)
                    related_ops_output_indexes.append(
                        [
                            [
                                op.results().index(global_output),
                                global_outputs.index(global_output),
                            ]
                        ]
                    )
                else:
                    related_ops_output_indexes[related_ops.index(op)].append(
                        [
                            op.results().index(global_output),
                            global_outputs.index(global_output),
                        ]
                    )

    return tuple(related_ops), tuple(related_ops_output_indexes)


def replace_graph_outputs(
    global_outputs,
    op_outputs,
    op_index,
    related_ops_output_indexes,
):
    '''
    This API replace the outputs of the entire computation graph with the new outputs of the op,
    when the op contributes to the outputs of the entire computation graph.
    '''
    for index in related_ops_output_indexes[op_index]:
        global_outputs[index[1]] = op_outputs[index[0]]


def decompose_fwd_op(
    block: Block, fwd_op: pir.Operation, grad_var_to_var_map: dict
) -> tuple:
    '''
    Decompose the fwd_op into a list of primitive ops.

    Args:
        block (Block): the block to which the fwd_op belongs.
        fwd_op (pir.Operation): the forward op to be decomposed.
        grad_var_to_var_map (dict): a dict obtained from distributed processing,
            which maps the backward grad variable to its corresponding forward variable.
    Returns:
        new_outputs (tuple(Value)): the new outputs after decomposing.
    '''

    if not core._is_fwd_prim_enabled():
        raise RuntimeError(
            "To decompose forward op, please set `core._set_prim_forward_enabled(True)` firstly"
        )

    with pir.core.program_guard(block.program):
        op_name = fwd_op.name()
        orig_outs = fwd_op.results()
        decom_rule = register.get_decomp_rule(op_name)
        has_sink_decomp_rule = has_decomp(fwd_op)
        lower = decom_rule or has_sink_decomp_rule

        if lower:
            input_args = _prepare_python_api_arguments(fwd_op)
            pir.set_insertion_point(fwd_op)
            if has_sink_decomp_rule:
                decomp_outs = call_decomp(fwd_op)
                new_outs = _analyse_decomp_results(orig_outs, decomp_outs)
            else:
                new_outs = _build_tensor_tuple(decom_rule(*input_args))

            _check_op_results(op_name, orig_outs, new_outs)

            # update_grad_var_to_var_map
            for grad_var, var in grad_var_to_var_map.items():
                if var in orig_outs:
                    grad_var_to_var_map[grad_var] = new_outs[
                        orig_outs.index(var)
                    ]

            fwd_op.replace_all_uses_with(new_outs)
            block.remove_op(fwd_op)
            return new_outs
        else:
            return tuple(orig_outs)


def decompose_bwd_op(
    block: Block,
    bwd_op: pir.Operation,
    grad_var_to_var_map: dict,
    fwd_outputs: tuple,
    fwd_inputs: tuple,
) -> tuple:
    '''
    Lowering a first-order derivative PHI operator into primitive operators, steps are as follows:
    step1: get grad_outputs from the bwd_op's operands, which is a subset of bwd_op's operands after excluding the inputs and outputs of fwd_op;
    step2: get the new_fwd_outputs via grad_var_to_var_map, which correspond one-to-one with grad_outputs;
    step3: get the new_fwd_inputs, iterate over the initialized result in bwd_op's results, then find the corresponding fwd_input based on grad_var_to_var_map;
    step4: call grad() API, decompose the bwd_op, and get new gradients;
    step5: replace bwd_op with a set of primitive ops;
    step6: update grad_var_to_var_map.

    Args:
        block (Block): the block to which the backward op belongs.
        bwd_op (pir.Operation): the backward op to be decomposed.
        grad_var_to_var_map (dict): a dict obtained from distributed processing,
            which maps the backward grad variable to its corresponding forward variable.
        fwd_outputs (tuple(Value)): the output value tuple of the forward op.
        fwd_inputs (tuple(Value)): the input value tuple of the forward op.

    Returns:
        new_input_grads (tuple(Value)): the input grad value tuple, the i-th returned value is the sum of gradients of `fwd_outputs` with respect to the i-th `fwd_inputs`.
    '''

    if not core._is_bwd_prim_enabled():
        raise RuntimeError(
            "To get composite backward op, please set `core._set_prim_backward_enabled(True)` firstly"
        )

    # intercept grad_outputs from the original bwd_op
    # grad_outputs = bwd_op.operands() - fwd_inputs - fwd_outputs
    bwd_inputs = tuple(x.source() for x in bwd_op.operands())
    grad_outputs = tuple(
        bwd_input
        for bwd_input in bwd_inputs
        if not (bwd_input in fwd_inputs or bwd_input in fwd_outputs)
    )

    # new_fwd_outputs is a subset of fwd_outputs, because some fwd_output does not hold the gradients,
    # e.g., layer_norm op's output is [out, mean, variance], but only out holds gradient,
    # therefore, parse the new_fwd_outputs according to grad_outputs and grad_var_to_var_map
    new_fwd_outputs = tuple(
        grad_var_to_var_map[grad_output] for grad_output in grad_outputs
    )

    # new_fwd_inputs is a subset of fwd_inputs, because some fwd_input does not need to compute the gradients,
    # e.g., dropout op's input is [x, seed_tensor], but the seed_tensor is generated by the forward op, and does not need to compute the gradients,
    # therefore, parse the new_fwd_inputs according to bwd_op.results() and grad_var_to_var_map
    new_fwd_inputs = tuple(
        grad_var_to_var_map[grad_input]
        for grad_input in bwd_op.results()
        if grad_input.initialized()
    )

    # when replace bwd_op with a list of primitive ops, a insertion point is needed
    bwd_op_idx = block.ops.index(bwd_op)
    # decompose bwd_op into a list of primitive ops
    before_num_ops = len(block.ops)
    input_grads = ir_backward.grad(
        new_fwd_outputs, new_fwd_inputs, grad_outputs
    )
    after_num_ops = len(block.ops)

    # update the bwd_op's results
    # when the original result of the bwd_op is None, then fake an OpResult for replacement
    # when the original result of the bwd_op is not None, then replace it with the new result of primitive ops
    new_input_grads = []
    input_grads_idx = 0
    for idx, input_grad in enumerate(bwd_op.results()):
        if input_grad.initialized():
            new_input_grads.append(input_grads[input_grads_idx])
            input_grads_idx += 1
        else:
            new_input_grads.append(pir.fake_op_result())

    # move the primitive ops to the insertion point
    insert_idx = bwd_op_idx
    for i in range(before_num_ops, after_num_ops):
        block.move_op(block.ops[i], insert_idx)
        insert_idx += 1

    # update_grad_var_to_var_map
    for idx, grad_var in enumerate(bwd_op.results()):
        if grad_var in grad_var_to_var_map.keys():
            grad_var_to_var_map[new_input_grads[idx]] = grad_var_to_var_map.pop(
                grad_var
            )

    # replace the following use of original bwd_op's results with new primitive ops' results, and then remove original bwd_op
    bwd_op.replace_all_uses_with(new_input_grads)
    block.remove_op(bwd_op)

    return tuple(new_input_grads)
