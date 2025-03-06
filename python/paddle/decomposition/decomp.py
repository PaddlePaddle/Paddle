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
import warnings

import paddle
from paddle import pir
from paddle.autograd import ir_backward
from paddle.autograd.backward_utils import ValueDict, ValueSet
from paddle.base.core import (
    call_decomp_rule,
    call_decomp_vjp,
    decomp_ops_contain_unused_output,
    has_decomp_rule,
    has_decomp_vjp,
)
from paddle.base.framework import (
    pir_chunk_id_guard,
    pir_op_name_guard,
    pir_op_role_guard,
)
from paddle.base.libpaddle.pir import Block, Operation
from paddle.base.wrapped_decorator import signature_safe_contextmanager
from paddle.decomposition.recompute import DebugPrint, auto_recompute
from paddle.framework import core

from . import register

logger = logging.getLogger(__name__)


@signature_safe_contextmanager
def prim_guard():
    prim_state = core._is_all_prim_enabled()
    try:
        if not prim_state:
            core._set_prim_all_enabled(True)
        yield
    finally:
        if not prim_state:
            core._set_prim_all_enabled(False)


def _build_tensor_tuple(xs):
    if isinstance(xs, pir.Value):
        return (xs,)
    elif isinstance(xs, typing.Sequence):
        return tuple(xs)
    return TypeError(f"Type {type(xs)} is not supported.")


def _analyse_decomp_results(orig_outs, decomp_outs, op):
    assert len(orig_outs) == len(decomp_outs)
    res = []
    for idx, value in enumerate(decomp_outs):
        if isinstance(orig_outs[idx], pir.Value):
            if (
                op.name() in decomp_ops_contain_unused_output.keys()
                and idx in decomp_ops_contain_unused_output[op.name()]
            ):
                assert value[0] is None
            else:
                assert len(value) == 1 and isinstance(value[0], pir.Value)
            res.append(value[0])
        else:
            res.append(value)
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
        if input.initialized():
            prev_op = input.get_defining_op()
            if (
                isinstance(prev_op, Operation)
                and prev_op.name() == combine_op_name
            ):
                input = [item.source() for item in prev_op.operands()]
            inputs.append(input)
        else:
            # for optional input, such as scale for layer_norm op,
            # if it is not set, there will be an empty Value which is not initialized in ops.operands
            # therefore append None for it.
            inputs.append(None)

    # The inputs of Pir op builtin.combine will be restored as list of tensor.
    if op.name() == combine_op_name:
        return (inputs,)

    api_arguments = inputs + [op.attrs()[x] for x in op.get_attr_names()]
    return tuple(api_arguments)


def _check_prim_dynamic(op):
    combine_op_name = "builtin.combine"
    inputs = []
    for x in op.operands():
        input = x.source()
        if input.initialized():
            prev_op = input.get_defining_op()
            if (
                isinstance(prev_op, Operation)
                and prev_op.name() == combine_op_name
            ):
                for item in prev_op.operands():
                    shape = item.source().shape
                    if -1 in shape:
                        warnings.warn(
                            f"Decomp op does not support dynamic shape -1, but got shape {item.source().shape} in inputs of op {op.name()} "
                        )
                        return True
            else:
                shape = input.shape
                if -1 in shape:
                    warnings.warn(
                        f"Decomp op does not support dynamic shape -1, but got shape {input.shape} in op {op.name()} "
                    )
                    return True


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
                if orig_out in orig_vars:
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
    start_index=0,
    end_index=-1,
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
        src_vars (list[Value]): In program, once some operator is decomposed, its vars will be replaced by new ones. This argument means some vars will be used later and corresponding vars will be returned for later usage.
        blacklist (frozenset): The Operators that will be exclude when decomposed into primitives.
        whitelist (frozenset): Only the operators in whitelist will be decomposed into primitives.
        start_index (int): The start index of decomposed operator in global block, default 0;
        end_index (int): The end index of decomposed operator in global block, default -1 means all ops will be composed. start_index and end_index follow the principle of left closed and right open, that is [start_index, end_index).

    Returns:
        dst_vars (list): A list contains all vars which replace origin ones in src_vars.
    """
    blacklist = core.prim_config["forward_blacklist"] | blacklist
    assert isinstance(start_index, int)
    assert isinstance(end_index, int)
    return core.sinking_decomp(
        program, src_vars, blacklist, whitelist, start_index, end_index
    )


def _check_combine_inputs(input1, input2):
    '''check whether the inputs of two builtins.combine ops are the same'''
    builtin_combine_op1 = input1.get_defining_op()
    builtin_combine_op2 = input2.get_defining_op()
    if builtin_combine_op1.num_operands() != builtin_combine_op2.num_operands():
        return False
    else:
        for i in range(builtin_combine_op1.num_operands()):
            if not (
                builtin_combine_op1.operand_source(i).is_same(
                    builtin_combine_op2.operand_source(i)
                )
            ):
                return False
    return True


def _check_op(
    fwd_op: pir.Operation,
    bwd_op: pir.Operation,
):
    '''check whether the bwd_op is corresponding to fwd_op'''
    if fwd_op is None or fwd_op.name() + "_grad" != bwd_op.name():
        return False

    bwd_op_input_names = bwd_op.get_input_names()
    bwd_inputs = [x.source() for x in bwd_op.operands()]
    assert len(bwd_op_input_names) == len(
        bwd_inputs
    ), "backward op names do not match backward op inputs"
    fwd_op_related_inputs_outputs = []
    for idx, name in enumerate(bwd_op_input_names):
        if "_grad" not in name:
            fwd_op_related_inputs_outputs.append(bwd_inputs[idx])
    fwd_inputs = [x.source() for x in fwd_op.operands()]
    fwd_outputs = fwd_op.results()
    fwd_vec_inputs = [
        x.source()
        for x in fwd_op.operands()
        if x.source().initialized()
        and x.source().get_defining_op().name() == "builtin.combine"
    ]

    inserted_op_name_list = ["pd_op.full_int_array", "pd_op.full"]
    for operand in fwd_op_related_inputs_outputs:
        if (
            operand.initialized()
            and operand.get_defining_op().name() == "builtin.combine"
        ):  # for pir::VectorType<paddle::dialect::DenseTensorType>
            in_fwd = False
            for vec_input in fwd_vec_inputs:
                if _check_combine_inputs(operand, vec_input):
                    in_fwd = True
                    break
            if not in_fwd:
                return False
        else:  # for pir::VectorType<paddle::dialect::DenseTensorType>
            if not (
                operand in ValueSet(fwd_inputs)
                or operand in ValueSet(fwd_outputs)
                or operand.get_defining_op().name() in inserted_op_name_list
            ):
                return False

    return True


def _get_fwd_op(bwd_op, grad_var_to_var):
    bwd_op_input_names = bwd_op.get_input_names()
    out_grad_name = ["out_grad", "Out_grad", "loss_grad"]
    for idx, input_name in enumerate(bwd_op_input_names):
        if input_name in out_grad_name:
            out_grad = bwd_op.operand(idx).source()
            if out_grad in grad_var_to_var:
                out = grad_var_to_var[out_grad]
                fwd_op = out.get_defining_op()
                return fwd_op
    return None


def _decomp_fwd_op(
    block: Block, fwd_op: pir.Operation, grad_var_to_var: dict, prev_op=None
) -> tuple:
    '''
    Decompose the forward op into a list of primitive ops.
    Args:
        block (Block): the block to which the forward op belongs.
        fwd_op (pir.Operation): the forward op to be decomposed.
        grad_var_to_var (dict): a dict obtained from distributed processing,
            which maps the backward grad variable to its corresponding forward variable.
        prev_op (pir.Operation): the previous op of fwd_op in the block. If prev_op is builtin.combine, insertion point when decomposing fwd_op will be set to prev_op.
    Returns:
        new_outputs (tuple(Value)): the new outputs after decomposing.
        has_decomposed: whether the forward op has been successfully decomposed.
    '''

    with pir.core.program_guard(block.program):
        op_name = fwd_op.name()
        orig_outs = fwd_op.results()
        decom_rule = register.get_decomp_rule(op_name)
        has_sink_decomp_rule = has_decomp_rule(fwd_op)
        lower = decom_rule or has_sink_decomp_rule

        if lower:
            # step1: check dynamic shape, currently not supported
            if _check_prim_dynamic(fwd_op):
                return None, False

            # step2: check insertion point, if prev_op is builtin.combine (such as concat op), insertion point will be set to prev_op
            if prev_op is not None:
                pir.set_insertion_point(prev_op)
            else:
                pir.set_insertion_point(fwd_op)

            # step3: decompose op, and get new outputs
            input_args = _prepare_python_api_arguments(fwd_op)
            if has_sink_decomp_rule:
                decomp_outs = call_decomp_rule(fwd_op)
                new_outs = _analyse_decomp_results(
                    orig_outs, decomp_outs, fwd_op
                )
            else:
                new_outs = _build_tensor_tuple(decom_rule(*input_args))
            _check_op_results(op_name, orig_outs, new_outs)

            # step4: upgrade grad_var_to_var with new outputs
            _upgrade_grad_var_to_var(
                grad_var_to_var, orig_outs=orig_outs, new_outs=new_outs
            )

            # step5: replace original op with new ops, replace original output with new outputs
            if fwd_op.name() in decomp_ops_contain_unused_output.keys():
                for idx in range(len(orig_outs)):
                    if (
                        idx
                        not in decomp_ops_contain_unused_output[fwd_op.name()]
                    ):
                        orig_outs[idx].replace_all_uses_with(new_outs[idx])
            else:
                if fwd_op.name() in decomp_ops_contain_unused_output.keys():
                    orig_outs[0].replace_all_uses_with(new_outs[0])
                else:
                    fwd_op.replace_all_uses_with(new_outs)
            block.remove_op(fwd_op)

            # step6: remove redundant prev_op (builtin.combine)
            if prev_op is not None:
                remove_op = True
                for item in prev_op.results():
                    if item.has_one_use():
                        remove_op = False
                        break
                if remove_op:
                    block.remove_op(prev_op)
                prev_op = None
            return new_outs, True

        else:
            return tuple(orig_outs), False


def _prepare_inputs(fwd_op):
    new_inputs = []
    for input in fwd_op.operands():
        if (
            input.source().initialized()
            and input.source().get_defining_op().name() == "builtin.combine"
        ):  # for pir::VectorType<paddle::dialect::DenseTensorType>
            builtin_combine_op = input.source().get_defining_op()
            new_input = [
                builtin_combine_op.operand_source(i)
                for i in range(0, builtin_combine_op.num_operands())
            ]
            new_inputs.append(new_input)
        else:
            new_inputs.append([input.source()])  # for DenseTensorType
    return new_inputs


def _prepare_grad_outputs(fwd_op, bwd_op):
    # check forward outputs and backward inputs
    fwd_outputs = fwd_op.results()
    fwd_output_names = fwd_op.get_output_names()
    assert len(fwd_output_names) == len(
        fwd_outputs
    ), "forward op output names do not match forward op outputs"
    bwd_inputs = [x.source() for x in bwd_op.operands()]
    bwd_input_names = bwd_op.get_input_names()
    assert len(bwd_input_names) == len(
        bwd_inputs
    ), "backward op input names do not match backward op inputs"

    # cut gradients from backward op's inputs
    fwd_inputs = [x.source() for x in fwd_op.operands()]
    fwd_vec_inputs = [
        x.source()
        for x in fwd_op.operands()
        if x.source().initialized()
        and x.source().get_defining_op().name() == "builtin.combine"
    ]
    grad_outputs = []
    grad_output_names = []
    for i, bwd_input in enumerate(bwd_inputs):
        if (
            bwd_input.initialized()
            and bwd_input.get_defining_op().name() == "builtin.combine"
        ):  # for pir::VectorType<paddle::dialect::DenseTensorType>
            in_fwd = False
            for vec_input in fwd_vec_inputs:
                if _check_combine_inputs(bwd_input, vec_input):
                    in_fwd = True
                    break
            if not in_fwd:
                grad_outputs.append([bwd_input])
                grad_output_names.append(bwd_input_names[i])
        else:
            if not (
                bwd_input in ValueSet(fwd_inputs)
                or bwd_input in ValueSet(fwd_outputs)
            ):  # for paddle::dialect::DenseTensorType
                grad_outputs.append([bwd_input])
                grad_output_names.append(bwd_input_names[i])

    # add fake grads for forward op's outputs which are not used in backward op
    # this is necessary for the call_vjp(), which ensures that len(out_grads) must be equal to len(outputs)
    new_grad_outputs = []
    index = 0
    for fwd_output_name in fwd_output_names:
        if (fwd_output_name + "_grad") in grad_output_names:
            new_grad_outputs.append(grad_outputs[index])
            index += 1
        else:
            new_grad_outputs.append([pir.fake_value()])
    return new_grad_outputs


def _prepare_stop_gradients(fwd_inputs, bwd_outputs):
    stop_gradients = []
    for idx, bwd_output in enumerate(bwd_outputs):
        if bwd_output.initialized():
            stop_gradient = [False] * len(fwd_inputs[idx])
        else:
            stop_gradient = [True] * len(fwd_inputs[idx])
        stop_gradients.append(stop_gradient)
    return stop_gradients


def _upgrade_grad_var_to_var(
    grad_var_to_var,
    orig_grads=None,
    new_grads=None,
    orig_outs=None,
    new_outs=None,
):
    assert grad_var_to_var is not None, "grad_var_to_var should not be None"
    if orig_grads is not None and new_grads is not None:
        for idx, grad_input in enumerate(orig_grads):
            if grad_input in grad_var_to_var:
                grad_var_to_var[new_grads[idx]] = grad_var_to_var.pop(
                    grad_input
                )
    if orig_outs is not None and new_outs is not None:
        for grad_var, var in grad_var_to_var.items():
            for i, orin_var in enumerate(orig_outs):
                if var.is_same(orin_var):
                    grad_var_to_var[grad_var] = new_outs[i]


def _decomp_bwd_with_vjp(
    block: Block,
    fwd_op: pir.Operation,
    bwd_op: pir.Operation,
    grad_var_to_var: dict,
) -> tuple:
    '''
    Decompose the backward op into a list of primitive ops.
    If forward op has composite vjp rules (including custom vjp), call call_vjp() to get a list of primitive operators in backward graph, then replace backward op.
    '''
    # step1: prepare arguments for call_vjp()
    fwd_inputs_ = _prepare_inputs(fwd_op)
    fwd_outputs_ = [[fwd_output] for fwd_output in fwd_op.results()]
    grad_outputs_ = _prepare_grad_outputs(fwd_op, bwd_op)
    stop_gradients_ = _prepare_stop_gradients(fwd_inputs_, bwd_op.results())

    # step2: call call_vjp() to get a list of primitive operators which has the same meaning as the backward op
    bwd_op_idx = block.ops.index(bwd_op)
    before_num_ops = len(block.ops)
    new_grad_inputs = core.call_vjp(
        fwd_op, fwd_inputs_, fwd_outputs_, grad_outputs_, stop_gradients_
    )
    after_num_ops = len(block.ops)
    num_appended_ops = after_num_ops - before_num_ops

    # if forward op has no composite vjp rules, call_vjp() appends the same op as original backward op, skip decomposing, return False
    if num_appended_ops == 1 and block.ops[-1].name() == bwd_op.name():
        block.remove_op(block.ops[-1])
        return None, False
    else:
        # step3: record new outputs of the decomposed backward op
        if block.ops[-1].name() == "builtin.split":
            new_grad_inputs = [[block.ops[-1].operand(0).source()]]
        res = []
        for grad_input in new_grad_inputs:
            if grad_input[0] is not None and grad_input[0].initialized():
                res.append(grad_input[0])
            else:
                res.append(pir.fake_value())
        assert len(res) == len(
            bwd_op.results()
        ), "results of original backward op do not match results of decomposed backward op"

        # step4: upgrade grad_var_to_var
        _upgrade_grad_var_to_var(
            grad_var_to_var, orig_grads=bwd_op.results(), new_grads=res
        )

        # step5: replace original backward op with new primitive ops
        insert_idx = bwd_op_idx
        for i in range(before_num_ops, after_num_ops):
            block.move_op(block.ops[i], insert_idx)
            insert_idx += 1
        bwd_op.replace_all_uses_with(res)
        block.remove_op(bwd_op)

        return tuple(res), True


def _decomp_bwd_without_vjp(
    block: Block,
    bwd_op: pir.Operation,
    grad_var_to_var: dict,
    fwd_inputs: list,
    fwd_outputs_after_decompose: tuple,
) -> tuple:
    '''
    Decompose the backward op into a list of primitive ops.
    If forward op has no composite vjp rules, and forward op has been decomposed to a list of primitive operators in forward graph previously,
    call grad() for the decomposed forward subgraph to get a list of primitive operators in backward graph, then replace backward op.
    '''

    if fwd_outputs_after_decompose is None:
        raise RuntimeError(
            "To decompose backward op, please decompose forward op firstly"
        )

    # step1: prepare arguments for grad()
    bwd_inputs = [x.source() for x in bwd_op.operands()]
    grad_inputs = bwd_op.results()
    grad_outputs = tuple(
        bwd_input
        for bwd_input in bwd_inputs
        if not (
            bwd_input in ValueSet(fwd_inputs)
            or bwd_input in ValueSet(fwd_outputs_after_decompose)
        )
    )
    fwd_outputs_ = tuple(
        grad_var_to_var[grad_output] for grad_output in grad_outputs
    )
    fwd_inputs_ = tuple(
        grad_var_to_var[grad_input]
        for grad_input in grad_inputs
        if grad_input.initialized()
    )

    # step2: call grad() to get a list of primitive operators which has the same meaning as the backward op
    bwd_op_idx = block.ops.index(bwd_op)
    before_num_ops = len(block.ops)
    new_grad_inputs = ir_backward.grad(fwd_outputs_, fwd_inputs_, grad_outputs)
    after_num_ops = len(block.ops)

    # step3: record new outputs of the decomposed backward op
    res = []
    input_grads_idx = 0
    for idx, grad_input in enumerate(grad_inputs):
        if grad_input.initialized():
            res.append(new_grad_inputs[input_grads_idx])
            input_grads_idx += 1
        else:
            res.append(pir.fake_value())

    # step4: upgrade grad_var_to_var
    _upgrade_grad_var_to_var(
        grad_var_to_var, orig_grads=grad_inputs, new_grads=res
    )

    # step5: replace original backward op with new primitive ops
    insert_idx = bwd_op_idx
    for i in range(before_num_ops, after_num_ops):
        block.move_op(block.ops[i], insert_idx)
        insert_idx += 1
    bwd_op.replace_all_uses_with(res)
    block.remove_op(bwd_op)
    has_decomposed = True

    return tuple(res), has_decomposed


def _decomp_bwd_op(
    block: Block,
    bwd_op: pir.Operation,
    grad_var_to_var: dict,
):
    '''
    Decompose a backward op in pir program.
    Get the corresponding forward op according to grad_var_to_var firstly, then
    (1) try to decompose backward op by calling _decompose_bwd_with_vjp, if forward op has composite vjp rules (including custom vjp),
    _decompose_bwd_with_vjp will call call_vjp() to get a list of primitive operators in backward graph, then replace backward op successfully and return True;
    (2) when _decompose_bwd_with_vjp return False, means there is no composite vjp rules,
    try to decompose forward op firstly by calling _decomp_fwd_op firstly and get corresponding primitive operators in backward graph by calling _decompose_bwd_without_vjp secondly, then replace backward op successfully and return True;
    (3) if the backward op is still not decomposed by the above two steps, returns False.

    Args:
        block (Block): the block to which the backward op belongs.
        bwd_op (pir.Operation): the backward op to be decomposed.
        grad_var_to_var (dict): a dict obtained from distributed processing,
            which maps the backward grad variable to its corresponding forward variable.
    Return:
        new_input_grads (tuple(Value)): new results of backward op after decomposing.
        has_decomposed: whether the backward op has been successfully decomposed.
    '''

    # get the corresponding forward op according to grad_var_to_var
    # check and ensure: bwd_inputs = out_grads + fwd_inputs[optional] + fwd_outputs[optional]
    fwd_op = _get_fwd_op(bwd_op, grad_var_to_var)
    if not _check_op(fwd_op, bwd_op):
        logger.debug(
            f'{bwd_op.name()} can not be decomposed due to the mismatch between forward op and backward op'
        )
        return None, False
    if _check_prim_dynamic(fwd_op) or _check_prim_dynamic(bwd_op):
        return None, False

    # try to decompose backward op directly
    (
        new_grads,
        bwd_has_decomposed,
    ) = _decomp_bwd_with_vjp(
        block,
        fwd_op,
        bwd_op,
        grad_var_to_var,
    )

    if not bwd_has_decomposed:
        # try to decompose the forward op
        fwd_inputs = [x.source() for x in fwd_op.operands()]
        (
            new_fwd_outputs,
            fwd_has_decomposed,
        ) = _decomp_fwd_op(
            block,
            fwd_op,
            grad_var_to_var,
        )
        if fwd_has_decomposed:
            # try to decompose the backward op
            (
                new_grads,
                bwd_has_decomposed,
            ) = _decomp_bwd_without_vjp(
                block,
                bwd_op,
                grad_var_to_var,
                fwd_inputs,
                new_fwd_outputs,
            )
    return new_grads, bwd_has_decomposed


def _get_all_bwd_ops(pir_program):
    bwd_ops = []
    global_block = pir_program.global_block()
    for op in global_block.ops:
        if (
            op.name().endswith("_grad") or op.name().endswith("_grad_")
        ) and op.name() not in bwd_ops:
            bwd_ops.append(op.name())
    return bwd_ops


def _set_prim_state():
    state = []
    prev_fwd_prim_state = core._is_fwd_prim_enabled()
    prev_bwd_prim_state = core._is_bwd_prim_enabled()
    state.append(prev_fwd_prim_state)
    state.append(prev_bwd_prim_state)
    core._set_prim_forward_enabled(True)
    core._set_prim_backward_enabled(True)
    prev_pir_api_flag = paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
        "FLAGS_enable_pir_api"
    ]
    paddle.framework.set_flags(
        {"FLAGS_enable_pir_api": True}
    )  # set in pir mode for operator overloading
    paddle.base.framework.global_var._use_pir_api_ = True
    state.append(prev_pir_api_flag)
    return state


def _reset_prim_state(state):
    assert (
        len(state) == 3
    ), "state should contain fwd_prim_state, bwd_prim_state and pir_api_state"
    core._set_prim_forward_enabled(state[0])
    core._set_prim_backward_enabled(state[1])
    paddle.framework.set_flags({"FLAGS_enable_pir_api": state[2]})
    paddle.base.framework.global_var._use_pir_api_ = state[2]


def _translate_gradvartovar_to_pir(param_mapping, grad_var_to_var):
    '''translate grad_var_to_var (mapping VarDesc->VarDesc) to pir_grad_var_to_var (mapping Value->Value)'''
    pir_grad_var_to_var = ValueDict()
    for grad_var, var in grad_var_to_var.items():
        if grad_var in param_mapping.keys() and var in param_mapping.keys():
            if (
                len(param_mapping[grad_var]) == 1
                and len(param_mapping[var]) == 1
            ):
                new_grad_var = param_mapping[grad_var][0]
                new_var = param_mapping[var][0]
                pir_grad_var_to_var[new_grad_var] = new_var
            else:
                new_grad_vars = []
                new_vars = []
                if len(param_mapping[grad_var]) == 1:
                    new_grad_vars.append(param_mapping[grad_var][0])
                elif (
                    len(param_mapping[grad_var]) == 2
                    and param_mapping[grad_var][1].get_defining_op().name()
                    == "builtin.slice"
                ):
                    new_grad_vars.append(param_mapping[grad_var][1])
                else:
                    for i in range(0, len(param_mapping[grad_var])):
                        new_grad_vars.append(param_mapping[grad_var][i])

                if len(param_mapping[var]) == 1:
                    new_vars.append(param_mapping[var][0])
                elif (
                    len(param_mapping[var]) == 2
                    and param_mapping[var][1].get_defining_op().name()
                    == "builtin.slice"
                ):
                    new_vars.append(param_mapping[var][1])
                else:
                    last_op = param_mapping[var][-1].get_defining_op()
                    if last_op.name().endswith("_"):
                        new_vars.append(param_mapping[var][0])

                assert len(new_vars) == 1, "translate pir_grad_var_to_var error"
                for i in range(0, len(new_grad_vars)):
                    pir_grad_var_to_var[new_grad_vars[i]] = new_vars[0]
    return pir_grad_var_to_var


def _decomp_bwd_program(pir_program, pir_grad_var_to_var):
    '''Traverse and decompose all backward OPs in program'''
    with paddle.pir.core.program_guard(pir_program):
        bwd_ops = _get_all_bwd_ops(pir_program)
        undecomposed_bwd_ops = []
        ops = pir_program.global_block().ops
        for op in ops:
            bwd_op_name = op.name()
            if op.name() in bwd_ops:
                _, bwd_has_decomposed = _decomp_bwd_op(
                    pir_program.global_block(), op, pir_grad_var_to_var
                )
                if (
                    not bwd_has_decomposed
                    and bwd_op_name not in undecomposed_bwd_ops
                ):
                    undecomposed_bwd_ops.append(bwd_op_name)

    logger.debug(
        f'Following backward ops can not be decomposed: {undecomposed_bwd_ops}'
    )


def _decomp_fwd_program(pir_program, pir_grad_var_to_var):
    '''Traverse and decompose all forward OPs in program'''
    with paddle.pir.core.program_guard(pir_program):
        ops = pir_program.global_block().ops
        bwd_ops = _get_all_bwd_ops(pir_program)
        # ops including compile-time infermeta, causing mismatched input shape and output shape, which is unsupported when decomposing.
        black_fwd_ops = ["pd_op.stack", "pd_op.squeeze"]
        undecomposed_fwd_ops = []

        prev_op = None
        for op in ops:
            fwd_op_name = op.name()
            if op.name() not in bwd_ops:
                if op.name() not in black_fwd_ops:
                    _, fwd_has_decomposed = _decomp_fwd_op(
                        pir_program.global_block(),
                        op,
                        pir_grad_var_to_var,
                        prev_op,
                    )
                    if (
                        not fwd_has_decomposed
                        and fwd_op_name not in undecomposed_fwd_ops
                    ):
                        undecomposed_fwd_ops.append(fwd_op_name)
                else:
                    if fwd_op_name not in undecomposed_fwd_ops:
                        undecomposed_fwd_ops.append(fwd_op_name)
            prev_op = op if op.name() == "builtin.combine" else None

    logger.debug(
        f'Following forward ops can not be decomposed: {undecomposed_fwd_ops}'
    )


def decompose_dist_program(pir_program):
    '''
    Decompose all non-primitive ops into primitive ops in a pir program. It may contain forward ops and backward ops.
    '''
    # decomp forward composite ops
    decompose(pir_program, [])

    # decomp backward ops
    blacklist = core.prim_config["backward_blacklist"]

    block = pir_program.global_block()
    pre_combine_op = None
    with paddle.pir.core.program_guard(pir_program):
        ops = pir_program.global_block().ops
        for op in ops:
            bwd_op_name = op.name()
            if bwd_op_name.split(".")[-1] in blacklist:
                continue
            skip_decomp = False
            if has_decomp_vjp(op):
                if (
                    not core._enable_prim_dynamic_shape()
                ) and _check_prim_dynamic(op):
                    skip_decomp = True
                if not skip_decomp:
                    with pir_op_name_guard(op.name()), pir_op_role_guard(
                        op.op_role
                    ), pir_chunk_id_guard(op.chunk_id):
                        pir.set_insertion_point(op)
                        orig_outs = op.results()

                        is_next_split = False
                        decomp_outs = call_decomp_vjp(op)
                        for i in range(len(orig_outs)):
                            if orig_outs[i].has_one_use():
                                next_op = orig_outs[i].first_use().owner()
                                if next_op.name() == "builtin.split":
                                    is_next_split = True
                                    _check_op_results(
                                        next_op.name(),
                                        next_op.results(),
                                        decomp_outs[i],
                                    )
                                    next_op.replace_all_uses_with(
                                        decomp_outs[i]
                                    )
                                    block.remove_op(next_op)

                    if not is_next_split:
                        new_outs = _analyse_decomp_results(
                            orig_outs, decomp_outs, op
                        )
                        _check_op_results(op.name(), orig_outs, new_outs)
                        op.replace_all_uses_with(new_outs)

                    block.remove_op(op)

                if op.name() == "builtin.combine":
                    pre_combine_op = op

                if pre_combine_op is not None:
                    remove_op = True
                    for item in pre_combine_op.results():
                        if item.has_one_use():
                            remove_op = False
                            break
                    if remove_op:
                        block.remove_op(pre_combine_op)
                    pre_combine_op = None
    paddle.pir.set_insertion_point_to_block_end(block)


def decompose_pir_program(pir_program, param_mapping, grad_var_to_var):
    '''
    Decompose all PHI ops into prim ops in a pir program.
    Args:
        pir_program (Program): the program to be decomposed
        param_mapping (dict): a map of program variables to pir program values
        grad_var_to_var (dict): a dict obtained from distributed processing,
            which maps the backward grad variable to its corresponding forward variable.
    '''
    # set prim flags and pir_api flags
    state = _set_prim_state()
    # translate grad_var_to_var to pir
    pir_grad_var_to_var = _translate_gradvartovar_to_pir(
        param_mapping, grad_var_to_var
    )
    # decompose
    _decomp_bwd_program(pir_program, pir_grad_var_to_var)
    _decomp_fwd_program(pir_program, pir_grad_var_to_var)
    # reset prim flags and pir_api flags
    _reset_prim_state(state)
    return pir_grad_var_to_var


def get_inputs_from_data_and_parameter(pir_program):
    results = []
    for op in pir_program.global_block().ops:
        if op.name() == "pd_op.data":
            results.append(op.results()[0])
        if op.name() == "builtin.parameter":
            results.append(op.results()[0])
    return results


def get_outputs_from_fetch_op(pir_program):
    results = []
    for op in pir_program.global_block().ops:
        if op.name() == "pd_op.fetch":
            results.append(op.operand(0).source())
    return results


def get_grad_var_for_list(outputs, pir_grad_var_to_var):
    results = []
    var2grad_var = ValueDict()
    for k, v in pir_grad_var_to_var.items():
        var2grad_var[v] = k
    for output in outputs:
        results.append(var2grad_var[output])
    return results


def get_defining_op_indices(program, output_values):
    def getIdx(op):
        for idx, op_iter in enumerate(program.global_block().ops):
            if op == op_iter:
                return idx
        raise RuntimeError("op not found in program")

    results = []
    for output in output_values:
        results.append(getIdx(output.get_defining_op()))
    return results


def get_forward_op_idxs(program, is_forward_op_func):
    def getIdx(op):
        for idx, op_iter in enumerate(program.global_block().ops):
            if op == op_iter:
                return idx
        raise RuntimeError("op not found in program")

    results = []
    for op in program.global_block().ops:
        if is_forward_op_func(op):
            results.append(getIdx(op))
    return results


def auto_recompute_pir_program(pir_program, is_forward_op_func=None):
    DebugPrint("Start Recompute Pir Program:")
    DebugPrint("Before Recompute: ", pir_program)
    # prepare essential inputs for auto_recompute
    inputs = get_inputs_from_data_and_parameter(pir_program)
    outputs = get_outputs_from_fetch_op(pir_program)
    fwd_op_end_idx = -1
    if len(outputs):
        fwd_op_end_idx = max(get_defining_op_indices(pir_program, outputs))

    logger = logging.getLogger("auto-recompute")
    logger.setLevel(logging.INFO)

    if is_forward_op_func is not None:
        try:
            fwd_op_end_idx = max(
                get_forward_op_idxs(pir_program, is_forward_op_func)
            )
        except:
            logger.info("No Forward Ops Found!")

    if fwd_op_end_idx == -1:
        logger.info("Skip Auto Recompute!")
        return pir_program
    backward_op_start_idx = fwd_op_end_idx + 1

    program, _ = auto_recompute(
        pir_program,
        inputs,
        outputs,
        [],
        fwd_op_end_idx,
        backward_op_start_idx,
    )

    return program
