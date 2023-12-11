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

import collections
import logging
from collections.abc import Sequence

import paddle.pir
from paddle.autograd.backward_utils import State, dynamic_shape_prim_vjp_guard
from paddle.base.libpaddle.pir import (
    build_pipe_for_block,
    get_used_external_value,
)

"""
    grad: for templete test, will combine in paddle.grad .
    calc_gradient: for internal use, optest, parallel etc .
    calc_gradient_helper: for dygraph to static .
"""
__all__ = ['grad', 'calc_gradient', 'calc_gradient_helper']


def check_type(input, input_name, expected_type, op_name, extra_message=''):
    if not isinstance(input, expected_type):
        raise TypeError(
            f"The type of '{input_name}' in {op_name} must be {expected_type}, but received {type(input)}. {extra_message}"
        )


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, Sequence) else [x]


def check_all_puts(block, inputs, outputs):
    for output in outputs:
        if output.get_defining_op().get_parent_block() != block:
            raise ValueError("all outputs must be in the same block")
    for input in inputs:
        if input.get_defining_op().get_parent_block() != block:
            raise ValueError(
                "all inputs must be in the same block with outputs"
            )


def append_full_like(float_value, value, state, backward_ops):
    value_grad = paddle.full_like(
        value,
        float_value,
        dtype=value.dtype,
    )
    full_like_op = value_grad.get_defining_op()
    full_op = full_like_op.operand_source(1).get_defining_op()
    update_bwdop_structure(
        backward_ops,
        state.op_to_opgrad[value.get_defining_op()],
        [full_like_op, full_op],
    )
    state.value_to_valuegrad[value] = [[value_grad]]
    return value_grad


def get_real_op_inputs(op):
    if op.name() in ["pd_op.if", "pd_op.while"]:
        return get_used_external_value(op)
    else:
        return op.operands_source()


def update_no_grad_set_by_stopgradient(block, no_grad_set):
    for op in block.ops:
        for value in op.results():
            if value.stop_gradient and value not in no_grad_set:
                no_grad_set.add(value)


def update_bwdop_structure(backward_ops, op_to_opgrad_list, grad_op_list):
    for grad_op in grad_op_list:
        backward_ops.append(grad_op)
        op_to_opgrad_list.append(grad_op)


def prepare_grad_outputs(grad_outputs, outputs, state):
    """
    if grad_outputs is none, add fill_1 op to create grad_outputs,
    else check whether outputs shape and dtype is same to grad_outputs, otherwise raise error.

    if only part of op's outputs in outputs, add fill_0 op to create other grad_outputs.
    eg: split.

    update value_to_valuegrad and op_to_opgrad.

    return complete_outputs and complete_gradoutputs, backward_ops.

    """
    if not grad_outputs:
        grad_outputs = [None] * len(outputs)

    if len(grad_outputs) != len(outputs):
        raise ValueError(
            "grad_outputs should have the same length of as outputs."
        )
    backward_ops = []
    for i, grad in enumerate(grad_outputs):
        output = outputs[i]
        # fwd : op1 -> op2 -> op3 -> output
        # bwd : op1G <- op2G <- op3G <- outputG <- full_likeop/feedop
        if grad is None:
            append_full_like(1.0, output, state, backward_ops)
        else:
            if output.shape != grad.shape:
                raise ValueError(
                    "The shape of grad_output[%d] %s should be the same as the shape of output[%d] %s"
                    % (i, str(grad.shape), i, str(output.shape))
                )
            if output.dtype != grad.dtype:
                raise ValueError(
                    "The dtype of grad_output[%d] %s should be the same as the dtype of output[%d] %s"
                    % (i, str(grad.dtype), i, str(output.dtype))
                )
            feedop = grad.get_defining_op()
            update_bwdop_structure(
                backward_ops,
                state.op_to_opgrad[output.get_defining_op()],
                [feedop],
            )
            state.value_to_valuegrad[output] = [[grad]]
    # add input for bwd first op
    complete_outputs = outputs
    complete_gradoutputs = grad_outputs

    visited_output = set()
    for output in outputs:
        if output in visited_output:
            continue
        for opresult in output.get_defining_op().results():
            if opresult in state.value_to_valuegrad:
                visited_output.add(opresult)
                continue
            else:
                if paddle.pir.is_fake_op_result(opresult):
                    state.value_to_valuegrad[opresult] = [
                        [paddle.pir.fake_op_result()]
                    ]
                else:
                    grad_value = append_full_like(
                        0.0, opresult, state, backward_ops
                    )
                    visited_output.add(opresult)

                    complete_outputs.append(opresult)
                    complete_gradoutputs.append(grad_value)

    return complete_outputs, complete_gradoutputs, backward_ops


def some_in_set(value_list, value_set):
    def operand2value(values):
        value_set = set()
        for item in values:
            if isinstance(item, paddle.pir.OpOperand):
                value_set.add(item.source())
            else:
                value_set.add(item)
        return value_set

    if operand2value(value_list) & operand2value(value_set):
        return True
    else:
        return False


def prune_ops(total_ops, inputs_set, outputs_set, no_grad_set):
    '''
    prune ops which do not in the path from inputs_set to outputs_set,
    prune ops which do not in the path from outputs_set to inputs_set,

    pruned op in total_ops is uneffective_ops, else is effective_ops

    '''
    intersection_op_flags = [True] * len(total_ops)
    union_op_flags = [False] * len(total_ops)
    # from input to output
    if inputs_set:
        for i, op in enumerate(total_ops):
            if some_in_set(op.results(), inputs_set):
                union_op_flags[i] = True
                continue

            if some_in_set(get_real_op_inputs(op), inputs_set):
                union_op_flags[i] = True
                for value in op.results():
                    if value not in no_grad_set:
                        inputs_set.add(value)
            else:
                intersection_op_flags[i] = False

    # from output to input
    for i, op in reversed(list(enumerate(total_ops))):
        if some_in_set(op.results(), outputs_set):
            union_op_flags[i] = True
            for operand in get_real_op_inputs(op):
                if operand not in no_grad_set:
                    outputs_set.add(operand)
        else:
            union_op_flags[i] = False
            intersection_op_flags[i] = False

    # some inputs in no_grad_set but its next op is effective,
    # add their defining op here.
    total_ops_list = list(total_ops)
    for i, op in enumerate(total_ops_list):
        if union_op_flags[i] is False:
            for result in op.results():
                if result.has_one_use():
                    next_op = result.first_use().owner()
                    if (
                        next_op in total_ops
                        and union_op_flags[total_ops_list.index(next_op)]
                        is True
                    ):
                        union_op_flags[i] = True
                else:
                    continue

    effective_ops = [
        total_ops[i] for i in range(len(total_ops)) if intersection_op_flags[i]
    ]
    uneffective_ops = [
        total_ops[i]
        for i in reversed(range(len(total_ops)))
        if not union_op_flags[i]
    ]

    return effective_ops, uneffective_ops


def update_no_grad_set_after_prune(
    block, effective_forward_ops, no_grad_set, inputs, outputs
):
    '''
    update no_grad_set after forward prune

    from inputs to outputs add value not in the path to no_grad_set,
    from outputs to inputs add value not in the path to no_grad_set,
    '''
    inputs_set = set(inputs)
    if inputs_set:
        for op in block.ops:
            if some_in_set(get_real_op_inputs(op), inputs_set):
                for value in op.results():
                    if value not in no_grad_set:
                        inputs_set.add(value)

        for op in effective_forward_ops:
            for value in get_real_op_inputs(op):
                if value not in inputs_set:
                    no_grad_set.add(value)

    outputs_set = set(outputs)
    no_grad_set_tmp = set()
    for op in reversed(effective_forward_ops):
        for output in op.results():
            if output not in outputs_set and not some_in_set(
                [output], set(get_real_op_inputs(op))
            ):
                no_grad_set_tmp.add(output)

        for input in get_real_op_inputs(op):
            if input not in no_grad_set:
                outputs_set.add(input)

    no_grad_set.update(no_grad_set_tmp)


def inverse_sort_op(ops):
    '''
    if topo graph is op1 -> op2 -> op3
    return [op3, op2, op1]

    '''

    # init pending_count[op] which descibes number of
    # pending edges for its grad_op

    pending_count = collections.defaultdict(int)
    ops_set = set(ops)
    sorted_list = []
    for op in ops:
        for x in get_real_op_inputs(op):
            if x and x.get_defining_op() in ops_set:
                pending_count[x.get_defining_op()] += 1

    queue = collections.deque()

    for op in ops:
        if pending_count[op] == 0:
            queue.append(op)

    while queue:
        op = queue.popleft()
        sorted_list.append(op)

        for x in get_real_op_inputs(op):
            x_op = x.get_defining_op()
            pending_count[x_op] -= 1
            if pending_count[x_op] == 0:
                queue.append(x_op)

    if len(sorted_list) != len(ops):
        raise ValueError(
            "inverse_sort_op wrong, sorted_list size is not equal to origin_list size"
        )

    return sorted_list


def append_backward_ops(
    base_op,
    base_inputs,
    base_input_grads,
    fwd_block,
    bwd_block,
    effective_forward_ops,
    no_grad_set,
    backward_ops,
    state,
    inside_value_to_outside_value_map,
):
    '''
    add grad_op in order of topological inverse sort
        eg:
        from :op1 -> v1 -> op2 -> v2 -> op3 -> v3
        to: og1_g <- v1_g <- op2_g <- v2_g <- op3_g <- v3_g

    if op has grad_op, prepare its grad_op's inputs by value_to_valuegrad,
        eg:
        value_to_valuegrad[v3] = [[v3_g]];
        v2_g = call_vjp(op3, [[v2]], [[v3]],[[v3_g]], [[v2_stopgradient]])


    special pattern 1:
        v11 -> combine_op -> v1 -> op -> v3
        v12 ->
                             v2 ->
        value_to_valuegrad[v3] = [[v3_g]]

        v1 is inside python api, we don't describe it in backward process(state)
        so v1_grad is inside vjp, we don't describe it in backward process(state)
        [[v11_g, v12_g], v2_g] = call_vjp(combine_op, [[v11, v12]], [[v3]],[[v3_g]], [[v11_stopgradient, v12_stopgradient], v2_stop_gradient])


        op_vjp is:
        v11_g <- split_op <- v1_g <- op_g <- v3_g
        v12_g <-
                             v2_g <-

        value_to_valuegrad[v11] = [[v11_g]]
        value_to_valuegrad[v12] = [[v12_g]]
        value_to_valuegrad[v2] = [[v2_g]]

    if op don't has grad_op:
        if it don't has input and it's output has more than
        one output_grad, add sumop for grad aggregation.
        (eg: full op and parameter op etc.)

        else continue to next op.
    '''

    def append_add_n(value):
        # one value is input of more than one fwd_op,
        # so more than one bwd_op create input_grad,
        # need add sum op to accumulate gradient
        add_n_value = paddle.add_n(
            [item[0] for item in state.value_to_valuegrad[value]]
        )
        add_n_op = add_n_value.get_defining_op()
        combine_op = add_n_op.operand_source(0).get_defining_op()
        update_bwdop_structure(
            backward_ops, state.op_to_opgrad[op], [combine_op, add_n_op]
        )

        for tmp in state.value_to_valuegrad[value]:
            state.value_to_sumvaluegrad[value].append(tmp)
        state.value_to_valuegrad[value] = [[add_n_value]]

    def make_output_with_output_grad(op):
        zero_flag = [False] * op.num_results()
        outputs = []
        output_grads = []
        for i, value in enumerate(op.results()):
            new_value = (
                [control_flow_value_to_copyvalue_map[value]]
                if value in control_flow_value_to_copyvalue_map
                else [value]
            )
            while value in inside_value_to_outside_value_map:
                value = inside_value_to_outside_value_map[value]

            if (
                value in state.value_to_valuegrad
                and len(state.value_to_valuegrad[value]) > 1
            ):
                append_add_n(value)

            if (
                value not in state.value_to_valuegrad
                or state.value_to_valuegrad[value] == []
            ):
                if (
                    not value.use_empty()
                    and value.first_use().owner().name() == "builtin.split"
                ):
                    # pattern case:
                    # this fwd_op's output is vectorType, it will split to
                    # Type by builtin.split op, so need get from split op's ouput
                    (
                        split_zero_flag,
                        split_outputs,
                        split_output_grad,
                    ) = make_output_with_output_grad(value.first_use().owner())
                    zero_flag[i] = all(split_zero_flag)
                    grad_values = [value[0] for value in split_output_grad]
                    state.value_to_valuegrad[value] = [grad_values]
                    new_value = [info[0] for info in split_outputs]
                else:
                    # first case:
                    # this fwd_op's output didn't used by other fwd_op,
                    # so no output_grad created.

                    # second case:
                    # last bwd_op return None because input in no_grad_set,
                    # but this bwd_op need a input.

                    append_full_like(0.0, value, state, backward_ops)
                    zero_flag[i] = True

            outputs.append(new_value)
            output_grads.append(state.value_to_valuegrad[value][0])

        return zero_flag, outputs, output_grads

    def make_input_with_input_stopgradient(op):
        inputs = []
        input_grad_stopgradients = []
        if op.name() in [
            "builtin.combine",
            "pd_op.if",
            "pd_op.while",
            "cf.tuple_push",
        ]:
            grad_semantic_info = [
                True for _ in range(len(get_real_op_inputs(op)))
            ]
        else:
            grad_semantic_info = op.get_input_grad_semantics()

        for input, grad_semantic in zip(
            get_real_op_inputs(op), grad_semantic_info
        ):
            if not grad_semantic:
                if (
                    input.get_defining_op() is not None
                    and input.get_defining_op().name() == "builtin.combine"
                ):
                    tmp_input = []
                    for tmp in input.get_defining_op().operands_source():
                        tmp_input.append(
                            control_flow_value_to_copyvalue_map[tmp]
                            if tmp in control_flow_value_to_copyvalue_map
                            else tmp
                        )

                    inputs.append(tmp_input)
                else:
                    tmp_input = (
                        [control_flow_value_to_copyvalue_map[input]]
                        if input in control_flow_value_to_copyvalue_map
                        else [input]
                    )
                    inputs.append(tmp_input)
                continue

            if (
                input.get_defining_op() is not None
                and input.get_defining_op().name() == "builtin.combine"
            ):
                (
                    combine_inputs,
                    combine_stop_gradient,
                ) = make_input_with_input_stopgradient(input.get_defining_op())
                inputs.append([info[0] for info in combine_inputs])
                input_grad_stopgradients.append(
                    [info[0] for info in combine_stop_gradient]
                )
            else:
                tmp_input = (
                    [control_flow_value_to_copyvalue_map[input]]
                    if input in control_flow_value_to_copyvalue_map
                    else [input]
                )
                inputs.append(tmp_input)
                if input.get_defining_op() is None or input in no_grad_set:
                    input_grad_stopgradients.append([True])
                else:
                    input_grad_stopgradients.append([False])

        return inputs, input_grad_stopgradients

    def update_input_grad_map(op, input_grads, origin_inputs):
        i = 0
        if (
            op.name() == "builtin.combine"
            or op.name() == "pd_op.if"
            or op.name() == "pd_op.while"
        ):
            grad_semantic_info = [True for _ in range(len(origin_inputs))]
        else:
            grad_semantic_info = op.get_input_grad_semantics()
        for input, grad_semantic in zip(origin_inputs, grad_semantic_info):
            if not grad_semantic:
                continue
            if (
                input.get_defining_op() is not None
                and input.get_defining_op().name() == "builtin.combine"
            ):
                update_input_grad_map(
                    input.get_defining_op(),
                    input_grads[i],
                    input.get_defining_op().operands_source(),
                )
            else:
                input_grad = input_grads[i]
                if isinstance(input_grad, list):
                    state.value_to_valuegrad[input].append(input_grad)
                else:
                    state.value_to_valuegrad[input].append([input_grad])
            i += 1

    def append_yield(block, base_inputs, base_inputs_grad):
        with block:
            inputs_grad = []
            for value, value_grad in zip(base_inputs, base_inputs_grad):
                if value_grad is None:
                    continue

                while value in inside_value_to_outside_value_map:
                    value = inside_value_to_outside_value_map[value]

                if value in state.value_to_valuegrad:
                    if len(state.value_to_valuegrad[value]) > 1:
                        append_add_n(value)
                    inputs_grad.append(state.value_to_valuegrad[value][0][0])
                else:
                    value_grad = append_full_like(
                        0.0, value, state, backward_ops
                    )
                    inputs_grad.append(value_grad)
            paddle.base.libpaddle.pir.cf_yield(inputs_grad)

    # there are four patterns:
    # [builtin.combine , op1] (op1's one input is vectorType, outputs are not vectorType)
    # [op2 , builtin.split] (op2's inputs are not vectorType, one output is vectorType)
    # [builtin.combine , op3 , buitin.split] (op3's one input and one output are vectorType)
    # [op4] (op4's inputs and outputs are not vectorType)

    # -----------------only for control flow-----------------#
    # tuple_push value to pop value
    control_flow_value_to_copyvalue_map = {}
    # tuple_push value to pop value
    control_flow_copyvalue_to_value_map = {}

    if (
        len(effective_forward_ops) > 1
        and effective_forward_ops[-1].name() == "cf.yield"
    ):
        yield_op = effective_forward_ops[-1]
        for outside_output, inside_output in zip(
            base_op.results(), yield_op.operands_source()
        ):
            inside_value_to_outside_value_map[inside_output] = outside_output
        forward_ops = effective_forward_ops[:-1]
    else:
        forward_ops = effective_forward_ops

    inverse_effective_forward_ops = inverse_sort_op(forward_ops)
    clear_effective_forward_ops = []
    for op in inverse_effective_forward_ops:
        if op.name() != "builtin.combine" and op.name() != "builtin.split":
            clear_effective_forward_ops.append(op)
    with bwd_block:
        for op in clear_effective_forward_ops:
            if paddle.framework.core.has_vjp(op):
                # prepare output_grad
                zero_flag, outputs, output_grads = make_output_with_output_grad(
                    op
                )

                # prepare input_grad stop_gradient info.
                (
                    inputs,
                    input_grad_stopgradients,
                ) = make_input_with_input_stopgradient(op)

                if op.name() == "cf.tuple_push":
                    with dynamic_shape_prim_vjp_guard(op, inputs):
                        copy_out = paddle.framework.core.call_vjp(
                            op,
                            inputs,
                            outputs,
                            output_grads,
                            input_grad_stopgradients,
                        )
                    pop_op = bwd_block.ops[-1]
                    bwd_ops = [pop_op]
                    for output, copy_output in zip(inputs[1:], copy_out[1:]):
                        control_flow_value_to_copyvalue_map[
                            output[0]
                        ] = copy_output[0]
                        control_flow_copyvalue_to_value_map[
                            copy_output[0]
                        ] = output[0]

                else:
                    # all(zero_flag) support this op has no contribution for grad
                    # should be delete (prune sub_graph)
                    if len(output_grads) == 0 or all(zero_flag):
                        continue

                    if op.name() == "pd_op.if" or op.name() == "pd_op.while":
                        origin_inputs = get_used_external_value(op)
                        for sub_block in op.blocks():
                            build_pipe_for_block(sub_block)
                        with dynamic_shape_prim_vjp_guard(op, inputs):
                            input_grads = paddle.framework.core.call_vjp(
                                op,
                                inputs,
                                outputs,
                                output_grads,
                                input_grad_stopgradients,
                            )
                        grad_op = bwd_block.ops[-1]
                        bwd_ops = [grad_op]

                        for sub_fwd_block, sub_bwd_block in zip(
                            op.blocks(), grad_op.blocks()
                        ):
                            sub_state = state.copy(sub_fwd_block)
                            sub_inside_value_to_outside_value_map = (
                                inside_value_to_outside_value_map.copy()
                            )
                            sub_backward_ops = []
                            append_backward_ops(
                                op,
                                [input[0] for input in inputs],
                                [input_grad[0] for input_grad in input_grads],
                                sub_fwd_block,
                                sub_bwd_block,
                                sub_fwd_block.ops,
                                no_grad_set,
                                sub_backward_ops,
                                sub_state,
                                sub_inside_value_to_outside_value_map,
                            )
                        # update input_grad map
                        update_input_grad_map(op, input_grads, origin_inputs)
                    else:
                        # create grad_op
                        before_ops_num = len(bwd_block.ops)
                        with dynamic_shape_prim_vjp_guard(op, inputs):
                            input_grads = paddle.framework.core.call_vjp(
                                op,
                                inputs,
                                outputs,
                                output_grads,
                                input_grad_stopgradients,
                            )
                        after_ops_num = len(bwd_block.ops)

                        # update grad_op structure
                        bwd_ops = [
                            bwd_block.ops[i]
                            for i in range(before_ops_num, after_ops_num)
                        ]

                        # update input_grad map
                        update_input_grad_map(
                            op, input_grads, op.operands_source()
                        )

                update_bwdop_structure(
                    backward_ops, state.op_to_opgrad[op], bwd_ops
                )

            else:
                if op.num_operands() == 0 and op.num_results() != 0:
                    for value in op.results():
                        if len(state.value_to_valuegrad[value]) > 1:
                            append_add_n(value)
                        else:
                            state.op_to_opgrad[op] = []
                else:
                    logging.warning("%s op has no grad op", op.name())
                    state.op_to_opgrad[op] = []

        if fwd_block != bwd_block:
            append_yield(bwd_block, base_inputs, base_input_grads)


def prepare_backward_prune_set(inputs, outputs):
    outputs_fwd_set = set()
    for input_ in inputs:
        if not input_.use_empty():
            for item in get_real_op_inputs(input_.first_use().owner()):
                outputs_fwd_set.add(item)
        else:
            logging.warning("input privided by inputs has no use")

    inputs_fwd_set = set()
    for output in outputs:
        inputs_fwd_set.add(output)

    return outputs_fwd_set, inputs_fwd_set


def create_backward_prune_set(
    outputs_fwd_set, inputs_fwd_set, no_grad_set, state
):
    outputs_set = set()
    for item in outputs_fwd_set:
        if state.value_to_valuegrad[item] != []:
            outputs_set.add(state.value_to_valuegrad[item][0][0])

    inputs_set = set()
    for item in inputs_fwd_set:
        if state.value_to_valuegrad[item] != []:
            inputs_set.add(state.value_to_valuegrad[item][0][0])

    inputs_set_tmp = set()
    for out_grad in inputs_set:
        if not out_grad.use_empty():
            for item in get_real_op_inputs(out_grad.first_use().owner()):
                inputs_set_tmp.add(item)
    inputs_set.update(inputs_set_tmp)

    no_gradvar_set = set()  # grad_value of value in no_grad_set
    for key in state.value_to_valuegrad:
        if key in no_grad_set and state.value_to_valuegrad[key] != []:
            no_gradvar_set.add(state.value_to_valuegrad[key][0][0])
    for key in state.value_to_sumvaluegrad:
        if key in no_grad_set:
            for item in state.value_to_sumvaluegrad[key][0]:
                no_gradvar_set.add(item)

    return outputs_set, inputs_set, no_gradvar_set


def remove_op(block, op, state):
    '''
    remove op from block
    '''
    block.remove_op(op)
    if state.opgrad_to_op[op] != []:
        fwd_op = state.opgrad_to_op[op][0]
        state.op_to_opgrad[fwd_op].remove(op)

    for valuegrad in op.results():
        if state.valuegrad_to_value[valuegrad] != []:
            value = state.valuegrad_to_value[valuegrad][0]
            state.value_to_valuegrad[value] = []

            if value in state.sumvaluegrad_to_value:
                raise ValueError(
                    'input_grad in [%s] is value which need to sum ', op.name()
                )


def calc_gradient_helper(outputs, inputs, grad_outputs, no_grad_set):
    block = outputs[0].get_defining_op().get_parent_block()
    state = State(block)

    # check all inputs and outputs in the same block
    check_all_puts(block, inputs, outputs)
    # update no_grad_set if some value stop_gradient=True
    update_no_grad_set_by_stopgradient(block, no_grad_set)
    complete_outputs, _, backward_ops = prepare_grad_outputs(
        grad_outputs, outputs, state
    )

    inputs_set = set(inputs)
    outputs_set = set(complete_outputs)
    effective_forward_ops, _ = prune_ops(
        block.ops, inputs_set, outputs_set, no_grad_set
    )
    update_no_grad_set_after_prune(
        block, effective_forward_ops, no_grad_set, inputs, complete_outputs
    )

    outputs_fwd_set, inputs_fwd_set = prepare_backward_prune_set(
        inputs, complete_outputs
    )

    # sub_block op output to parent_block op output
    inside_value_to_outside_value_map = {}

    append_backward_ops(
        None,
        None,
        None,
        block,
        block,
        effective_forward_ops,
        no_grad_set,
        backward_ops,
        state,
        inside_value_to_outside_value_map,
    )
    # now value_to_valuegrad should be value <-> value (add sum op for the same values's gradvalue)
    outputs_set, inputs_set, no_gradvar_set = create_backward_prune_set(
        outputs_fwd_set, inputs_fwd_set, no_grad_set, state
    )
    _, remove_ops = prune_ops(
        backward_ops, inputs_set, outputs_set, no_gradvar_set
    )

    state.turn_map()

    for bwd_op in inverse_sort_op(remove_ops):
        if bwd_op.result(0) in grad_outputs:
            continue
        if bwd_op.result(0).use_empty():
            remove_op(block, bwd_op, state)
    state.turn_map()

    input_grad_map = state.value_to_valuegrad

    return input_grad_map


def calc_gradient(outputs, inputs, grad_outputs, no_grad_set):
    """
    caclulate gradient of input

    Args:
        outputs (Value|list(Value)|tuple(Value)): the output Value or
            Value list/tuple of the graph to compute gradients.
        inputs (Value|list(Value)|tuple(Value)): the input Value or
            Value list/tuple of the graph to compute gradients. The returned
            values of this API are the gradients of `inputs` .
        grad_outputs (Value|list(Value|None)|tuple(Value|None), optional):
            initial gradient values of `outputs` . If `grad_outputs` is None,
            the initial gradient values of `outputs` would be Values filled with 1;
            if `grad_outputs` is not None, it must have the same length as `outputs` ,
            and in this case, the initial gradient value of the i-th `outputs` would
            be: (1) a Value filled with 1 when the i-th element of `grad_outputs`
            is None; (2) the i-th element of `grad_outputs` when the i-th element of
            `grad_outputs` is a Value. Default None.
        no_grad_set (set(Value), optional):
            the Values whose gradients are not needed to compute. Default None.

    Return:
        list[Value]:A list of gradients for inputs
        If an input does not affect targets, the corresponding gradient Tensor
        will be None
        TODO if allow_unused=False raise TypeError() if input_grad has None
    """
    # record input value and its gradient (Value to Value)
    input_to_inputgrad_map = calc_gradient_helper(
        outputs, inputs, grad_outputs=grad_outputs, no_grad_set=no_grad_set
    )

    inputgrad = []
    for input in inputs:
        inputgrad.append(
            input_to_inputgrad_map[input][0][0]
            if input_to_inputgrad_map[input] != []
            else None
        )
    return inputgrad


def grad(
    outputs,
    inputs,
    grad_outputs=None,
    retain_graph=None,
    create_graph=False,
    only_inputs=True,
    allow_unused=False,
    no_grad_vars=None,
):
    '''
    .. note::
        **This API is ONLY available in imperative mode.**

    This API computes the sum of gradients of `outputs` with respect to each `inputs` .

    Parameters:
        outputs (Value|list(Value)|tuple(Value)): the output Value or
            Value list/tuple of the graph to compute gradients.
        inputs (Value|list(Value)|tuple(Value)): the input Value or
            Value list/tuple of the graph to compute gradients. The returned
            values of this API are the gradients of `inputs` .
        grad_outputs (Value|list(Value|None)|tuple(Value|None), optional):
            initial gradient values of `outputs` . If `grad_outputs` is None,
            the initial gradient values of `outputs` would be Values filled with 1;
            if `grad_outputs` is not None, it must have the same length as `outputs` ,
            and in this case, the initial gradient value of the i-th `outputs` would
            be: (1) a Value filled with 1 when the i-th element of `grad_outputs`
            is None; (2) the i-th element of `grad_outputs` when the i-th element of
            `grad_outputs` is a Value. Default None.
        retain_graph (bool, optional): whether to retain the forward graph which
            is used to calculate the gradient. When it is True, the graph would
            be retained, in which way users can calculate backward twice for the
            same graph. When it is False, the graph would be freed. Default None,
            which means it is equal to `create_graph` .
        create_graph (bool, optional): whether to create the gradient graphs of
            the computing process. When it is True, higher order derivatives are
            supported to compute; when it is False, the gradient graphs of the
            computing process would be discarded. Default False.
        only_inputs (bool, optional): whether to only compute the gradients of
            `inputs` . If it is False, the gradients of all remaining leaf
            Values in the graph would be also computed and accumulated.
            If it is True, only the gradients of `inputs` would be computed.
            Default True. only_inputs=False is under development, and it is
            not supported yet.
        allow_unused (bool, optional): whether to raise error or return None if some
            Values of `inputs` are unreachable in the graph. If some Values of
            `inputs` are unreachable in the graph (i.e., their gradients are None),
            error would be raised if allow_unused=False, or None would be returned as
            their gradients if allow_unused=True. Default False.
        no_grad_vars (Value|list(Value)|tuple(Value)|set(Value), optional):
            the Values whose gradients are not needed to compute. Default None.

    Returns:
        list: a list of Values, whose length is the same as the Value number
        inside `inputs`, and the i-th returned Value is the sum of gradients of
        `outputs` with respect to the i-th `inputs`.
    '''
    check_type(
        outputs,
        'outputs',
        ((paddle.pir.Value, paddle.pir.OpResult), list, tuple),
        'paddle.autograd.ir_backward.grad',
    )
    check_type(
        inputs,
        'inputs',
        ((paddle.pir.Value, paddle.pir.OpResult), list, tuple),
        'paddle.autograd.ir_backward.grad',
    )
    check_type(
        grad_outputs,
        'grad_outputs',
        ((paddle.pir.Value, paddle.pir.OpResult), list, tuple, type(None)),
        'paddle.autograd.ir_backward.grad',
    )

    check_type(
        no_grad_vars,
        'no_grad_vars',
        ((paddle.pir.Value, paddle.pir.OpResult), list, tuple, set, type(None)),
        'paddle.autograd.ir_backward.grad',
    )
    outputs = _as_list(outputs)
    inputs = _as_list(inputs)
    grad_outputs = _as_list(grad_outputs)
    if no_grad_vars is None:
        no_grad_set = set()
    elif no_grad_vars is not set:
        no_grad_set = set(no_grad_vars)
    else:
        no_grad_set = no_grad_vars

    input_grad = calc_gradient(outputs, inputs, grad_outputs, no_grad_set)

    return input_grad


# only for test
def append_backward(loss, parameter_list=None, no_grad_set=None):
    '''
    Parameters:
        loss (Value): The loss Tensor of the network
        parameter_list (Value|list(Value)|tuple(Value)):  List/Tuple of Parameters
            that need to be updated by optimizers.
            If it is None, all parameters
            will be updated.
            Default: None.
        no_grad_vars (Value|list(Value)|tuple(Value)|set(Value), optional):
            the Values whose gradients are not needed to compute. Default None.

    Returns:
        list of tuple (Value): Pairs of parameter and its corresponding gradients.
        The key is the parameter and the value is gradient Tensor.
    '''

    check_type(
        loss,
        'loss',
        (paddle.pir.Value, paddle.pir.OpResult),
        'paddle.autograd.ir_backward.append_backward',
    )

    if parameter_list is not None:
        check_type(
            parameter_list,
            'parameter_list',
            (list, tuple, set),
            'paddle.autograd.ir_backwardappend_backward',
        )
        for i, param in enumerate(parameter_list):
            check_type(
                param,
                'parameter_list[%s]' % i,
                (paddle.pir.Value, paddle.pir.OpResult),
                'base.backward.append_backward',
            )

    else:
        parameter_list = (
            loss.get_defining_op().get_parent_block().all_parameters()
        )

    inputs_grad = paddle.autograd.ir_backward.grad(loss, parameter_list)

    input_inputs_grad = []
    for input, input_grad in zip(parameter_list, inputs_grad):
        input_inputs_grad.append((input, input_grad))

    return input_inputs_grad
