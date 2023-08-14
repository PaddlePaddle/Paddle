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
from collections.abc import Sequence

import paddle.ir
from paddle.autograd.backward_utils import State

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


def update_no_grad_set_by_stopgradient(block, no_grad_set):
    for op in block.ops:
        for opresult_idx in range(op.num_results()):
            value = op.result(opresult_idx)
            if value.stop_gradient and value not in no_grad_set:
                no_grad_set.add(value)


def update_bwdop_structure(backward_ops, op_to_opgrad_list, grad_op):
    backward_ops.append(grad_op)
    op_to_opgrad_list.append(grad_op)


def prepare_grad_outputs(
    block, grad_outputs, outputs, value_to_valuegrad, op_to_opgrad
):
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
        # bwd : op1G <- op2G <- op3G <- outputG <- fillop/feedop
        if grad is None:
            output_grad = paddle.full(
                output.shape,
                1.0,
                dtype=output.dtype,
            )
            fillop = output_grad.get_defining_op()

            update_bwdop_structure(
                backward_ops,
                op_to_opgrad[output.get_defining_op()],
                fillop,
            )
            value_to_valuegrad[output] = [[output_grad]]
        else:
            if output.shape != grad.shape:
                raise ValueError(
                    "The shape of grad_output[%d] should be the same as the shape of output[%d]"
                    % (i, i)
                )
            if output.dtype != grad.dtype:
                raise ValueError(
                    "The dtype of grad_output[%d] should be the same as the dtype of output[%d]"
                    % (i, i)
                )
            feedop = grad.get_defining_op()
            update_bwdop_structure(
                backward_ops, op_to_opgrad[output.get_defining_op()], feedop
            )
            value_to_valuegrad[output] = [[grad]]

    # add input for bwd first op
    complete_outputs = outputs
    complete_gradoutputs = grad_outputs

    visited_output = set()
    for output in outputs:
        if output in visited_output:
            continue
        for opresult in output.get_defining_op().results():
            if opresult in value_to_valuegrad:
                visited_output.add(opresult)
                continue
            else:
                grad_value = paddle.full(
                    opresult.shape,
                    0.0,
                    opresult.dtype,
                )
                fillop = grad.get_defining_op()

                update_bwdop_structure(
                    backward_ops,
                    op_to_opgrad[opresult.get_defining_op()],
                    fillop,
                )
                value_to_valuegrad[opresult] = [grad_value]

                visited_output.add(opresult)

                complete_outputs.append(opresult)
                complete_gradoutputs.append(grad_value)

    return complete_outputs, complete_gradoutputs, backward_ops


def some_in_set(value_list, value_set):
    def operand2value(values):
        value_set = set()
        for item in values:
            if isinstance(item, paddle.ir.OpOperand):
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
    relevant_op_flags = [True] * len(total_ops)

    # from input to output
    if inputs_set:
        for i, op in enumerate(total_ops):
            if some_in_set(op.results(), inputs_set):
                continue

            if some_in_set(op.operands_source(), inputs_set):
                for value in op.results():
                    if value not in no_grad_set:
                        inputs_set.add(value)

            else:
                relevant_op_flags[i] = False

    # from output to input
    for i, op in reversed(list(enumerate(total_ops))):
        # while op support
        if some_in_set(op.results(), outputs_set):
            for operand in op.operands_source():
                if operand not in no_grad_set:
                    outputs_set.add(operand)
        else:
            relevant_op_flags[i] = False

    effective_ops = [
        total_ops[i] for i in range(len(total_ops)) if relevant_op_flags[i]
    ]
    uneffective_ops = [
        total_ops[i]
        for i in reversed(range(len(total_ops)))
        if not relevant_op_flags[i]
    ]

    return effective_ops, uneffective_ops


def update_no_grad_set_after_purne(
    block, effective_forward_op, no_grad_set, inputs, outputs
):
    '''
    update no_grad_set after forward purne

    from inputs to outputs add value not in the path to no_grad_set,
    from outputs to inputs add value not in the path to no_grad_set,
    '''
    inputs_set = set(inputs)
    if inputs_set:
        for op in block.ops:
            if some_in_set(op.operands_source(), inputs_set):
                for value in op.results():
                    if value not in no_grad_set:
                        inputs_set.add(value)

        for op in effective_forward_op:
            for value in op.operands_source():
                if value not in inputs_set:  # and value.get_stopgradient():
                    no_grad_set.add(value)

    outputs_set = set(outputs)
    no_grad_set_tmp = set()
    for op in reversed(effective_forward_op):
        for output in op.results():
            if output not in outputs_set and not some_in_set(
                [output], set(op.operands_source())
            ):
                no_grad_set_tmp.add(output)

        for input in op.operands_source():
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
        for x in op.operands():
            if x.source().get_defining_op() in ops_set:
                pending_count[x.source().get_defining_op()] += 1

    queue = collections.deque()

    for op in ops:
        if pending_count[op] == 0:
            queue.append(op)

    while queue:
        op = queue.popleft()
        sorted_list.append(op)

        for x in op.operands():
            x_op = x.source().get_defining_op()
            pending_count[x_op] -= 1
            if pending_count[x_op] == 0:
                queue.append(x_op)

    if len(sorted_list) != len(ops):
        raise ValueError(
            "inverse_sort_op wrong, sorted_list size is not equal to origin_list size"
        )

    return sorted_list


def append_backward_ops(
    block, effective_forward_op, no_grad_set, backward_ops, state
):
    '''
    add grad_op in order of topological inverse sort
        eg:
        from :op1 -> v1 -> op2 -> v2 -> op3 -> v3
        to: og1_g <- v1_g <- op2_g <- v2_g <- op3_g <- v3_g

    if op has grad_op, prepare its grad_op's inputs by value_to_valuegrad,
        eg:
        value_to_valuegrad[v3] = [[v3_g]];
        v2_g = call_vjp(op3, [v3_g], [v2_stopgradient])


    if op don't has grad_op, if it don't has input and it's output has more than
    one output_grad, add sumop for grad aggregation.
        (eg: full op and get_parameter op etc.)

    else continue to next op.
    '''
    for op in effective_forward_op:
        if paddle.framework.core.has_vjp(op):
            # prepare output_grad
            output_grad_list = []  # (opresult)
            zero_flag = [False] * op.num_results()
            for i, value in enumerate(op.results()):
                if (
                    value not in state.value_to_valuegrad
                    or state.value_to_valuegrad[value] is None
                ):
                    # first case:
                    # this fwd_op's output didn't used by other fwd_op,
                    # so no output_grad created.

                    # second case:
                    # last bwd_op return None because input in no_grad_set,
                    # but this bwd_op need a input.

                    grad_value = paddle.full(
                        value.shape,
                        0.0,
                        dtype=value.dtype,
                    )
                    fillop = grad_value.get_defining_op()

                    update_bwdop_structure(
                        backward_ops, state.op_to_opgrad[op], fillop
                    )
                    state.value_to_valuegrad[value] = [[grad_value]]
                    zero_flag[i] = True

                if len(state.value_to_valuegrad[value]) > 1:
                    # one value is input of more than one fwd_op,
                    # so more than one bwd_op create input_grad,
                    # need add sum op to accumulate gradient

                    paddle.add_n(list(state.value_to_valuegrad[value]))
                    sumop = block.ops[len(block.ops) - 1]
                    update_bwdop_structure(
                        backward_ops, state.op_to_opgrad[op], sumop
                    )
                    state.value_to_valuegrad[value] = [[sumop.result(0)]]
                    state.value_to_sumvaluegrad[
                        value
                    ] = state.value_to_valuegrad[value]

                output_grad = state.value_to_valuegrad[value][0]
            output_grad_list.append(output_grad)

            # all(zero_flag) support this op has no contribution for grad
            # should be delete (prune sub_graph)
            if len(output_grad_list) == 0 or all(zero_flag):
                continue

            # prepare input_grad stop_gradient info.
            input_grad_stopgradient_list = []
            for input in op.operands_source():
                if input in no_grad_set:
                    input_grad_stopgradient_list.append([True])
                else:
                    input_grad_stopgradient_list.append([False])

            before_ops_num = len(block.ops)
            # prim should be a globel flag, it will make create_grad_op choose diffrient func
            input_grad_list = paddle.framework.core.call_vjp(
                op, output_grad_list, input_grad_stopgradient_list
            )
            after_ops_num = len(block.ops)

            # find new grad_op_list
            grad_op_list = []
            for i in range(before_ops_num, after_ops_num):
                grad_op_list.append(block.ops[i])

            for i, input in enumerate(op.operands()):
                input_grad = input_grad_list[i]
                state.value_to_valuegrad[input.source()].append(input_grad)

            # add grad_op
            for grad_op in grad_op_list:
                update_bwdop_structure(
                    backward_ops, state.op_to_opgrad[op], grad_op
                )

        else:
            if op.num_operands() == 0 and op.num_results() != 0:
                for value in op.results():
                    if len(state.value_to_valuegrad[value]) > 1:
                        # need add sum op
                        paddle.add_n(list(state.value_to_valuegrad[value]))
                        sumop = block.ops[len(block.ops) - 1]
                        update_bwdop_structure(
                            backward_ops, state.op_to_opgrad[op], sumop
                        )
                        state.value_to_valuegrad[value] = [[sumop.result(0)]]
                        state.value_to_sumvaluegrad[
                            value
                        ] = state.value_to_valuegrad[value]
                    else:
                        state.op_to_opgrad[op] = []
                else:
                    state.op_to_opgrad[op] = []


def create_backward_purne_set(inputs, outputs, no_grad_set, state):
    outputs_set = set()
    for input in inputs:
        if state.value_to_valuegrad[input] != []:
            outputs_set.add(state.value_to_valuegrad[input][0][0])

    inputs_set = set()
    for output in outputs:
        if state.value_to_valuegrad[output] != []:
            inputs_set.add(state.value_to_valuegrad[output][0][0])

    no_gradvar_set = set()  # grad_value of value in no_grad_set
    for key in state.value_to_valuegrad:
        if key in no_grad_set:
            no_gradvar_set.add(state.value_to_valuegrad[key][0][0])

    for key in state.value_to_sumvaluegrad:
        if key in no_grad_set:
            for item in state.value_to_valuegrad[key][0]:
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
        value = state.valuegrad_to_value[valuegrad][0]
        state.value_to_valuegrad[value] = []
        if value in state.sumvaluegrad_to_value:
            raise ValueError('input_grad in [%s] is value which need to sum ')


def calc_gradient_helper(outputs, inputs, grad_outputs, no_grad_set):
    block = outputs[0].get_defining_op().get_parent_block()
    state = State(block.get_parent_program())
    # check all inputs and outputs in the same block
    check_all_puts(block, inputs, outputs)
    # update no_grad_set if some value stop_gradient=True
    update_no_grad_set_by_stopgradient(block, no_grad_set)
    complete_outputs, _, backward_ops = prepare_grad_outputs(
        block,
        grad_outputs,
        outputs,
        state.value_to_valuegrad,
        state.op_to_opgrad,
    )

    inputs_set = set(inputs)
    outputs_set = set(complete_outputs)
    effective_forward_op, _ = prune_ops(
        block.ops, inputs_set, outputs_set, no_grad_set
    )
    update_no_grad_set_after_purne(
        block, effective_forward_op, no_grad_set, inputs, complete_outputs
    )

    sorted_effective_forward_op = inverse_sort_op(effective_forward_op)

    append_backward_ops(
        block, sorted_effective_forward_op, no_grad_set, backward_ops, state
    )
    # now value_to_valuegrad should be value <-> value (add sum op for the same values's gradvalue)

    outputs_set, inputs_set, no_gradvar_set = create_backward_purne_set(
        inputs, complete_outputs, no_grad_set, state
    )
    _, remove_ops = prune_ops(
        backward_ops, inputs_set, outputs_set, no_gradvar_set
    )

    state.turn_map()

    for bwd_op in inverse_sort_op(remove_ops):
        remove_op(block, bwd_op, state)

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
        ((paddle.ir.Value, paddle.ir.OpResult), list, tuple),
        'paddle.ir.grad',
    )
    check_type(
        inputs,
        'inputs',
        ((paddle.ir.Value, paddle.ir.OpResult), list, tuple),
        'paddle.ir.grad',
    )
    check_type(
        grad_outputs,
        'grad_outputs',
        ((paddle.ir.Value, paddle.ir.OpResult), list, tuple, type(None)),
        'paddle.ir.grad',
    )

    check_type(
        no_grad_vars,
        'no_grad_vars',
        ((paddle.ir.Value, paddle.ir.OpResult), list, tuple, set, type(None)),
        'paddle.ir.grad',
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
