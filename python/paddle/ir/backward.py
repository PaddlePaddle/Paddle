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

from audioop import reverse
from collections.abc import Sequence

from . import bind, core

__all__ = [
    'grad',
]
def check_type(input, input_name, expected_type, op_name, extra_message=''):
    if not isinstance(input, expected_type):
        raise TypeError(
            "The type of '%s' in %s must be %s, but received %s. %s"
            % (input_name, op_name, expected_type, type(input), extra_message)
        )

def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, Sequence) else [x]

def append_backward_ops(block, effective_forward_op, value_to_valuegrad, op_to_opgrad):

    for op in reverse(effective_forward_op):
        if op.is_parameter():
            continue

        if op.has_gradfunc_infer():
            value_operand = []
            for value in op.operand():
                # backward op use forward op's outputs'grad,
                #forward op's input or output privied by forward op
                if len(value_to_valuegrad[value]) == 1:
                    list_valuegrad = value_to_valuegrad[value]
                else:
                    for item in value_to_valuegrad[value]:
                        if item[0] == op:
                            list_valuegrad = item  # 获取此输入的使用op对应的反向变量
                grad_value = list_valuegrad[2]
                value_operand.append(grad_value)
            gradop, gradop_list = op.create_grad_op(value_operand)
            # 非组合模式
            if len(gradop) == 1:
                if op.opoperands().size() != gradop.outputgrad_opresult_num():
                    raise ValueError()
                else:
                    # make sure all forward op inputs order is same to the correspond gradop's order
                    output_grad_startid = gradop.opresult().size() - gradop.outputgrad_opresult_num()
                    for i in range(output_grad_startid, gradop.opresult().size()):
                        value_to_valuegrad[op.operand()[i]].append((gradop, i , gradop.opresults()[output_grad_startid + i]))
                    op_to_opgrad[op].append(gradop)
                    block.push_back(gradop)
            else:#组合模式
                # 在create_grad_op中调用op.create_compgrad_op(输入)，从输入出发遍历拆解逻辑，
                # 得到小op的输出，使用set_output时将小op的输出value放置在gradop的opresults() 中；小op放置在gradop_list
                for i in range(output_grad_startid, gradop.opresult().size()):
                    valuegrad = gradop.opresults()[output_grad_startid + i]
                    valuegrad_sourceop = valuegrad.get_operation()
                    value_to_valuegrad[op.operand()[i]].append((valuegrad_sourceop, i , valuegrad))
                for op in gradop_list:
                    op_to_opgrad[op].append(op)
                    block.push_back(op)
                gradop.weak_destory()

def check_all_puts(block, inputs, outputs):
    for output in outputs:
        if ouput.get_operation().get_block() != block:
            raise ValueError("all outputs must be in the same block")
    for input in inputs:
        if input.get_operation().get_block() != block:
            raise ValueError("all inputts must be in the same block with outputs")

def update_no_grad_set(block, no_grad_set):
    for op in block.ops:
        for opresult_idx in range(op.num_results()):
            value = op.result(opresult_idx)
            if value.get_stop_gradient and value not in no_grad_set:
                no_grad_set.add(value)

def check_grad_outputs(block, grad_outputs, outputs, value_to_valuegrad):
    if not grad_outputs:
        grad_outputs = [None] * len(outputs)

    if len(grad_outputs) != len(outputs):
        raise ValueError(
            "Should have the same number of grad_outputs as outputs"
        )
    backward_ops = set()
    for i, grad in grad_outputs:
        output = outputs[i]
        if grad is None:
            fillop = bind.create_operation(
                    "pd.fill",
                    {input: {}},
                    {attr: {value : 1.0, dtype : opresult.dtype()}},
                )
            
            block.push_back(fillop)
            #set vale_to_valuegrad None represent output is foward leaf value
            # fwd : op1 -> op2 -> op3 -> output
            # bwd : op1G <- op2G <- op3G <- outputG <- fillop/feedop
            backward_ops.add(fillop)
            grad_value = fillop.result(0)
            value_to_valuegrad[output] = tuple(fillop, 0,  grad_value)
        else:
            if output.shape != grad.shape:
                raise ValueError(
                    "The shape of grad_output[%d] should be the same as the shape of output[%d]" % (i, i)
                )
            if output.dtype!= grad.dtype:
                raise ValueError(
                    "The dtype of grad_output[%d] should be the same as the dtype of output[%d]" % (i, i)
                )
            backward_ops.add(feedop)
            feedop = grad.get_operation()
            value_to_valuegrad[output] = tuple(feedop, 0,  grad)

    # add input for bwd first op
    complete_outputs = outputs
    complete_gradoutputs = grad_outputs

    visited_output = set()
    for output in outputs:
        if output in visited_output:
            continue
        for i in range(output.get_operation().num_results()):
            opresult =  output.get_operation().result(i)
            if opresult in value_to_valuegrad:
                visited_output.add(opresult)
                continue
            else:
                fillop = bind.create_operation(
                    "pd.fill",
                    {input: {}},
                    {attr: {value : 0.0, dtype : opresult.dtype()}},
                )
                grad_value = fillop.result(0)
                value_to_valuegrad[opresult] = tuple(fillop, 0,  grad_value)
                visited_output.add(opresult)
                complete_outputs.append(opresult)
                complete_gradoutputs.append(grad_value)

    return complete_outputs, complete_gradoutputs, backward_ops

def some_in_set(value_list, value_set):
    for item in value_list:
        if item in value_set:
            return True
    return False
def prune_forward_op(block, outputs, complete_outputs, inputs, no_grad_set, backward_ops):
    relevant_op_flags = [False] * len(block.ops)
    inputs_set =  set(inputs)
    outputs_set = set(complete_outputs)

    #from input to output
    if inputs:
        for i, op in block.ops:
            if some_in_set(op.opoperands(), inputs_set) and op.has_gradfunc_infer():
                for value in op.opresults():
                    if value not in no_grad_set:
                        inputs_set.add(value)
                relevant_op_flags[i] = True

    #from output to input
    for i, op in reverse(list(enumerate(block.ops))):
        # while op support

        #pass backward input create op (feedop/fillop)maybe need , because those op has no grad op
        if op in backward_ops:
            continue
        if some_in_set(op.opresults(), outputs_set) and op.has_gradfunc_infer():
            for value in op.opoperands():
                if value not in no_grad_set:
                    outputs_set.add(value)
            relevant_op_flags[i] = True

    # while op support
    # if is_while:
    #     pass

    effective_forward_op = [block.ops[i] for i in range(len(block.ops)) if relevant_op_flags[i] ]
    #update no_grad_set
    if inputs:
        for op in effective_forward_op:
            for value in op.opoperands():
                if value not in inputs_set and value.get_stopgradient():
                    no_grad_set.add(value)

    return effective_forward_op
def add_sum_op(block, value_to_valuegrad):
    for key in value_to_valuegrad:
        if len(value_to_valuegrad[key]) > 1:
            sumop = bind.create_operation(
                "pd.add_n",
                {input: {[item[2] for item in value_to_valuegrad[key]]}},
                {attr:{}},
            )
            block.push_back(sumop)
            grad_value = sumop.result(0)
            value_to_valuegrad[key] = tuple(sumop, 0,  grad_value)

def calc_gradient_helper(outputs, inputs, grad_outputs, no_grad_set):
    block = outputs[0].get_operation().get_block()
    # check all inputs and outputs in the same block
    check_all_puts(block, inputs, outputs)
    # update no_grad_set if some value stop_gradient=True
    update_no_grad_set(block, no_grad_set)
    # rebind value and its gradient, maybe one value has more than one gradient, because it canbe more than one op's input
    # [input] = {tuple(fwd_next_op, input_slot, input_grad), tuple(fwd_next_op, input_slot, input_grad)}
    value_to_valuegrad = dict()
    # check if grad_outputs is none, if none create fill_any_like op and its opresult

    # form target find its source op and check all opresult of this op , if outputs do not has , add fill_any_like op
    # if has pass

    # check if outputs shape and dtype is same to grad_outputs, else raise error

    # update value_to_valuegrad
    # return origin outputs/gradoutputs privided by users + compute need outputs we add in second part.
    complete_outputs, complete_gradoutputs, backward_ops = check_grad_outputs(block, grad_outputs, outputs, value_to_valuegrad)

    # if operand in no_grad_set do not need add its used op so
    # from inputs check all block op if donot need set flag = False

    # from outputs , get its source op -> operand -> source op , set flag=True
    # record flag = True op* in op_path,

    # record all value in visited set. check if all inputs in visited set. throw eeror when allow_used =False

    # reverse visited op list from complete_outputs to inputs, if output not in effective graph add it in no_grad_set

    # update no_grad_set
    # return visited op*list

    #****while op op_path special****#
    effective_forward_op = prune_forward_op(block, complete_outputs, inputs, no_grad_set)

    #record op and its gradop, op -> grad op/in composite it will be op -> gradop list
    # [Operation] = List(Operation, Operation)
    # may be auto parallel need this map
    op_to_opgrad = dict()

    append_backward_ops(block, effective_forward_op, complete_gradoutputs, no_grad_set, value_to_valuegrad, op_to_opgrad)
    add_sum_op(block, value_to_valuegrad)
    # now value_to_valuegrad should be value <-> value (add sum op for the same values's gradvalue)
    return value_to_valuegrad

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
    input_to_inputgrad_map = calc_gradient_helper(outputs, inputs, grad_outputs = grad_outputs, no_grad_set = grad_outputs)

    inputgrad = []
    for input in inputs:
        inputgrad.append(input_to_inputgrad_map[inputs])

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
        (core.Value, list, tuple),
        'paddle.static.gradients',
    )
    check_type(
        inputs,
        'inputs',
        (core.Value, list, tuple),
        'paddle.static.gradients',
    )
    check_type(
        grad_outputs,
        'grad_outputs',
        (core.Value, list, tuple, type(None)),
        'paddle.static.gradients',
    )

    check_type(
        no_grad_vars,
        'no_grad_vars',
        (core.Value, list, tuple, set),
        'paddle.static.gradients',
    )
    outputs = _as_list(outputs)
    inputs = _as_list(inputs)
    grad_outputs = _as_list(grad_outputs)

    if no_grad_vars is not set:
        no_grad_set = []
        for item in no_grad_vars:
            no_grad_set.add(item)
        input_grad = calc_gradient(outputs, inputs, grad_outputs, no_grad_set)
    else:
        input_grad = calc_gradient(outputs, inputs, grad_outputs, no_grad_vars)


    return input_grad
