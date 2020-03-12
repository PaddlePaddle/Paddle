#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import copy
import six
import warnings

import functools
from . import layers
from . import framework
from . import core
from . import name_scope
from .dygraph import base as imperative_base

__all__ = [
    'set_gradient_clip', 'ErrorClipByValue', 'grad_clip_value',
    'grad_clip_norm', 'grad_clip_global_norm'
]


class BaseErrorClipAttr(object):
    def __str__(self):
        raise NotImplementedError()

    def _append_clip_op(self, block, grad_name):
        raise NotImplementedError()


class ErrorClipByValue(BaseErrorClipAttr):
    """
    Clips tensor values to the range [min, max].

    Given a tensor ``t`` (see Examples below), this operation clips its value \
    to ``min`` and ``max`` inplace.

    - Any values less than min are set to min.
    - Any values greater than max are set to max.

    Args:
        max (float): The maximum value to clip by.
        min (float, optional): The minimum value to clip by. if not set by user, \
        will be set to ``-max`` by framework.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            BATCH_SIZE = 128
            CLIP_MAX = 2e-6
            CLIP_MIN = -1e-6
            prog = fluid.framework.Program()
            with fluid.program_guard(main_program=prog):
                image = fluid.layers.data(
                    name='x', shape=[784], dtype='float32')
                hidden1 = fluid.layers.fc(input=image, size=128, act='relu')
                hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')
                predict = fluid.layers.fc(
                    input=hidden2, size=10, act='softmax')
                label = fluid.layers.data(name='y', shape=[1], dtype='int64')
                cost = fluid.layers.cross_entropy(input=predict, label=label)
                avg_cost = fluid.layers.mean(cost)
            prog_clip = prog.clone()
            prog_clip.block(0).var(hidden1.name)._set_error_clip(
                fluid.clip.ErrorClipByValue(
                    max=CLIP_MAX, min=CLIP_MIN)
    """

    def __init__(self, max, min=None):
        max = float(max)
        if min is None:
            min = -max
        else:
            min = float(min)
        self.max = max
        self.min = min

    def __str__(self):
        return "ByValue, min=%f, max=%f" % (self.min, self.max)

    def _append_clip_op(self, block, grad_name):
        clip_op_desc = block.desc.append_op()
        clip_op_desc.set_type("clip")
        clip_op_desc.set_input("X", [grad_name])
        clip_op_desc.set_output("Out", [grad_name])
        clip_op_desc._set_attr("min", self.min)
        clip_op_desc._set_attr("max", self.max)


def error_clip_callback(block, context):
    # the context is a grad_to_var map
    grad_to_var = context
    op_desc = block.desc.op(block.desc.op_size() - 1)
    for grad_n in [n for n in op_desc.output_arg_names() if n in grad_to_var]:
        fwd_var = block._var_recursive(grad_to_var[grad_n])
        error_clip = getattr(fwd_var, "error_clip", None)
        if not (error_clip is None or isinstance(error_clip,
                                                 BaseErrorClipAttr)):
            raise TypeError(
                "Variable's error_clip should be an instance of BaseErrorClipAttr or None."
            )
        if error_clip is not None:
            error_clip._append_clip_op(block, grad_n)


@framework.dygraph_not_support
def set_gradient_clip(clip, param_list=None, program=None):
    """
    This API is deprecated since 2.0. Please do not use it. The better
    gradient clip strategy is to clip by :ref:`api_fluid_grad_clip_value` ,
    :ref:`api_fluid_grad_clip_norm` , :ref:`api_fluid_grad_clip_global_norm` .
    """
    warnings.warn(
        "Caution! paddle.fluid.clip.set_gradient_clip is deprecated since 2.0 "
        "and not maintained any more, because it is easily to be misused.\n"
        "After that, we recommend a new gradient clipping strategy where you "
        "do gradient clipping through the following API:\n"
        "1. paddle.fluid.grad_clip_value: clip gradient by value.\n"
        "2. paddle.fluid.grad_clip_norm: clip gradient by norm.\n"
        "3. paddle.fluid.grad_clip_global_norm: clip gradient by global norm.\n"
        "This method can reduce the mistakes, please see documention of these "
        "gradient clip APIs")


def _correct_clip_op_role_var(params_grads):
    for param, grad in params_grads:
        if grad is None:
            continue
        for op in param.block.program.global_block().ops:
            if 'op_namescope' in op.all_attrs() and op.attr(
                    "op_namescope") is "gradient_clip":
                if op.attr('op_role_var'):
                    param_name = op.attr('op_role_var')[0]
                    index = 0
                    for i in range(len(params_grads)):
                        if params_grads[i][0].name == param_name:
                            index = i
                    correct_p_g = [param_name, params_grads[index][1].name]
                    op._set_attr('op_role_var', correct_p_g)


def grad_clip_value(params_grads, max, min=None):
    """
    This API is used to clip gradient values to the range [min, max].

    Given a list ``params_grads``, this operation clips its grads to ``min`` and ``max`` inplace.
    
    Every element of list ``params_grads`` is pair (param, grad), and  ``params_grads``can be obtained 
    by optimizer.backward(loss).
    
    - Any gradient values less than min are set to ``min``.
    - Any gradient values greater than max are set to ``max``.

    Args:
        params_grads (list): list of (param, grad) pair to do gradient clipping, it can be obtained by
            optimizer.backward(loss).
        max (float): The maximum value to clip by.
        min (float, optional): The minimum value to clip by. if not set by user, \
            will be set to -max by framework.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np
                        
            main_prog = fluid.Program()
            startup_prog = fluid.Program()
            with fluid.program_guard(
                    main_program=main_prog, startup_program=startup_prog):
                image = fluid.data(
                    name='x', shape=[-1, 2], dtype='float32')
                predict = fluid.layers.fc(input=image, size=3, act='relu')
                loss = fluid.layers.mean(predict)
                
                sgd_optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.1)
                params_grads = sgd_optimizer.backward(loss)
                params_grads = fluid.grad_clip_value(params_grads, max=1.0, min=0.5)
                sgd_optimizer.apply_optimize(params_grads)
                
                grads = [elem[1] for elem in params_grads]

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            x = np.random.uniform(-100, 100, (10, 2)).astype('float32')
            exe.run(startup_prog)
            out = exe.run(main_prog, feed={'x': x}, fetch_list=grads)
            print("The weights'gradient of fc after clipping is:\n {}"
                "\n\nThe bias'gradient of fc after clipping is:\n {}"
                .format(out[0], out[1]))         
    """

    max = float(max)
    if min is None:
        min = -max
    else:
        min = float(min)
    params_and_grads = []
    with framework.name_scope('gradient_clip'):
        for p, g in params_grads:
            if g is None:
                params_and_grads.append((p, g))
                continue

            with p.block.program._optimized_guard([p, g]):
                new_g = layers.clip(x=g, max=max, min=min)

            params_and_grads.append((p, new_g))

    _correct_clip_op_role_var(params_and_grads)
    return params_and_grads


def grad_clip_norm(params_grads, clip_norm):
    """
    This API is used to clip gradient values by L2 norm.

    Given a list ``params_grads``, every element of list ``params_grads`` is pair (param, grad).
    `grad` is a multidimensional Tensor, this operation limits the L2 norm of `grad` within 
    `clip_norm`. 
    
    - If the L2 norm of `grad` is less than or equal to `clip_norm`, Out will be the same as `grad`. 
    - If the L2 norm of `grad` is greater than `clip_norm`, `grad` will be linearly scaled to make 
    the L2 norm of Out equal to `clip_norm`, as shown in the following formula: 
    
    
    .. math::
        Out =
        \\left \{
        \\begin{aligned}
        & grad & & if (norm(grad) \\leq clip\_norm) \\\\
        & \\frac{clip\_norm*grad}{norm(grad)} & & if (norm(grad) > clip\_norm) \\\\
        \\end{aligned}
        \\right.


    where :math:`norm(grad)` represents the L2 norm of :math:`grad`.

    .. math::
        norm(grad) = ( \\sum_{i=1}^{n}|grad\_i|^2)^{ \\frac{1}{2}}

    Args:
        params_grads (list): list of (param, grad) pair to do gradient clipping, it can be obtained by
            optimizer.backward(loss).
        clip_norm (float): The maximum norm value

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np
                        
            main_prog = fluid.Program()
            startup_prog = fluid.Program()
            with fluid.program_guard(
                    main_program=main_prog, startup_program=startup_prog):
                image = fluid.data(
                    name='x', shape=[-1, 2], dtype='float32')
                predict = fluid.layers.fc(input=image, size=3, act='relu')
                loss = fluid.layers.mean(predict)
                
                sgd_optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.1)
                params_grads = sgd_optimizer.backward(loss)
                params_grads = fluid.grad_clip_norm(params_grads, clip_norm=5.0)
                sgd_optimizer.apply_optimize(params_grads)
                
                grads = [elem[1] for elem in params_grads]
                grads_norm = []
                for grad in grads:
                    squre = fluid.layers.square(grad)
                    squre_sum = fluid.layers.reduce_sum(squre)
                    grad_norm = fluid.layers.sqrt(squre_sum)
                    grads_norm.append(grad_norm)
                grads.extend(grads_norm)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            x = np.random.uniform(-100, 100, (10, 2)).astype('float32')
            exe.run(startup_prog)
            out = exe.run(main_prog, feed={'x': x}, fetch_list=grads)
            print("The weights'gradient of fc after clipping is:\n {}"
                "\n\nThe corresponding norm:\n {} "
                "\n\nThe bias'gradient of fc after clipping is:\n {}"
                "\n\nThe corresponding norm:\n {}"
                .format(out[0], out[2], out[1], out[3]))
    """
    clip_norm = float(clip_norm)
    params_and_grads = []
    with framework.name_scope('gradient_clip'):
        for p, g in params_grads:
            if g is None:
                params_and_grads.append((p, g))
                continue

            with p.block.program._optimized_guard([p, g]):
                new_g = layers.clip_by_norm(x=g, max_norm=clip_norm)

            params_and_grads.append((p, new_g))

    _correct_clip_op_role_var(params_and_grads)
    return params_and_grads


def grad_clip_global_norm(params_grads, clip_norm):
    """
    This API is used to clip gradient values by the sum of all `grad's` L2 norm.

    Given a list ``params_grads``, every element of list ``params_grads`` is pair (param, grad).
    `grad` is a multidimensional Tensor, this operation limits the sum of all `grad's` L2 norm within 
    `clip_norm`. 
    
    - If the sum of all `grad's` L2 norm is less than or equal to `clip_norm`, every `grad` will remain the same. 
    - If the sum of all `grad's` L2 norm is greater than `clip_norm`, every `grad` will be linearly scaled. 

    To perform the clipping, the values :math:`t\_list[i]` are set to:

    .. math::

        t\_list[i] = t\_list[i] * \\frac{clip\_norm}{\max(global\_norm, clip\_norm)}

    where:

    .. math::

        global\_norm = \sqrt{\sum_{i=0}^{N-1}(l2norm(t\_list[i]))^2}

    If :math:`clip\_norm > global\_norm` then the entries in t_list remain as they are,
    otherwise they're all shrunk by the global ratio.

    Args:
        params_grads (list): list of (param, grad) pair to do gradient clipping, it can be obtained by
            optimizer.backward(loss).
        clip_norm (float): The maximum norm value

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np
                        
            main_prog = fluid.Program()
            startup_prog = fluid.Program()
            with fluid.program_guard(
                    main_program=main_prog, startup_program=startup_prog):
                image = fluid.data(
                    name='x', shape=[-1, 2], dtype='float32')
                predict = fluid.layers.fc(input=image, size=3, act='relu')
                loss = fluid.layers.mean(predict)
                
                sgd_optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.1)
                params_grads = sgd_optimizer.backward(loss)
                params_grads = fluid.grad_clip_global_norm(params_grads, clip_norm=5.0)
                sgd_optimizer.apply_optimize(params_grads)
                
                grads = [elem[1] for elem in params_grads]
                squres_sum = []
                for grad in grads:
                    squre = fluid.layers.square(grad)
                    squre_sum = fluid.layers.reduce_sum(squre)
                    squres_sum.append(squre_sum)
                    
                squres_sum = fluid.layers.sums(squres_sum)
                global_norm = fluid.layers.sqrt(squres_sum)
                grads.append(global_norm)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            x = np.random.uniform(-100, 100, (10, 2)).astype('float32')
            exe.run(startup_prog)
            out = exe.run(main_prog, feed={'x': x}, fetch_list=grads)
            print("The weights'gradient of fc after clipping is:\n {}"
                "\n\nThe bias'gradient of fc after clipping is:\n {}"
                "\n\nThe corresponding global norm:\n {}"
                .format(out[0], out[1], out[2]))
    """

    clip_norm = float(clip_norm)
    params_and_grads = []
    square_list = []
    with framework.name_scope('gradient_clip'):
        for p, g in params_grads:
            if g is None:
                continue

            merge_grad = g
            with p.block.program._optimized_guard([p, g]):
                if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                    merge_grad = layers.merge_selected_rows(g)
                    merge_grad = layers.get_tensor_from_selected_rows(
                        merge_grad)

                square = layers.square(merge_grad)
                local_norm_var = layers.reduce_sum(input=square)
                square_list.append(local_norm_var)

        with p.block.program._optimized_guard([p, g]):
            global_norm_var = layers.sums(square_list)
            global_norm_var = layers.sqrt(x=global_norm_var)

            clip_var = layers.fill_constant(
                shape=[1], dtype="float32", value=clip_norm)

            scale_var = layers.elementwise_div(
                x=clip_var,
                y=layers.elementwise_max(
                    x=clip_var, y=global_norm_var))

        for p, g in params_grads:
            if g is None:
                params_and_grads.append((p, g))
                continue

            with p.block.program._optimized_guard([p, g]):
                new_grad = layers.elementwise_mul(x=g, y=scale_var)
            params_and_grads.append((p, new_grad))

    _correct_clip_op_role_var(params_and_grads)
    return params_and_grads
