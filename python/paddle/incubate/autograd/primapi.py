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

import typing

from paddle.fluid import backward, framework
from paddle.incubate.autograd import primx, utils


@framework.static_only
def forward_grad(outputs, inputs, grad_inputs=None):
    """Forward mode of automatic differentiation.

    .. note::
        **ONLY available in the static mode and primitive operators.**

    Args:
        outputs(Tensor|Sequence[Tensor]): The output tensor or tensors.
        inputs(Tensor|Sequence[Tensor]): The input tensor or tensors.
        grad_inputs(Tensor|Sequence[Tensor]): Optional, the gradient Tensor or
            Tensors of inputs which has the same shape with inputs, Defaults to
            None, in this case is equivalent to all ones.

    Returns:
        grad_outputs(Tensor|Sequence[Tensor]): The gradients for outputs.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            paddle.enable_static()
            paddle.incubate.autograd.enable_prim()

            startup_program = paddle.static.Program()
            main_program = paddle.static.Program()

            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data('x', shape=[1], dtype='float32')
                y = x * x
                y_grad = paddle.incubate.autograd.forward_grad(y, x)
                paddle.incubate.autograd.prim2orig()

            exe = paddle.static.Executor()
            exe.run(startup_program)
            y_grad = exe.run(main_program, feed={'x': np.array([2.]).astype('float32')}, fetch_list=[y_grad])
            print(y_grad)
            # [array([4.], dtype=float32)]

            paddle.incubate.autograd.disable_prim()
            paddle.disable_static()
    """
    if not utils.prim_enabled():
        raise RuntimeError('forward_grad must be running on primitive'
                           'operators, use enable_prim to turn it on.')

    if not isinstance(outputs, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected outputs is Tensor|Sequence[Tesnor], '
                        f'but got {type(outputs)}.')

    if not isinstance(inputs, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected inputs is Tensor|Sequence[Tesnor], '
                        f'but got {type(inputs)}.')

    ys, xs, xs_dot = utils.as_tensors(outputs), utils.as_tensors(
        inputs), utils.as_tensors(grad_inputs)

    block = framework.default_main_program().current_block()
    if any(x.block != block for x in xs + ys):
        raise RuntimeError(
            'Variable in inputs and targets should exist in current block of '
            'main program.')

    primx.orig2prim(block)
    ad = primx.Transform(ys[0].block)
    _, ys_dot = ad.linearize(xs, ys, xs_dot)

    return ys_dot[0] if isinstance(outputs, framework.Variable) else ys_dot


@framework.static_only
def grad(outputs, inputs, grad_outputs=None):
    """Reverse mode of automatic differentiation.

    .. note::
        **ONLY available in the static mode and primitive operators**

    Args:
        outputs(Tensor|Sequence[Tensor]): The output Tensor or Tensors.
        inputs(Tensor|Sequence[Tensor]): The input Tensor or Tensors.
        grad_outputs(Tensor|Sequence[Tensor]): Optional, the gradient Tensor or
            Tensors of outputs which has the same shape with outputs, Defaults
            to None, in this case is equivalent to all ones.

    Returns:
        grad_inputs(Tensor|Tensors): The gradients for inputs.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            paddle.enable_static()
            paddle.incubate.autograd.enable_prim()

            startup_program = paddle.static.Program()
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data('x', shape=[1], dtype='float32')
                x.stop_gradients = False
                y = x * x
                x_grad = paddle.incubate.autograd.grad(y, x)
                paddle.incubate.autograd.prim2orig()

            exe = paddle.static.Executor()
            exe.run(startup_program)
            x_grad = exe.run(main_program, feed={'x': np.array([2.]).astype('float32')}, fetch_list=[x_grad])
            print(x_grad)
            # [array([4.], dtype=float32)]

            paddle.incubate.autograd.disable_prim()
            paddle.disable_static()
    """
    if not utils.prim_enabled():
        grad_inputs = backward.gradients(outputs, inputs, grad_outputs)
        # backward.gradients returns a list though the inputs is a signle Tensor.
        # The follow code snippet fixes the problem by return the first element
        # of grad_inputs when the inputs is a signle Tensor.
        if isinstance(inputs, framework.Variable) and isinstance(
                grad_inputs, typing.Sequence) and len(grad_inputs) > 0:
            return grad_inputs[0]
        else:
            return grad_inputs

    if not isinstance(outputs, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected outputs is Tensor|Sequence[Tesnor], '
                        f'but got {type(outputs)}.')

    if not isinstance(inputs, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected inputs is Tensor|Sequence[Tesnor], '
                        f'but got {type(inputs)}.')

    ys, xs, ys_bar = utils.as_tensors(outputs), utils.as_tensors(
        inputs), utils.as_tensors(grad_outputs)
    block = framework.default_main_program().current_block()
    if any((x is not None and x.block != block) for x in xs + ys):
        raise RuntimeError(
            'Variable in inputs and outputs should be None or in current block of main program'
        )

    # TODO(Tongxin) without any prior knowledge about whether the program
    # is completely lowered to primitive ops, it's mandatory to run the lowering
    # pass once and again. This is obviously inefficient and needs to be
    # optimized.
    primx.orig2prim(block)
    ad = primx.Transform(block)
    xs_dot, ys_dot = ad.linearize(xs, ys)
    if any(var is None for var in ys_dot):
        raise RuntimeError(
            'Grads cannot be computed. The given outputs does not depend on inputs'
        )
    ys_bar, xs_bar = ad.transpose(ys_dot, xs_dot, ys_bar)

    # remove xs_dot and their constructor ops
    op_indexes = []
    for var in xs_dot:
        if var is not None:
            op_index = block.ops.index(var.op)
            if op_index < 0:
                raise ValueError(
                    f'op_index should be greater than or equal to 0, but op_index={op_index}.'
                )
            op_indexes.append(op_index)

    ad.erase_ops(sorted(op_indexes))
    ad.erase_dots(xs_dot)

    return xs_bar[0] if isinstance(inputs, framework.Variable) else xs_bar
