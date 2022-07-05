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

import paddle.autograd.utils as tensor_utils
import paddle.incubate.autograd.utils as prim_utils
from paddle.fluid import framework
from paddle.incubate.autograd import primx


@framework.static_only
def forward_gradients(targets, inputs, input_gradients=None):
    """Forward mode of automatic differentiation.

    .. note::
        **ONLY available in the static mode and primitive operators.**

    Args:
        targets: The target tensor or tensors
        inputs: The input tensor or tensors
        input_gradients: The gradient Tensor or Tensors of inputs which has 
            the same shape with inputs, Defaults to None, in this case is 
            equivalent to all ones .

    Returns:
        target_gradients (Tensor|Sequence[Tensor]): The gradients for targets.

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
                y_grad = paddle.incubate.autograd.forward_gradients(y, x)
                paddle.incubate.autograd.prim2orig()

            exe = paddle.static.Executor()
            exe.run(startup_program)
            y_grad = exe.run(main_program, feed={'x': np.array([2.]).astype('float32')}, fetch_list=[y_grad])
            print(y_grad)
            # [array([4.], dtype=float32)]

            paddle.incubate.autograd.disable_prim()
            paddle.disable_static()
    """
    if not prim_utils.prim_enabled():
        raise RuntimeError('forward_gradients must be running on primitive'
                           'operators, use enable_prim to turn it on.')

    if not isinstance(targets, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected targets is Tensor|Sequence[Tesnor], '
                        f'but got {type(targets)}.')

    if not isinstance(inputs, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected inputs is Tensor|Sequence[Tesnor], '
                        f'but got {type(inputs)}.')

    ys, xs, xs_dot = tensor_utils.as_tensors(targets), tensor_utils.as_tensors(
        inputs), tensor_utils.as_tensors(input_gradients)

    block = framework.default_main_program().current_block()
    if any(x.block != block for x in xs + ys):
        raise RuntimeError(
            'Variable in inputs and targets should exist in current block of '
            'main program.')

    primx.orig2prim(block)
    ad = primx.Transform(ys[0].block)
    _, ys_dot = ad.linearize(xs, ys, xs_dot)

    return ys_dot[0] if isinstance(targets, framework.Variable) else ys_dot
