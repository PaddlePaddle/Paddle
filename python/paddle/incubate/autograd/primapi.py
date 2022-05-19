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

from paddle.fluid import framework
from paddle.incubate.autograd import primx, utils


@framework.static_only
def gradients(targets, inputs, target_gradients=None):
    """Reverse mode of automatic differentiation.

    .. note::
        **ONLY available in the static mode and primitive operators.**

    Args:
        targets (Tensor|Sequence[Tensor]): The target Tensor or Tensors.
        inputs (Tensor|Sequence[Tensor]): The input Tensor or Tensors.
        target_gradients (Tensor|Sequence[Tensor]): The gradient Tensor or 
            Tensors of targets which has the same shape with targets, Defaults 
            to None, in this case is equivalent to all ones .

    Returns:
        input_gradients (Tensor|Tensors): The gradients for inputs. 

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
                x_grad = paddle.incubate.autograd.gradients(y, x)
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
        raise RuntimeError('gradients must be running on primitive operators, '
                           'use enable_prim to turn on it.')

    if not isinstance(targets, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected targets is Tensor|Sequence[Tesnor], '
                        f'but got {type(targets)}.')

    if not isinstance(inputs, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected inputs is Tensor|Sequence[Tesnor], '
                        f'but got {type(inputs)}.')

    ys, xs, ys_bar = utils.to_tensors(targets), utils.to_tensors(
        inputs), utils.to_tensors(target_gradients)
    block = ys[0].block
    # TODO(Tongxin) without any prior knowledge about whether the program
    # is completely lowered to primitive ops, it's mandatory to run the lowering
    # pass once and again. This is obviously inefficient and needs to be
    # optimized.
    primx.orig2prim(block)

    ad = primx.Transform(block)

    xs_dot, ys_dot = ad.linearize(xs, ys)
    if any(var is None for var in ys_dot):
        assert False, f'Gradients cannot be computed. The given output `ys` does not depend on input `xs`.'
    ys_bar, xs_bar = ad.transpose(ys_dot, xs_dot, ys_bar)
    # remove xs_dot and their constructor ops

    op_indexes = []
    for var in xs_dot:
        if var is not None:
            op_index = block.ops.index(var.op)
            assert op_index >= 0, f'op_index should be greater than or equal to 0, but op_index={op_index}.'
            op_indexes.append(op_index)

    ad.erase_ops(sorted(op_indexes))
    ad.erase_dots(xs_dot)

    return xs_bar


@framework.static_only
def forward_gradients(targets, inputs, input_gradients=None):
    """Forward mode of automatic differentiation.

    .. note::
        **ONLY available in the static mode and primitive operators.**

    Args:
        targets: The target tensor or tensors
        inputs: The input tensor or tensors
        target_gradients: The gradient Tensor or Tensors of targets which has 
            the same shape with targets, Defaults to None, in this case is 
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
                x.stop_gradients = False
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
    if not utils.prim_enabled():
        raise RuntimeError('gradients must be running on primitive operators, '
                           'use enable_prim to turn on it.')

    if not isinstance(targets, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected targets is Tensor|Sequence[Tesnor], '
                        f'but got {type(targets)}.')

    if not isinstance(inputs, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected inputs is Tensor|Sequence[Tesnor], '
                        f'but got {type(inputs)}.')

    ys, xs, xs_dot = utils.to_tensors(targets), utils.to_tensors(
        inputs), utils.to_tensors(input_gradients)
    primx.orig2prim(ys[0].block)
    ad = primx.Transform(ys[0].block)
    _, ys_dot = ad.linearize(xs, ys, input_gradients)
    return ys_dot[0] if isinstance(ys, framework.Variable) else ys_dot
