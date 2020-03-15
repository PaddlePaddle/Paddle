# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from ... import core
from ... import framework
from ... import unique_name

__all__ = ["fp16_compression"]


def fp16_compression(param_and_grads):
    """
    Compress fp32 gradients to fp16 before allreduce.

    Args:
        param_and_grads(list): list of (param, grad) variable paris, param is ``Parameter``,
            grad is the gradient value corresponding to the parameter.
    Returns:
        list: new param_and_grads

    Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            from paddle.fluid.contrib import fp16_compression
            x = fluid.data(name='x', shape=[None, 10], dtype='float32')
            trans = fluid.layers.fc(x, 100)
            loss = fluid.layers.reduce_mean(trans)
            optimizer = fluid.optimizer.Adam(learning_rate=0.001)
            optimizer.register_gradients_hook(fp16_compression)
            ops, param_grads = optimizer.minimize(loss)
    """
    op_maker = core.op_proto_and_checker_maker

    new_param_and_grads = []  # param, grad, is_cast
    # cast grad from fp32->fp16 before allreduce,
    for param, grad in param_and_grads:
        if grad is None or grad.dtype != core.VarDesc.VarType.FP32:
            # no need cast
            new_param_and_grads.append((param, grad, False))
            continue

        op = grad.op
        block = grad.block
        var_attr = op.all_attrs()[op_maker.kOpRoleVarAttrName()]
        if param.name not in var_attr:
            new_param_and_grads.append((param, grad, False))
            continue

        # remove (param, grad) from op_role_var
        var_attr.remove(param.name)
        var_attr.remove(grad.name)
        if len(var_attr) > 1:
            op._set_attr(op_maker.kOpRoleVarAttrName(), var_attr)
        else:
            op._remove_attr(op_maker.kOpRoleVarAttrName())

        new_grad = block.create_var(
            name=unique_name.generate(grad.name + ".cast_fp16"),
            dtype=core.VarDesc.VarType.FP16,
            persistable=False,
            stop_gradient=True)

        with block.program._backward_role_guard():
            cast_op = block.append_op(
                type="cast",
                inputs={"X": grad},
                outputs={"Out": new_grad},
                attrs={
                    "in_dtype": core.VarDesc.VarType.FP32,
                    "out_dtype": core.VarDesc.VarType.FP16
                },
                stop_gradient=True)

            backward = op_maker.OpRole.Backward
            cast_op._set_attr(op_maker.kOpRoleAttrName(), backward)
            cast_op._set_attr(op_maker.kOpRoleVarAttrName(),
                              [param.name, new_grad.name])
            new_grad.op = cast_op

        new_param_and_grads.append((param, new_grad, True))

    ret_param_and_grads = []
    # cast grad from fp16->fp32 after allreduce.
    for param, grad, cast in new_param_and_grads:
        if not cast:
            ret_param_and_grads.append((param, grad))
            continue

        block = grad.block
        new_grad = block.create_var(
            name=unique_name.generate(grad.name + ".cast_fp32"),
            dtype=core.VarDesc.VarType.FP32,
            persistable=False,
            stop_gradient=True)

        with block.program._optimized_guard(
            [param, grad]), framework.name_scope('fp16_compressor'):
            cast_op = block.append_op(
                type="cast",
                inputs={"X": grad},
                outputs={"Out": new_grad},
                attrs={
                    "in_dtype": core.VarDesc.VarType.FP16,
                    "out_dtype": core.VarDesc.VarType.FP32
                },
                stop_gradient=True)
        ret_param_and_grads.append((param, new_grad))

    return ret_param_and_grads
