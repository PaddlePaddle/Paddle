#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from ... import default_main_program
from ... import framework
from ... import layers
from ... import optimizer

__all__ = ["fp16_compression"]


class FP16Compressor(optimizer.Optimizer):
    """
    Compress all fp32 gradients to fp16 during allreduce.

    Args:
        optimizer (Optimizer): A common Optimizer object.

    """

    def __init__(self, optimizer):
        self._optimizer = optimizer

    def _append_cast_ops(self, param_and_grads):
        """
        Compress fp32 gradients to fp16 during allreduce.
        """
        main_program = default_main_program()
        block = main_program.global_block()
        op_maker = core.op_proto_and_checker_maker

        for param, grad in param_and_grads:
            if grad.dtype != core.VarDesc.VarType.FP32:
                continue

            op = grad.op
            var_attr = op.all_attrs()[op_maker.kOpRoleVarAttrName()]
            if param.name not in var_attr:
                continue

            var_attr.remove(param.name)
            var_attr.remove(grad.name)
            if len(var_attr) > 1:
                op._set_attr(op_maker.kOpRoleVarAttrName(), var_attr)
            else:
                op._remove_attr(op_maker.kOpRoleVarAttrName())

            cast_name = grad.name + '.cast_fp16'
            out_var = block.vars.get(cast_name)
            if out_var is None or out_var.dtype != core.VarDesc.VarType.FP16:
                out_var = block.create_var(
                    name=cast_name,
                    dtype=core.VarDesc.VarType.FP16,
                    persistable=False)

            # cast grad from fp32->fp16 before allreduce,
            with block.program._backward_role_guard():
                cast_op = block.append_op(
                    type="cast",
                    inputs={"X": grad},
                    outputs={"Out": out_var},
                    attrs={
                        "in_dtype": core.VarDesc.VarType.FP32,
                        "out_dtype": core.VarDesc.VarType.FP16
                    },
                    stop_gradient=True)

                backward = op_maker.OpRole.Backward
                cast_op._set_attr(op_maker.kOpRoleAttrName(), backward)
                cast_op._set_attr(op_maker.kOpRoleVarAttrName(),
                                  [param.name, out_var.name])

            # cast grad from fp16->fp32 after allreduce.
            with param.block.program._optimized_guard(
                [param, grad]), framework.name_scope('fp16_compressor'):
                cast_op = block.append_op(
                    type="cast",
                    inputs={"X": out_var},
                    outputs={"Out": grad},
                    attrs={
                        "in_dtype": core.VarDesc.VarType.FP16,
                        "out_dtype": core.VarDesc.VarType.FP32
                    },
                    stop_gradient=True)

    def backward(self, **kargs):
        return self._optimizer.backward(**kargs)

    def apply_optimize(self, **kargs):
        self._append_cast_ops(kargs['params_grads'])
        return self._optimizer.apply_optimize(**kargs)

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 grad_clip=None):
        params_grads = self.backward(
            loss=loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        if grad_clip:
            pass

        optimize_ops = self.apply_optimize(
            loss=loss,
            params_grads=params_grads,
            startup_program=startup_program)
        return optimize_ops, params_grads


def fp16_compression(optimizer):
    """
    Compress all fp32 gradients to fp16 during allreduce.

    Args:
        optimizer(Optimizer): A common Optimizer.

    Returns:
        An optimizer acting like a normal one but with fp16 comperssor training 
        enabled.

    Examples:
	.. code-block:: python

	    loss = network()
            optimizer = fluid.optimizer.Adam(learning_rate=0.001)
	
            optimizer = fluid.contrib.fp16_compression(
	              optimizer=optimizer)
	
            ops, param_grads = optimizer.minimize(loss)
    """

    return FP16Compressor(optimizer)
