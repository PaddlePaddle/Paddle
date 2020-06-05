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

from .. import default_main_program
from .. import core

__all__ = ["SwapMemory"]


class SwapMemory(object):
    def __init__(self, swap_vars=None, optimizer=None):
        if swap_vars is None:
            raise ValueError('swap_vars must be')
        self._swap_vars = swap_vars
        self._optimizer = optimizer
        self.bwd_rename_var = {}

    def swap_forward(self):
        """
        add Swap Op from CPU to GPU for self._swap_vars
        """
        self._swap_vars_dict = {var.name: var for var in self._swap_vars}
        program = default_main_program()
        block = program.global_block()
        inserted_op_idx = []
        for op_idx, op in enumerate(block.ops):
            for var_name in op.desc.output_arg_names():
                if var_name not in self._swap_vars_dict:
                    continue
                var = self._swap_vars_dict[var_name]
                print("op_type: ", op.type)
                print("var name: ", var_name)
                swapped_var = block.create_var(
                    name=var_name + "@CPU",
                    dtype=var.dtype,
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    persistable=False,
                    stop_gradient=False)
                block._insert_op(
                    index=op_idx + 1,
                    type='swapmem_gpu2cpu',
                    inputs={'X': [var]},
                    outputs={'Out': [swapped_var]},
                    attrs={})
                self.bwd_rename_var[var_name] = var_name + "@CPU"
                inserted_op_idx.append(op_idx + 1)

    def swap_backward(self):
        program = default_main_program()
        block = program.global_block()
        BACKWARD = core.op_proto_and_checker_maker.OpRole.Backward
        inserted_op_idx = []
        for op_idx, op in enumerate(block.ops):
            role = op.attr('op_role')
            if not role & int(BACKWARD):
                continue
            for var_name in op.desc.input_arg_names():
                if var_name not in self._swap_vars_dict:
                    continue
                if not block.has_var(var_name + "@GPU"):
                    var = block.var(var_name + "@CPU")
                    swapped_var = block.create_var(
                        name=var_name + "@GPU",
                        dtype=var.dtype,
                        type=core.VarDesc.VarType.LOD_TENSOR,
                        persistable=False,
                        stop_gradient=False)
                    swap_op = block._insert_op(
                        index=op_idx,
                        type='swapmem_cpu2gpu',
                        inputs={'X': [var]},
                        outputs={'Out': [swapped_var]},
                        attrs={})
                    swap_op._set_attr('op_role', BACKWARD)
                    inserted_op_idx.append(op_idx)
                op.desc._rename_input(var_name, var_name + "@GPU")

    def minimize(self, *args, **kwargs):
        print("start forward")
        self.swap_forward()
        print("finish forward")
        optimize_ops, param_grads = self._optimizer.minimize(*args, **kwargs)
        print("finish minimize")
        self.swap_backward()
        print("finish backward")
        return optimize_ops, param_grads
