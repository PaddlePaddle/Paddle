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
import numpy as np
from paddle.fluid import framework, backward, core
from paddle.fluid.dygraph import layers
from paddle.fluid.dygraph.base import switch_to_static_graph
import paddle.compat as cpt

__all__ = ['partial_program_from']


class PartialProgramLayer(layers.Layer):
    def __init__(self,
                 main_program,
                 inputs,
                 outputs,
                 parameters=None,
                 is_test=False):
        super(PartialProgramLayer, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self._input_names = [var.name for var in inputs]
        self._output_names = [var.name for var in outputs]
        self._forward_program = main_program

        self._program = self._append_backward_desc()
        self._params = self._parse_parameter(parameters)
        # TODO: deal with is_test
        self._is_test = is_test
        self._inner_scope = core.Scope()

    @switch_to_static_graph
    def _append_backward_desc(self):
        program = self._forward_program.clone()
        # TODO: could the targets be in sub block?
        targets = []
        for out in self._output_names:
            targets.append(program.global_block().var(out))

        # Step 2. append backward
        backward.gradients(targets=targets, inputs=[])

        return program

    def _parse_parameter(self, parameters):
        params = []
        for param in parameters:
            if append_grad_suffix(param.name) in self._program.block(0).vars:
                params.append(param)
        return params

    def forward(self, inputs):
        # Step 1. prepare inputs, outputs, attrs
        # todo: only feed data right data
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        input_vars = []
        for i, value in enumerate(inputs):
            if not isinstance(value, (np.ndarray, core.VarBase)):
                print(
                    "The type of inputs.value in forward must be numpy array or Variable(VarBase), but received %s."
                    % type(value))
                continue
            # NOTE: In order to unify the API, firstly convert the input to VarBase
            if isinstance(value, np.ndarray):
                var = core.VarBase(
                    value=value,
                    name=self._input_names[i],
                    persistable=False,
                    place=framework._current_expected_place(),
                    zero_copy=True)  # TODO: zero_copy only used in CPU
            else:
                var = value
                # TODO: here may have important name set by user
                var.name = self._input_names[i]
            input_vars.append(var)

        out_vars = []
        for var in self.outputs:
            var_desc = var.desc
            var_base = core.VarBase(var_desc.dtype(),
                                    var_desc.shape(),
                                    var_desc.name(), var_desc.type(), False)
            out_vars.append(var_base)

        # hold forward variables
        tmp_scope_vec = core.VarBase(core.VarDesc.VarType.FP32, [],
                                     "program_out_scope",
                                     core.VarDesc.VarType.STEP_SCOPES, True)
        tmp_scope_vec.value().set_scope(self._inner_scope)

        # Step 2. run program by op
        # TODO: Params can be null.
        if not self._params:
            self._params = input_vars

        framework._dygraph_tracer().trace_op(
            type='run_program',
            inputs={'X': input_vars,
                    'Params': self._params},
            outputs={'Out': out_vars,
                     'OutScope': tmp_scope_vec},
            attrs={
                'global_block': self._program.desc.block(0),
                'start_op_index': 0,
                'end_op_index': self._forward_program.desc.block(0).op_size(),
                'is_test': self._is_test
            })

        # NOTE: [ why need set param's gradient type here ]
        # if user set sparse gradient mode, the param's gradient
        # will be SelectedRows, not LoDTensor. But tracer will just
        # set param grad VarBase by forward VarBase(LoDTensor)
        # If we don't change grad_var type here, RunProgramOp need
        # transform SelectedRows to LoDTensor forcely, it may not
        # be user wanted result.
        # if not self._set_type:
        for param in self._params:
            grad_name = param.name + core.grad_var_suffix()
            grad_var = self._program.desc.block(0).find_var(
                cpt.to_bytes(grad_name))
            # NOTE: cannot find var desc maybe no problem, such as in batch_norm
            if grad_var is None:
                continue
            param._set_grad_type(grad_var.type())

        # Step 3. prepare output, keep same form with inputs
        outs = out_vars
        if len(out_vars) == 1:
            outs = out_vars[0]
        return outs


def append_grad_suffix(name):
    """
    Append grad suffix to the given variable name
    e.g. x ==> x@GRAD
    """
    suffix = core.kGradVarSuffix()
    name = cpt.to_text(name)
    if suffix not in name:
        name = name + suffix
    return name


def partial_program_from(program_cache):
    return PartialProgramLayer(program_cache.main_program, program_cache.inputs,
                               program_cache.outputs, program_cache._parameters)
