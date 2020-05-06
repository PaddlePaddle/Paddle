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

__all__ = ['partial_program_from', 'PartialProgramLayer']


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
        self._infer_program = main_program
        self._train_program = self._append_backward_desc()
        # switch infer or train by train() and eval()
        self._trace_program = None
        self._params = parameters
        self._set_grad_type(self._params)
        self._is_test = is_test
        self._inner_scope = core.Scope()
        # set default mode to train
        self.train()

    @switch_to_static_graph
    def _append_backward_desc(self):
        program = self._infer_program.clone()
        targets = []
        for out in self.outputs:
            if isinstance(out, framework.Variable):
                targets.append(program.global_block().var(out.name))

        if targets:
            backward.gradients(targets=targets, inputs=[])

        return program

    def train(self):
        self._is_test = False
        self._trace_program = self._train_program

    def eval(self):
        self._is_test = True
        self._trace_program = self._infer_program

    def forward(self, inputs):
        in_vars, out_vars, tmp_scope_vec = self._prepare(inputs)

        framework._dygraph_tracer().trace_op(
            type='run_program',
            inputs={
                'X': valid_vars(in_vars),
                'Params': valid_vars(self._params)
            },
            outputs={'Out': valid_vars(out_vars),
                     'OutScope': tmp_scope_vec},
            attrs={
                'global_block': self._trace_program.desc.block(0),
                'start_op_index': 0,
                'end_op_index': self._infer_program.desc.block(0).op_size(),
                'is_test': self._is_test
            })

        outs = out_vars
        if len(outs) == 1:
            outs = outs[0]
        return outs

    def _prepare(self, inputs):
        """
        prepare inputs, outputs, attrs
        """
        assert isinstance(inputs, (tuple, list))
        # Convert variable into VarBase and feed in training data.
        input_vars = []
        for i, value in enumerate(inputs):
            if isinstance(value, np.ndarray):
                var = core.VarBase(
                    value=value,
                    name=self.inputs[i].desc.name(),
                    persistable=False,
                    place=framework._current_expected_place(),
                    zero_copy=True)
            elif isinstance(value, core.VarBase):
                var = value
                var.name = self.inputs[i].desc.name()
            else:
                continue
            input_vars.append(var)
        # Create VarBase to receive output data.
        out_vars = []
        for var in self.outputs:
            if not isinstance(var, framework.Variable):
                continue
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

        return input_vars, out_vars, tmp_scope_vec

    def _set_grad_type(self, params):
        # NOTE: if user set sparse gradient mode, the param's gradient
        # will be SelectedRows, not LoDTensor. But tracer will just
        # set param grad VarBase by forward VarBase(LoDTensor)
        # If we don't change grad_var type here, RunProgramOp need
        # transform SelectedRows to LoDTensor forcibly, it may not
        # be user wanted result.
        for param in params:
            grad_name = param.name + core.grad_var_suffix()
            grad_var = self._train_program.desc.block(0).find_var(
                cpt.to_bytes(grad_name))
            # NOTE: cannot find var desc maybe no problem, such as in batch_norm
            if grad_var is None:
                continue
            param._set_grad_type(grad_var.type())


def valid_vars(vars):
    """
    Note: run_program_op.InferShape requires `X`/'Out' not be null.
    But it's common in dy2static, fake varBase is created to handle the
    problem.
    """
    if vars:
        return vars
    return [
        core.VarBase(
            value=[1],
            name='Fake_var',
            place=framework._current_expected_place())
    ]


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
    inputs = program_cache.inputs
    if inputs and isinstance(inputs[0], layers.Layer):
        inputs = inputs[1:]

    return PartialProgramLayer(program_cache.main_program, inputs,
                               program_cache.outputs, program_cache.parameters)
