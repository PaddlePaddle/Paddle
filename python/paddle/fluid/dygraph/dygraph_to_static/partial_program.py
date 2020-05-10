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


class PartialProgramLayer(layers.Layer):
    """
    PartialProgramLayer wraps all the ops from layers decorated by `@declarative`
    and execute them as a static subgraph.

    .. note::
        **1. This is a very low level API. Users should not use this API
             directly. Please use `partial_program_from(concrete_program)`
             to create it.
        **2. LoDTensorArray is not currently supported in the output.

    Args:
        main_program(Program): The main program that contains ops need to be executed.
        inputs(list[Variable]): The input list of the decorated function by `@declarative`.
        outputs(list[Variable]): The output list of the decorated function by `@declarative`.
        parameters(list[VarBase]|None): All trainable parameters included in the program. Default None.

    Returns:
        Layer: A Layer object that run all ops internally in static mode.
    """

    def __init__(self, main_program, inputs, outputs, parameters=None):
        super(PartialProgramLayer, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self._params = parameters if parameters is not None else []
        # Check all params from main program can be found in self._params:
        # 1. parameter in self._params should be type `framework.ParamBase` which are created in dygraph.
        # 2. parameter from transformed program shall be found in self._params.
        #    Because they share same data with ParamBase of original dygraph.
        self._check_params_all_inited(main_program)

        self._infer_program = main_program
        self._train_program = self._append_backward_desc()
        # Switch infer or train by train() and eval()
        self._trace_program = None
        self._set_grad_type(self._params)
        self._inner_scope = core.Scope()
        # Set default mode to train
        self.train()

    @switch_to_static_graph
    def _append_backward_desc(self):
        program = self._infer_program.clone()
        targets = []
        for out in self.outputs:
            if isinstance(out, framework.Variable):
                targets.append(program.global_block().var(out.name))

        if targets and self._params:
            backward.gradients(targets=targets, inputs=[])

        return program

    def train(self):
        # self.training is inherited from layers.Layer
        self.training = True
        self._trace_program = self._train_program

    def eval(self):
        self.training = False
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
                'is_test': not self.training
            })

        outs = out_vars
        if len(outs) == 1:
            outs = outs[0]
        return outs

    def _prepare(self, inputs):
        """
        Prepare inputs, outputs, attrs.
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

        # Hold forward variables
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

    def _check_params_all_inited(self, main_program):
        """
        Check all params from main program are already initialized, see details as follows:
            1. all parameters in self._params should be type `framework.ParamBase` which are created in dygraph.
            2. all parameters from transformed program can be found in self._params.
               Because they share same data with ParamBase of original dygraph.
        """
        if not isinstance(self._params, (list, tuple)):
            raise TypeError(
                "Type of self._params in PartialProgramLayer should be list or tuple, but received %s."
                % type(self._params))

        params_name_set = set()
        for i, param in enumerate(self._params):
            if not isinstance(param, framework.ParamBase):
                raise TypeError(
                    'Type of self._params[{}] in PartialProgramLayer should be framework.ParamBase, but received {}.'.
                    format(i, type(param)))
            params_name_set.add(param.name)

        for block in main_program.blocks:
            for name, var in block.vars.items():
                if isinstance(var, framework.Parameter):
                    if name not in params_name_set:
                        raise ValueError(
                            "\n\tWe don't support to define layer with parameters in the function "
                            "decorated by `@declarative`.\n\tBecause that will re-defined parameters "
                            "every time when you run the function.\n\t"
                            "But we found parameter(%s) was created in the decorated function.\n\t"
                            "Please define the layer with parameters in `__init__` function."
                            % name)


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


def partial_program_from(concrete_program):
    inputs = concrete_program.inputs
    if inputs and isinstance(inputs[0], layers.Layer):
        inputs = inputs[1:]

    return PartialProgramLayer(concrete_program.main_program, inputs,
                               concrete_program.outputs,
                               concrete_program.parameters)
