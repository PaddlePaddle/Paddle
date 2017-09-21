# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

# recurrent_units.py
# Version 2.0
#
# Some recurrent units can be used in recurrent layer group,
#   to use these units, import this module in your config_file:
#     import trainer.recurrent_units
#
# The modules in this file are DEPRECATED.
# If you would like to use lstm/gru
# please use the functions defined in paddle.trainer_config_helpers.

from paddle.trainer.config_parser import *


# long short term memory, can be used in recurrent machine
# *inputs* must be a list of Projections, for example:
#   inputs = [FullMatrixProjection("input_layer_name")],
# *para_prefix* defines parameter names, if the *para_prefix* of
#   two LstmRecurrentUnit is same, they share same parameters
# *out_memory* can be defined outside if it's used outside
def LstmRecurrentUnit(name,
                      size,
                      active_type,
                      state_active_type,
                      gate_active_type,
                      inputs,
                      para_prefix=None,
                      error_clipping_threshold=0,
                      out_memory=None):

    if para_prefix is None:
        para_prefix = name
    if out_memory is None:
        out_memory = Memory(name=name, size=size)

    state_memory = Memory(name=name + "_" + "state", size=size)

    Layer(
        name=name + "_" + "input_recurrent",
        type="mixed",
        size=size * 4,  #(input_s, input_gate, forget_gate, output_gate)
        error_clipping_threshold=error_clipping_threshold,
        bias=Bias(
            initial_std=0, parameter_name=para_prefix + "_input_recurrent.b"),
        inputs=inputs + [
            FullMatrixProjection(
                out_memory, parameter_name=para_prefix + "_input_recurrent.w"),
        ], )
    LstmStepLayer(
        name=name,
        size=size,
        bias=Bias(parameter_name=para_prefix + "_check.b"),
        inputs=[name + "_" + "input_recurrent", state_memory],
        active_type=active_type,
        active_gate_type=gate_active_type,
        active_state_type=state_active_type, )
    GetOutputLayer(
        name=name + "_" + "state",
        size=size,
        inputs=Input(
            name, input_layer_argument="state"), )


def LstmRecurrentUnitNaive(name,
                           size,
                           active_type,
                           state_active_type,
                           gate_active_type,
                           inputs,
                           para_prefix=None,
                           error_clipping_threshold=0,
                           out_memory=None):

    if para_prefix is None:
        para_prefix = name
    if out_memory is None:
        out_memory = Memory(name=name, size=size)

    state_memory = Memory(name=name + "_" + "state", size=size)

    Layer(
        name=name + "_" + "input_recurrent",
        type="mixed",
        size=size * 4,  #(input_s, input_gate, forget_gate, output_gate)
        error_clipping_threshold=error_clipping_threshold,
        bias=Bias(
            initial_std=0, parameter_name=para_prefix + "_input_recurrent.b"),
        inputs=inputs + [
            FullMatrixProjection(
                out_memory, parameter_name=para_prefix + "_input_recurrent.w"),
        ], )
    ExpressionLayer(
        name=name + "_" + "input_s",
        size=size,
        active_type=active_type,
        inputs=[
            IdentityOffsetProjection(
                name + "_" + "input_recurrent", offset=0)
        ], )
    ExpressionLayer(
        name=name + "_" + "input_gate",
        active_type=gate_active_type,
        inputs=[
            IdentityOffsetProjection(
                name + "_" + "input_recurrent", offset=size), DotMulProjection(
                    state_memory, parameter_name=para_prefix + "_input_check.w")
        ], )
    ExpressionLayer(
        name=name + "_" + "forget_gate",
        active_type=gate_active_type,
        inputs=[
            IdentityOffsetProjection(
                name + "_" + "input_recurrent", offset=size * 2),
            DotMulProjection(
                state_memory, parameter_name=para_prefix + "_forget_check.w")
        ], )
    ExpressionLayer(
        name=name + "_" + "state",
        inputs=[
            DotMulOperator([name + "_" + "input_s", name + "_" + "input_gate"]),
            DotMulOperator([state_memory, name + "_" + "forget_gate"]),
        ], )
    ExpressionLayer(
        name=name + "_" + "output_gate",
        active_type=gate_active_type,
        inputs=[
            IdentityOffsetProjection(
                name + "_" + "input_recurrent", offset=size * 3),
            DotMulProjection(
                name + "_" + "state",
                parameter_name=para_prefix + "_output_check.w")
        ], )
    ExpressionLayer(
        name=name + "_" + "state_atv",
        active_type=state_active_type,
        inputs=IdentityProjection(name + "_" + "state"), )
    ExpressionLayer(
        name=name,
        inputs=DotMulOperator(
            [name + "_" + "state_atv", name + "_" + "output_gate"]), )


# like LstmRecurrentUnit, but it's a layer group.
# it is equivalent to LstmLayer
def LstmRecurrentLayerGroup(name,
                            size,
                            active_type,
                            state_active_type,
                            gate_active_type,
                            inputs,
                            para_prefix=None,
                            error_clipping_threshold=0,
                            seq_reversed=False):

    input_layer_name = name + "_" + "transform_input"
    Layer(
        name=input_layer_name,
        type="mixed",
        size=size * 4,
        active_type="",
        bias=False,
        inputs=inputs, )

    RecurrentLayerGroupBegin(
        name + "_layer_group",
        in_links=[input_layer_name],
        out_links=[name],
        seq_reversed=seq_reversed)

    LstmRecurrentUnit(
        name=name,
        size=size,
        active_type=active_type,
        state_active_type=state_active_type,
        gate_active_type=gate_active_type,
        inputs=[IdentityProjection(input_layer_name)],
        para_prefix=para_prefix,
        error_clipping_threshold=error_clipping_threshold, )

    RecurrentLayerGroupEnd(name + "_layer_group")


# gated recurrent unit, can be used in recurrent machine
# *inputs* should be a list of Projections, for example:
#   inputs = [FullMatrixProjection("input_layer_name")],
# *para_prefix* defines parameter names, if the *para_prefix* of
#   two GatedRecurrentUnit is same, they share same parameters
# *out_memory* can be defined outside if it's used outside


def GatedRecurrentUnit(name,
                       size,
                       active_type,
                       gate_active_type,
                       inputs,
                       para_prefix=None,
                       error_clipping_threshold=0,
                       out_memory=None):
    if type_of(inputs) == str:  #only used by GatedRecurrentLayerGroup
        input_layer_name = inputs
    else:
        input_layer_name = name + "_" + "transform_input"
        Layer(
            name=input_layer_name,
            type="mixed",
            size=size * 3,
            active_type="",
            bias=False,
            inputs=inputs, )

    if para_prefix is None:
        para_prefix = name
    if out_memory is None:
        out_memory = Memory(name=name, size=size)

    GruStepLayer(
        name=name,
        size=size,
        bias=Bias(parameter_name=para_prefix + "_gate.b"),
        inputs=[
            input_layer_name, Input(
                out_memory, parameter_name=para_prefix + "_gate.w")
        ],
        active_type=active_type,
        active_gate_type=gate_active_type, )


def GatedRecurrentUnitNaive(name,
                            size,
                            active_type,
                            gate_active_type,
                            inputs,
                            para_prefix=None,
                            error_clipping_threshold=0,
                            out_memory=None):

    if type_of(inputs) == str:  #only used by GatedRecurrentLayerGroup
        input_layer_name = inputs
    else:
        input_layer_name = name + "_" + "transform_input"
        Layer(
            name=input_layer_name,
            type="mixed",
            size=size * 3,
            active_type="",
            bias=False,
            inputs=inputs, )

    if para_prefix is None:
        para_prefix = name
    if out_memory is None:
        out_memory = Memory(name=name, size=size)

    Layer(
        name=name + "_" + "update_gate",
        type="mixed",
        size=size,
        active_type=gate_active_type,
        error_clipping_threshold=error_clipping_threshold,
        bias=Bias(
            initial_std=0, parameter_name=para_prefix + "_update_gate.b"),
        inputs=[
            IdentityOffsetProjection(
                input_layer_name, offset=0), FullMatrixProjection(
                    out_memory, parameter_name=para_prefix + "_update_gate.w")
        ], )
    Layer(
        name=name + "_" + "reset_gate",
        type="mixed",
        size=size,
        active_type=gate_active_type,
        error_clipping_threshold=error_clipping_threshold,
        bias=Bias(
            initial_std=0, parameter_name=para_prefix + "_reset_gate.b"),
        inputs=[
            IdentityOffsetProjection(
                input_layer_name, offset=size), FullMatrixProjection(
                    out_memory, parameter_name=para_prefix + "_reset_gate.w")
        ], )
    ExpressionLayer(
        name=name + "_" + "reset_output",
        inputs=DotMulOperator([out_memory, name + "_" + "reset_gate"]), )
    Layer(
        name=name + "_" + "output_candidate",
        type="mixed",
        size=size,
        active_type=active_type,
        error_clipping_threshold=error_clipping_threshold,
        bias=Bias(
            initial_std=0, parameter_name=para_prefix + "_output_candidate.b"),
        inputs=[
            IdentityOffsetProjection(
                input_layer_name, offset=size * 2), FullMatrixProjection(
                    name + "_" + "reset_output",
                    parameter_name=para_prefix + "_output_candidate.w")
        ], )
    ExpressionLayer(  #element-wise interpolation
        name=name,
        inputs=[
            IdentityProjection(out_memory),
            DotMulOperator(
                [out_memory, name + "_" + "update_gate"], scale=-1.0),
            DotMulOperator(
                [name + "_" + "output_candidate", name + "_" + "update_gate"]),
        ], )


# like GatedRecurrentUnit, but it's a layer group.
# it is equivalent to GatedRecurrentLayer.
def GatedRecurrentLayerGroup(name,
                             size,
                             active_type,
                             gate_active_type,
                             inputs,
                             para_prefix=None,
                             error_clipping_threshold=0,
                             seq_reversed=False):

    input_layer_name = name + "_" + "transform_input"
    Layer(
        name=input_layer_name,
        type="mixed",
        size=size * 3,
        active_type="",
        bias=False,
        inputs=inputs, )

    RecurrentLayerGroupBegin(
        name + "_layer_group",
        in_links=[input_layer_name],
        out_links=[name],
        seq_reversed=seq_reversed)

    GatedRecurrentUnit(
        name=name,
        size=size,
        active_type=active_type,
        gate_active_type=gate_active_type,
        inputs=input_layer_name,  #transform outside
        para_prefix=para_prefix,
        error_clipping_threshold=error_clipping_threshold, )

    RecurrentLayerGroupEnd(name + "_layer_group")
