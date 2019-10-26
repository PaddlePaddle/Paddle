# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ['trace']

from .base import program_desc_tracing_guard
from .layers import Layer
from paddle.fluid.framework import Program, Block, Variable, _dygraph_tracer, dygraph_only, _dygraph_guard


def create_program_from_desc(program_desc):
    program = Program()
    program.desc = program_desc
    program.blocks = [Block(program, 0)]
    program._sync_with_cpp()
    return program


def _extract_vars(inputs, result_list):
    if isinstance(inputs, Variable):
        result_list.append(inputs._ivar)

    if isinstance(inputs, (list, tuple)):
        for var in inputs:
            _extract_vars(var, result_list)


def extract_vars(inputs):
    result_list = []
    _extract_vars(inputs, result_list)
    return result_list


@dygraph_only
def trace(layer, inputs, feed_names=None, fetch_names=None):
    """
    Trace dygraph network into a :code:`Program`. The returned :code:`Program`
    can be run in static graph mode. This method would simply record all
    operators in the network with :code:`inputs` . Users should guarantee that
    the traced dygraph network is independent with input data, input shapes,
    and would not be changed between different batches. Otherwise, the traced
    result may be different.

    Parameters:
        layer(Layer): the layer to be traced.
        inputs(list): the input arguments of :code:`layer.forward()` method.
        feed_names(list(str), optional): the input variable names in the 
                traced :code:`Program` corresponding to :code:`inputs` . If it
                is None, the variable name of :code:`inputs` would be used. 
                It is suggested that users should set :code:`feed_names` 
                manually. Otherwise, the input variable names would be 
                different between different batches. Default None.
        fetch_names(list(str), optional): the output variable names in the 
                traced :code:`Program` corresponding to the output variables
                of :code:`layer.forward()` method. If it is None, the variable
                name of the outputs of :code:`layer.forward()` would be used.
                It is suggested that users should set :code:`fetch_names`
                manually. Otherwise, the output variable names would be
                different between different batches. Default None.
                
    Returns:
        A tuple of 2 items, whose first item is the outputs of 
        :code:`layer.forward()` method, and second item is the traced 
        :code:`Program` .

    Examples:

        .. code-blocks: python:

            import paddle.fluid as fluid
            from paddle.fluid.dygraph import FC, to_variable
            import paddle.fluid.dygraph.jit as jit
            import numpy as np

            class ExampleLayer(fluid.dygraph.Layer):
                def __init__(self, name_scope):
                    super(ExampleLayer, self).__init__(name_scope)
                    self._fc = FC(self.full_name(), 10)

                def forward(self, input):
                    return self._fc(input)

            with fluid.dygraph.guard():
                layer = ExampleLayer("example_layer")
                in_np = np.random.random([2, 3]).astype('float32')
                in_var = to_variable(in_np)
                out, program = jit.trace(layer, inputs=[in_var],
                                         feed_names=['input'],
                                         fetch_names=['fc_out'])

    """
    assert isinstance(layer, Layer)

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if feed_names is None:
        feed_names = []

    if fetch_names is None:
        fetch_names = []

    tracer = _dygraph_tracer()._get_program_desc_tracer()

    var_list = extract_vars(inputs)
    tracer.set_feed_vars(var_list, feed_names)

    with program_desc_tracing_guard(True):
        original_outputs = layer(*inputs)
        if not isinstance(original_outputs, (list, tuple)):
            outputs = [original_outputs]
        else:
            outputs = original_outputs
        out_vars = [var._ivar for var in outputs]

        tracer.set_fetch_vars(out_vars, fetch_names)
        tracer.set_name_prefix('t_')

        program_desc = tracer.create_program_desc()
        tracer.reset()

    with _dygraph_guard(None):
        program = create_program_from_desc(program_desc)

    return original_outputs, program
