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

from . import layers
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
def trace(module, inputs, feed_names=None, fetch_names=None):
    assert isinstance(module, Layer)

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
        original_outputs = module.__call__(*inputs)
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
