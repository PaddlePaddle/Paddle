# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import sys
import numpy as np

from paddle.fluid import core
from paddle.fluid import framework

__all__ = ['PyLayer']


@contextlib.contextmanager
def trace_scope(scope, block):
    tmp_scope = framework._imperative_tracer().scope
    tmp_block = framework._imperative_tracer().block
    framework._imperative_tracer().scope = scope
    framework._imperative_tracer().block = block
    yield
    framework._imperative_tracer().scope = tmp_scope
    framework._imperative_tracer().block = tmp_block


class PyLayer(core.Layer):
    def __init__(self):
        self._scope = core.Scope()
        self._block = framework.default_main_program().current_block()

    def __call__(self, inputs):
        with trace_scope(self._scope, self._block.desc):
            if not isinstance(inputs, list) and not isinstance(inputs, tuple):
                inputs = [inputs]

            var_inputs = []
            for x in inputs:
                if isinstance(x, np.ndarray):
                    py_var = framework.Variable(
                        self._block,
                        type=core.VarDesc.VarType.LOD_TENSOR,
                        name=None,
                        shape=x.shape,
                        dtype=x.dtype)
                    var = self._scope.var(py_var.name)
                    tensor = var.get_tensor()
                    tensor.set(x, core.CPUPlace())
                    var_inputs.append(py_var)
                elif isinstance(x, framework.Variable):
                    var_inputs.append(x)
                else:
                    raise ValueError("not var or ndarray %s" % type(x))
            outputs = self.forward(var_inputs)
            return outputs

    def forward(self, inputs):
        print("at python.")
        return []
