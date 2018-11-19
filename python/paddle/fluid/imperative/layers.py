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

import sys
import numpy as np

from paddle.fluid import core
from paddle.fluid import framework

__all__ = ['PyLayer']


class PyLayer(core.Layer):
    def __init__(self):
        self._scope = core.Scope()

    def __call__(self, inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, tuple):
            inputs = [inputs]

        var_inputs = []
        for x in inputs:
            if isinstance(x, np.ndarray):
                tensor = core.LoDTensor()
                tensor.set(x, core.CPUPlace())
                x = framework.Variable(
                    framework.default_main_program().current_block(),
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    name=None,
                    shape=x.shape,
                    dtype=x.dtype)
            elif not isinstance(x, framework.Variable):
                raise ValueError("not var or ndarray %s" % type(x))
            self._scope.var(x.name)
            var_inputs.append(x)
        outputs = self.forward(var_inputs)
        for out in outputs:
            self._scope.var(out.name)
        return outputs

    def forward(self, inputs):
        print("at python.")
        return []
