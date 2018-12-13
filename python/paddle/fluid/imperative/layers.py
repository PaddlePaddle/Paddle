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
from paddle.fluid.imperative import base

__all__ = ['PyLayer']


class PyLayer(core.Layer):
    def __init__(self):
        self._built = False

    def __call__(self, inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, tuple):
            inputs = [inputs]

        var_inputs = []
        for x in inputs:
            py_var = base.to_variable(x)
            var_inputs.append(py_var)
        if not self._built:
            self._build_once(inputs)
            self._built = True

        outputs = self.forward(var_inputs)
        return outputs

    def _build_once(self, inputs):
        pass

    def forward(self, inputs):
        return []
