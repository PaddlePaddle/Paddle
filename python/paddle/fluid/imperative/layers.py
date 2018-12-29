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
    def __init__(self, dtype=core.VarDesc.VarType.FP32, name=None):
        self._once_built = False
        self._dtype = dtype

    def _build_once(self, inputs):
        pass

    def __call__(self, *inputs):
        if not self._once_built:
            self._build_once(*inputs)
            self._once_built = True

        outputs = self.forward(*inputs)

        return outputs

    def forward(self, *inputs):
        raise NotImplementedError
