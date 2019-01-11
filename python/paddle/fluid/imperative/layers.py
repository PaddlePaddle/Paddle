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

__all__ = ['Layer', 'PyLayer']


class Layer(core.Layer):
    """Layers composed of operators."""

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

    def backward(self, *inputs):
        raise ValueError("Layer shouldn't implement backward")


class PyLayer(core.PyLayer):
    """Layers composed of user-defined python codes."""

    def __init__(self):
        super(PyLayer, self).__init__()

    @staticmethod
    def forward(inputs):
        raise NotImplementedError

    @staticmethod
    def backward(douts):
        raise NotImplementedError

    @classmethod
    def __call__(cls, inputs):
        tracer = framework._imperative_tracer()
        block = framework.default_main_program().current_block()
        inputs = [x._ivar for x in inputs]

        if not hasattr(cls, 'forward_id'):
            cls.forward_id = core.PyLayer.num_funcs() + 1
            PyLayer.register_func(cls.forward_id, cls.forward)
            cls.backward_id = core.PyLayer.num_funcs() + 1
            PyLayer.register_func(cls.backward_id, cls.backward)

        iop = core.OpBase()
        iop.forward_id = cls.forward_id
        iop.backward_id = cls.backward_id
        block.ops.append(iop)
        ivars = tracer.py_trace(iop, inputs, False)
        # ivars = core.PyLayer.apply(cls.forward, inputs)
        ret = []
        for ivar in ivars:
            tensor = ivar.value.get_tensor()
            py_var = framework.Variable(
                block,
                type=core.VarDesc.VarType.LOD_TENSOR,
                name=None,
                shape=tensor.shape(),
                dtype=tensor._dtype(),
                ivar=ivar)
            ret.append(py_var)
        return ret
