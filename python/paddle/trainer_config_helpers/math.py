# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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

from .layers import LayerOutput, mixed_layer, identity_projection, \
    slope_intercept_layer
from .attrs import is_compatible_with
from .default_decorators import *
import activations as act

__all__ = []

def register_unary_math_op(op_name, act):
    def op(input, name=None):
        return mixed_layer(input=[identity_projection(input=input)],
                           name=name,
                           act=act)
    op = wrap_name_default(op_name)(op)
    op.__doc__ = type(act).__doc__
    globals()[op_name] = op
    __all__.append(op_name)

register_unary_math_op('exp', act.ExpActivation())
register_unary_math_op('log', act.LogActivation())
register_unary_math_op('abs', act.AbsActivation())
register_unary_math_op('sigmoid', act.SigmoidActivation())
register_unary_math_op('tanh', act.TanhActivation())
register_unary_math_op('square', act.SquareActivation())

def add(layeroutput, other):
    if is_compatible_with(other, float):
        return slope_intercept_layer(input=layeroutput, intercept=other)
    assert isinstance(other, LayerOutput)
    return mixed_layer(input=[identity_projection(input=layeroutput),
                              identity_projection(input=other)])

LayerOutput.__radd__ = add
LayerOutput.__add__ = add

def sub(layeroutput, other):
    if is_compatible_with(other, float):
        return slope_intercept_layer(input=layeroutput, intercept=other)
    assert isinstance(other, LayerOutput)
    neg = slope_intercept_layer(input=other, slope=-1.0)
    return mixed_layer(input=[identity_projection(input=layeroutput),
                              identity_projection(input=neg)])

LayerOutput.__sub__ = sub

def rsub(layeroutput, other):
    neg = slope_intercept_layer(input=layeroutput, slope=-1.0)
    return add(neg, other)

LayerOutput.__rsub__ = rsub
