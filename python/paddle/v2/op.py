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

import layer
import activation as act
from config_base import Layer
from paddle.trainer_config_helpers.attrs import is_compatible_with
from paddle.trainer_config_helpers.default_decorators import wrap_name_default

__all__ = []


def __register_unary_math_op__(op_name, act):
    def op(input, name=None):
        return layer.mixed(
            input=[layer.identity_projection(input=input)], name=name, act=act)

    op = wrap_name_default(op_name)(op)
    op.__doc__ = type(act).__doc__
    globals()[op_name] = op
    __all__.append(op_name)


__register_unary_math_op__('exp', act.Exp())
__register_unary_math_op__('log', act.Log())
__register_unary_math_op__('abs', act.Abs())
__register_unary_math_op__('sigmoid', act.Sigmoid())
__register_unary_math_op__('tanh', act.Tanh())
__register_unary_math_op__('square', act.Square())
__register_unary_math_op__('relu', act.Relu())
__register_unary_math_op__('sqrt', act.Sqrt())
__register_unary_math_op__('reciprocal', act.Reciprocal())
__register_unary_math_op__('softmax', act.Softmax())


def __add__(layeroutput, other):
    if is_compatible_with(other, float):
        return layer.slope_intercept(input=layeroutput, intercept=other)
    if not isinstance(other, Layer):
        raise TypeError("Layer can only be added with"
                        " another Layer or a number")
    if layeroutput.size == other.size:
        return layer.mixed(input=[
            layer.identity_projection(input=layeroutput),
            layer.identity_projection(input=other)
        ])
    if other.size != 1 and layeroutput.size != 1:
        raise TypeError("Two Layer can be added only if they have equal size"
                        " or one of their sizes is 1. sizes are %s and %s" %
                        (layeroutput.size, other.size))
    elif layeroutput.size == 1:
        tmp = layeroutput
        layeroutput = other
        other = tmp
    other = layer.repeat(other, layeroutput.size)
    return layer.mixed(input=[
        layer.identity_projection(input=layeroutput),
        layer.identity_projection(input=other)
    ])


Layer.__radd__ = __add__
Layer.__add__ = __add__


def __neg__(layeroutput):
    return layer.slope_intercept(input=layeroutput, slope=-1.0)


Layer.__neg__ = __neg__


def __sub__(layeroutput, other):
    if is_compatible_with(other, float):
        return layer.slope_intercept(input=layeroutput, intercept=other)
    if not isinstance(other, Layer):
        raise TypeError("Layer can only be subtracted with"
                        " another Layeroutput or a number")
    return __add__(layeroutput, -other)


Layer.__sub__ = __sub__


def __rsub__(layeroutput, other):
    neg = layer.slope_intercept(input=layeroutput, slope=-1.0)
    return __add__(neg, other)


Layer.__rsub__ = __rsub__


def __mul__(layeroutput, other):
    if is_compatible_with(other, float):
        return layer.slope_intercept(input=layeroutput, slope=other)
    if not isinstance(other, Layer):
        raise TypeError("Layer can only be multiplied with"
                        " another Layer or a number")
    elif layeroutput.size == 1:
        return layer.scaling(input=other, weight=layeroutput)
    elif other.size == 1:
        return layer.scaling(input=layeroutput, weight=other)
    else:
        raise TypeError("At least one of the operand of '*' must be a number"
                        " or a Layer with size=1")


Layer.__mul__ = __mul__
Layer.__rmul__ = __mul__
