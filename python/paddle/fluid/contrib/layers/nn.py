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
"""
Contrib layers just related to the neural network.
"""

from __future__ import print_function

import numpy as np
import six
import os
import inspect
from paddle.fluid.layer_helper import LayerHelper

__all__ = ['fused_elemwise_activation', ]


def fused_elemwise_activation(x,
                              y,
                              functor_list,
                              axis=-1,
                              scale=0.0,
                              save_intermediate_out=True):
    """
    **Fused elementwise_add/mul and activation layers**

    This function computes an elementwise_add/mul cooperated with an activation.

    .. math::

        out = Unary(Binary(x, y))

    or

    .. math::

        out = Binary(x, Unary(y))

    Unary operators can be: `scale`, `relu`, `tanh`. Binary operators can be:
    `elementwise_add`, `elementwise_mul`.

    Args:
        x (Variable): left operation of the binary operator.
        y (Variable): right operator of the binary operator.
        functor_list (list of str): types of operator which will be executed
            by this layer. For example, ['elementwise_add', 'relu']
            (out = elementwise_add(x, relu(y))),
            or ['relu', 'elemmentwise_add'] (out = relu(elementwise_add(x, y))).
        axis (int32, default -1): axis of elementwise op.
        scale (float32, default 0): parameter of scale op.
        save_intermediate_out (bool, default True): whether to save the
            intermediate result, Unary(y) or Binary(x, y).

    Returns:
        Variable: The computation result.
    """
    if isinstance(functor_list, str):
        functor_list = functor_list.split(',')

    if not isinstance(functor_list, list) or len(functor_list) != 2:
        raise ValueError(
            'functor_list should be a list of str, and the length should be 2.')

    helper = LayerHelper('fused_elemwise_activation', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    intermediate_out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='fused_elemwise_activation',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out,
                 'IntermediateOut': intermediate_out},
        attrs={
            'axis': axis,
            'scale': scale,
            'save_intermediate_out': save_intermediate_out,
            'functor_list': functor_list
        })
    return out
