#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import warnings
import six
import os
import inspect
from ..fluid.layer_helper import LayerHelper
from ..fluid.initializer import Normal, Constant, NumpyArrayInitializer
from ..fluid.framework import Variable, OpProtoHolder, in_dygraph_mode, dygraph_only, _dygraph_tracer, default_main_program
from ..fluid import dygraph_utils
from ..fluid.param_attr import ParamAttr
from ..fluid import unique_name
from ..fluid import core, layers
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype

# TODO: define searching & indexing functions of a tensor  
__all__ = [
    #            'argmax',
    #            'argmin',
    #            'argsort',
    #            'has_inf',
    #            'has_nan',
    #            'masked_select',
    #            'topk',
    'where',
    #            'index_select',
    #            'nonzero',
    #            'sort'
]


def where(Condition, X, Y):
    """
    Return a tensor of elements selected from either $X$ or $Y$, depending on $Condition$.
    Args:
        Condition(Variable): A bool tensor with rank at least 1, the data type is bool.
        X(Variable): X is a Tensor Variable.
        Y(Variable): Y is a Tensor Variable.
    Returns:
        out : The tensor. 
    Examples:
        .. code-block:: python
          import paddle.tensor as tensor
          import paddle.fluid as fluid
          import numpy as np

          x = fluid.layers.data(name='x', shape=[4], dtype='float32')
          y = fluid.layers.data(name='y', shape=[4], dtype='float32')
          result = tensor.where(x>1, X=x, Y=y)
          exe = fluid.Executor(fluid.CPUPlace())
          exe.run(fluid.default_startup_program())
          x_i = np.array([0.9383, 0.1983, 3.2,1.2]).astype("float32")
          y_i = np.array([1.0, 1.0, 1.0,1.0]).astype("float32")
          out = exe.run(fluid.default_main_program(),feed={'x':x_i, 'y':y_i}, fetch_list=[result])
          print(out[0])
    """
    if in_dygraph_mode():
        X_shape = list(X.shape)
        Y_shape = list(Y.shape)
        if X_shape == Y_shape:
            inputs = {'Condition': [Condition], 'X': [X], 'Y': [Y]}
            outs = core.ops.where(inputs)
            return outs['Out'][0]
        else:
            cond_int = layers.cast(Condition, X.dtype)
            cond_not_int = layers.cast(layers.logical_not(Condition), X.dtype)
            out1 = layers.elementwise_mul(X, cond_int)
            out2 = layers.elementwise_mul(Y, cond_not_int)
            out = layers.elementwise_add(out1, out2)
            return out

    helper = LayerHelper("where", **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)

    check_type(Condition, 'Condition', (Variable), 'where')
    check_type(X, 'X', (Variable), 'where')
    check_type(Y, 'Y', (Variable), 'where')

    if isinstance(Condition, Variable):
        check_dtype(Condition.dtype, 'Condition', ['bool'], 'where',
                    '(When the type of Condition in where is Variable.)')
    if isinstance(X, Variable):
        check_dtype(X.dtype, 'X', ['float32', 'float64', 'int32', 'int64'],
                    'where', '(When the type of X in where is Variable.)')
    if isinstance(Y, Variable):
        check_dtype(Y.dtype, 'Y', ['float32', 'float64', 'int32', 'int64'],
                    'where', '(When the type of Y in where is Variable.)')
    X_shape = list(X.shape)
    Y_shape = list(Y.shape)

    if X_shape == Y_shape:
        helper.append_op(
            type='where',
            inputs={'Condition': Condition,
                    'X': X,
                    'Y': Y},
            outputs={'Out': [out]})
        return out
    else:
        cond_int = layers.cast(Condition, X.dtype)
        cond_not_int = layers.cast(layers.logical_not(Condition), X.dtype)
        out1 = layers.elementwise_mul(X, cond_int)
        out2 = layers.elementwise_mul(Y, cond_not_int)
        out = layers.elementwise_add(out1, out2)
        return out
