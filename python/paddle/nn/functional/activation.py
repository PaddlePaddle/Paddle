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

import warnings
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode, convert_np_dtype_to_dtype_
from ...fluid import core
from ...fluid.data_feeder import check_variable_and_dtype

# TODO: define activation functions of neural network  
__all__ = [
    # 'brelu',
    # 'elu',
    # 'erf',
    # 'gelu',
    # 'hard_shrink',
    # 'hard_sigmoid',
    # 'hard_swish',
    # 'hsigmoid',
    # 'leaky_relu',
    # 'logsigmoid',
    # 'maxout',
    # 'prelu',
    'relu',
    # 'relu6',
    # 'selu',
    'sigmoid',
    # 'soft_relu',
    # 'softmax',
    # 'softplus',
    # 'softshrink',
    # 'softsign',
    # 'swish',
    # 'tanh_shrink',
    # 'thresholded_relu',
    'log_softmax',
]


def relu(input, inplace=False, name=None):
    """
    ReLU Activation.

    .. math:

        out = max(x, 0)

    Parameters:
        input (Variable): The input variable. A multi-dimension Tensor with type float16, float32, or float64.
        inplace (bool, optional): If inplace is True, the input and output of ``ReLU`` are the same variable.
            Otherwise, the input and output of ``ReLU`` are different variables. Default: False. Note that if x is
            more than one OPs' input, inplace must be False.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Output of relu operator, a Tensor with shape same as input

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.nn.functional as functional
          import numpy as np

          data = np.array([-2, 0, 1]).astype('float32')
          with fluid.dygraph.guard():
              data = fluid.dygraph.to_variable(data)
              res = functional.relu(data)  # [0, 0, 1]
    """

    if in_dygraph_mode():
        if inplace:
            warnings.warn(
                "Inplace on ReLU is not allowed and will be discarded in dygraph mode currently."
            )
        return core.ops.relu(input)

    helper = LayerHelper('relu', **locals())

    outs = input if inplace else helper.create_variable_for_type_inference(
        input.dtype)
    helper.append_op(type='relu', inputs={'X': [input]}, outputs={'Out': outs})
    return outs


def sigmoid(input, name=None):
    """
    Sigmoid Activation.

    .. math:

        output = \frac{1}{1 + e^{-input}}
    
    Parameters:
        input (Variable): The input variable. A multi-dimension Tensor with type float16, float32, or float64.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
    
    Returns:
        Output of sigmoid operator, a Tensor with shape same as input
    
    Examples:
        .. code-block:: python
          
          import paddle.fluid as fluid
          import paddle.nn.functional as functional
          import numpy as np
          # In the static graph mode
          input = fluid.data(name="input", shape=[None, 4])
          output = functional.sigmoid(input)
          place = fluid.CPUPlace()
          exe = fluid.Executor(place)
          exe.run(fluid.default_startup_program())
          input_data = np.array([1.0, 2.0, 3.0, 4.0]).astype('float32')
          output_data = exe.run(feed={"input": input_data},
                                fetch_list=[output])
          print(output_data) # [0.7310586, 0.880797, 0.95257413, 0.98201376]
          # In the dynamic graph mode
          with fluid.dygraph.guard():
              input = fluid.dygraph.to_variable(input_data)
              output = functional.sigmoid(input)
              print(output) # [0.7310586, 0.880797, 0.95257413, 0.98201376]
    """

    check_variable_and_dtype(input, 'X', ['float16', 'float32', 'float64'],
                             'sigmoid')
    if in_dygraph_mode():
        return core.ops.sigmoid(input)

    helper = LayerHelper("sigmoid", **locals())
    outputs = helper.create_variable_for_type_inference(input.dtype)
    helper.append_op(
        type='sigmoid', inputs={'X': [input]}, outputs={'Out': outputs})
    return outputs


def log_softmax(input, axis=None, dtype=None, name=None):
    """
    This operator implements the log_softmax layer. The calculation process is as follows:

    .. math::

        Out[i, j] = log(softmax(x)) 
                  = log(\\frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])})

    Parameters:
        input (Variable): The input variable. A multi-dimension Tensor with type float32, or float64.
        axis (int, optional): The index of dimension to perform softmax calculations, it should be in
            range :math:`[-1, rank-1]`, while :math:`rank` is the rank of input variable. Default: None. 
            None and -1 means the last dimension.
        dtype (np.dtype|core.VarDesc.VarType|str): The desired data type of returned tensor. If specified,
            the input tensor is casted to dtype before the operation is performed. This is useful for
            preventing data type overflows. Default: None. Supported dtype: float32 or float64
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
 
    Returns:
        Variable: ``Tensor`` indicates the output of softmax. The data type and shape are the same as ``input``.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.nn.functional as F
          import numpy as np

          data = np.array([[[-2.0, 3.0, -4.0, 5.0],
                            [3.0, -4.0, 5.0, -6.0],
                            [-7.0, -8.0, 8.0, 9.0]],
                           [[1.0, -2.0, -3.0, 4.0],
                            [-5.0, 6.0, 7.0, -8.0],
                            [6.0, 7.0, 8.0, 9.0]]]).astype('float32')
          with fluid.dygraph.guard():
              data = fluid.dygraph.to_variable(data)
              res = F.log_softmax(data, -1)
              # [[[ -7.1278396   -2.1278396   -9.127839    -0.12783948]
              #   [ -2.1270514   -9.127051    -0.12705144 -11.127051  ]
              #   [-16.313261   -17.313261    -1.3132617   -0.31326184]]
              #  [[ -3.0518122   -6.051812    -7.051812    -0.051812  ]
              #   [-12.313267    -1.3132664   -0.3132665  -15.313267  ]
              #   [ -3.4401896   -2.4401896   -1.4401896   -0.44018966]]]
    """

    axis = -1 if axis is None else axis
    dtype = convert_np_dtype_to_dtype_(dtype) if dtype is not None else dtype

    if in_dygraph_mode():
        outs_cast = input if dtype is None \
            else core.ops.cast(input, 'in_dtype', input.dtype, 'out_dtype', dtype)
        outs_softmax = core.ops.softmax(outs_cast, 'axis', axis, 'use_cudnn',
                                        False)
        return core.ops.log(outs_softmax)

    helper = LayerHelper("log_softmax", **locals())

    outs_cast = input
    if dtype is not None:
        outs_cast = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='cast',
            inputs={'X': input},
            outputs={'Out': outs_cast},
            attrs={'in_dtype': input.dtype,
                   'out_dtype': dtype})

    outs_softmax = helper.create_variable_for_type_inference(outs_cast.dtype)
    helper.append_op(
        type='softmax',
        inputs={'X': outs_cast},
        outputs={'Out': outs_softmax},
        attrs={'axis': axis,
               'use_cudnn': False})

    outs_log = helper.create_variable_for_type_inference(outs_softmax.dtype)
    helper.append_op(
        type='log', inputs={'X': outs_softmax}, outputs={'Out': outs_log})

    return outs_log
