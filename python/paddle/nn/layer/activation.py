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

from ...fluid.dygraph import layers
from ...fluid import core
from ...fluid.framework import in_dygraph_mode
from .. import functional

# TODO: define activation functions of neural network  
__all__ = [
    # 'PReLU',
    'ReLU',
    'Sigmoid',
    # 'Softmax',
    'LogSoftmax',
]


class ReLU(layers.Layer):
    """
    ReLU Activation.

    .. math:

        out = max(x, 0)

    Parameters:
        inplace (bool, optional): If inplace is True, the input and output of 
            ``ReLU`` are the same variable. Otherwise, the input and output of
            ``ReLU`` are different variables. Default False. Note that if x is
            more than one OPs' input, inplace must be False.

    Returns:
        None

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.nn as nn
          import numpy as np

          data = np.array([-2, 0, 1]).astype('float32')
          my_relu = nn.ReLU()
          with fluid.dygraph.guard():
              data = fluid.dygraph.to_variable(data)
              res = my_relu(data)  # [0, 0, 1]
    """

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self._inplace = inplace

    def forward(self, input):
        return functional.relu(input, self._inplace)


class Sigmoid(layers.Layer):
    """
    Sigmoid Activation.
    
    .. math:

        output = \frac{1}{1 + e^{-input}}

    Parameters:
        inplace (bool, optional): If inplace is True, the input and output
            are the same variable. Otherwise, the input and output
            are different variables. Default False. Note that if x is
            more than one OPs' input, inplace must be False.
    
    Returns:
        None
    
    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.nn as nn
          import numpy as np
          input = fluid.data(name="input", shape=[None, 4])
          output = nn.Sigmoid()(input)
          place = fluid.CPUPlace()
          exe = fluid.Executor(place)
          exe.run(fluid.default_startup_program())
          input_data = np.array([1.0, 2.0, 3.0, 4.0]).astype('float32')
          output_data = exe.run(feed={"input": input_data},
                                fetch_list=[output])
          print(output_data) # [0.7310586, 0.880797, 0.95257413, 0.98201376]
    """

    def __init__(self, inplace=False):
        super(Sigmoid, self).__init__()
        self._inplace = inplace

    def forward(self, input):
        return functional.sigmoid(input, self._inplace)


class LogSoftmax(layers.Layer):
    """
    This operator implements the log_softmax layer. The calculation process is as follows:

    .. math::

        Out[i, j] = log(softmax(x)) 
                  = log(\\frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])})

    Parameters:
        axis (int, optional): The index of dimension to perform softmax calculations, it should be in
            range :math:`[-1, rank-1]`, while :math:`rank` is the rank of input variable. Default: None. 
            None and -1 means the last dimension.
        dtype (np.dtype|core.VarDesc.VarType|str): The desired data type of returned tensor. If specified,
            the input tensor is casted to dtype before the operation is performed. This is useful for
            preventing data type overflows. Default: None. Supported dtype: float32 or float64
 
    Returns:
        None

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.nn as nn
          import numpy as np

          data = np.array([[[-2.0, 3.0, -4.0, 5.0],
                            [3.0, -4.0, 5.0, -6.0],
                            [-7.0, -8.0, 8.0, 9.0]],
                           [[1.0, -2.0, -3.0, 4.0],
                            [-5.0, 6.0, 7.0, -8.0],
                            [6.0, 7.0, 8.0, 9.0]]]).astype('float32')
          my_log_softnmax = nn.LogSoftmax()
          with fluid.dygraph.guard():
              data = fluid.dygraph.to_variable(data)
              res = my_log_softnmax(data)
              # [[[ -7.1278396   -2.1278396   -9.127839    -0.12783948]
              #   [ -2.1270514   -9.127051    -0.12705144 -11.127051  ]
              #   [-16.313261   -17.313261    -1.3132617   -0.31326184]]
              #  [[ -3.0518122   -6.051812    -7.051812    -0.051812  ]
              #   [-12.313267    -1.3132664   -0.3132665  -15.313267  ]
              #   [ -3.4401896   -2.4401896   -1.4401896   -0.44018966]]]
    """

    def __init__(self, axis=None):
        super(LogSoftmax, self).__init__()
        self._axis = axis

    def forward(self, input):
        return functional.log_softmax(input, self._axis)
