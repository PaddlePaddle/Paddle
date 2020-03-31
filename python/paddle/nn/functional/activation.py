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
    # 'relu',
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
    # 'log_softmax',
]


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
          input = fluid.data(name="input", shape=[None, 4])
          output = functional.relu(input)
          place = fluid.CPUPlace()
          exe = fluid.Executor(place)
          exe.run(fluid.default_startup_program())
          input_data = np.array([1.0, 2.0, 3.0, 4.0]).astype('float32')
          output_data = exe.run(feed={"input": input_data},
                                fetch_list=[output[0]])
          print(output_data) # [0 , 0 , 1]
    """

    helper = LayerHelper("sigmoid", **locals())
    outputs = helper.create_variable_for_type_inference(input.dtype)
    helper.append_op(
        type='sigmoid', inputs={'X': [input]}, outputs={'Out': outputs})
    return outputs
