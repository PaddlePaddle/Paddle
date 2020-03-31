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
    # 'ReLU',
    'Sigmoid',
    # 'Softmax',
    # 'LogSoftmax',
]


class Sigmoid(layers.Layer):
    """
    Sigmoid Activation.
    .. math:
        output = \frac{1}{1 + e^{-input}}
    Parameters:
        None
    Returns:
        None
    Examples:
        .. code-block:: python
          import paddle.fluid as fluid
          import paddle.nn as nn
          import numpy as np
          input = fluid.data(name="input", shape=[None, 4])
          output = nn.Sigmoid(input)
          place = fluid.CPUPlace()
          exe = fluid.Executor(place)
          exe.run(fluid.default_startup_program())
          input_data = np.array([1.0, 2.0, 3.0, 4.0]).astype('float32')
          output_data = exe.run(feed={"input": input_data},
                                fetch_list=[output[0]])
          print(output_data) # [0, 0, 1]
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        return functional.sigmoid(input)
