# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np
import six

from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_dtype, check_type

__all__ = ['data', 'InputSpec']


def data(name, shape, dtype=None, lod_level=0):
    """
    **Data Layer**

    This function creates a variable on the global block. The global variable
    can be accessed by all the following operators in the graph. The variable
    is a placeholder that could be fed with input, such as Executor can feed
    input into the variable. When `dtype` is None, the dtype
    will get from the global dtype by `paddle.get_default_dtype()`.

    Args:
       name (str): The name/alias of the variable, see :ref:`api_guide_Name`
           for more details.
       shape (list|tuple): List|Tuple of integers declaring the shape. You can
           set "None" or -1 at a dimension to indicate the dimension can be of any
           size. For example, it is useful to set changeable batch size as "None" or -1.
       dtype (np.dtype|str, optional): The type of the data. Supported
           dtype: bool, float16, float32, float64, int8, int16, int32, int64,
           uint8. Default: None. When `dtype` is not set, the dtype will get 
           from the global dtype by `paddle.get_default_dtype()`.
       lod_level (int, optional): The LoD level of the LoDTensor. Usually users
           don't have to set this value. For more details about when and how to
           use LoD level, see :ref:`user_guide_lod_tensor` . Default: 0.

    Returns:
        Variable: The global variable that gives access to the data.

    Examples:
        .. code-block:: python

          import numpy as np
          import paddle.fluid as fluid
          import paddle

          # Creates a variable with fixed size [3, 2, 1]
          # User can only feed data of the same shape to x
          # the dtype is not set, so it will set "float32" by
          # paddle.get_default_dtype(). You can use paddle.get_default_dtype() to 
          # change the global dtype
          x = paddle.static.data(name='x', shape=[3, 2, 1])

          # Creates a variable with changeable batch size -1.
          # Users can feed data of any batch size into y,
          # but size of each data sample has to be [2, 1]
          y = paddle.static.data(name='y', shape=[-1, 2, 1], dtype='float32')

          z = x + y

          # In this example, we will feed x and y with np-ndarray "1"
          # and fetch z, like implementing "1 + 1 = 2" in PaddlePaddle
          feed_data = np.ones(shape=[3, 2, 1], dtype=np.float32)

          exe = fluid.Executor(fluid.CPUPlace())
          out = exe.run(fluid.default_main_program(),
                        feed={
                            'x': feed_data,
                            'y': feed_data
                        },
                        fetch_list=[z.name])

          # np-ndarray of shape=[3, 2, 1], dtype=float32, whose elements are 2
          print(out)

    """
    helper = LayerHelper('data', **locals())
    check_type(name, 'name', (six.binary_type, six.text_type), 'data')
    check_type(shape, 'shape', (list, tuple), 'data')

    shape = list(shape)
    for i in six.moves.range(len(shape)):
        if shape[i] is None:
            shape[i] = -1

    if dtype:
        return helper.create_global_variable(
            name=name,
            shape=shape,
            dtype=dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            stop_gradient=True,
            lod_level=lod_level,
            is_data=True,
            need_check_feed=True)
    else:
        return helper.create_global_variable(
            name=name,
            shape=shape,
            dtype=paddle.get_default_dtype(),
            type=core.VarDesc.VarType.LOD_TENSOR,
            stop_gradient=True,
            lod_level=lod_level,
            is_data=True,
            need_check_feed=True)


class InputSpec(object):
    """
    Define input specification of the model.

    Args:
        name (str): The name/alias of the variable, see :ref:`api_guide_Name`
            for more details.
        shape (tuple(integers)|list[integers]): List|Tuple of integers
            declaring the shape. You can set "None" or -1 at a dimension
            to indicate the dimension can be of any size. For example,
            it is useful to set changeable batch size as "None" or -1.
        dtype (np.dtype|str, optional): The type of the data. Supported
            dtype: bool, float16, float32, float64, int8, int16, int32, int64,
            uint8. Default: float32.

    Examples:
        .. code-block:: python

        from paddle.static import InputSpec

        input = InputSpec([None, 784], 'float32', 'x')
        label = InputSpec([None, 1], 'int64', 'label')
    """

    def __init__(self, shape=None, dtype='float32', name=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def _create_feed_layer(self):
        return data(self.name, shape=self.shape, dtype=self.dtype)

    def __repr__(self):
        return '{}(shape={}, dtype={}, name={})'.format(
            type(self).__name__, self.shape, self.dtype, self.name)
