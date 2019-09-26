#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

from . import core
from .layer_helper import LayerHelper

__all__ = ['data']


def data(name, shape, dtype='float32', type=core.VarDesc.VarType.LOD_TENSOR):
    """
    **Data Layer**

    This function creates a variable on the global block. The global variables
    can be accessed by all the following operators in the graph.

    Note: 
       `paddle.fluid.layers.data` is deprecated. It will be removed in a future
       version. Please use this `paddle.fluid.data`. 
       
       The `paddle.fluid.layers.data` set shape at compile time but does NOT
       check the shape or the dtype of feeded data, this `paddle.fluid.data`
       checks the shape and the dtype of data feeded by Executor or 
       ParallelExecutor during run time.

    Args:
       name (str): The name/alias of the variable, see :ref:`api_guide_Name`
           for more details.
       shape (list|tuple): List|Tuple of integers declaring the shape.
       dtype (np.dtype|VarType|str): The type of the data. Supported dtype:
           bool, float16, float32, float64, int8, int16, int32, int64, uint8.
       type (VarType): The output type. Supported type: VarType.LOD_TENSOR, 
           VarType.SELECTED_ROWS, VarType.NCCL_ID. Default: VarType.LOD_TENSOR.

    Returns:
        Variable: The global variable that gives access to the data.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          # Creates a variable with fixed size [1, 2, 3]
          # User can only feed data of the same shape to x
          x = fluid.data(name='x', shape=[1, 2, 3], dtype='int64')

          # Creates a variable with changable batch size -1.
          # Users can feed data of any batch size into y, 
          # but size of each data sample has to be [3, 224, 224]
          y = fluid.data(name='y', shape=[-1, 3, 224, 224], dtype='float32')

    """
    helper = LayerHelper('data', **locals())
    return helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=type,
        stop_gradient=True,
        lod_level=0,
        is_data=True,
        need_check_feed=True)
