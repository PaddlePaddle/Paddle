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

    This function takes in the input and based on whether the data has
    to be returned back as a minibatch, it creates the global variable by using
    the helper functions. The global variables can be accessed by all the
    following operators in the graph.

    All the input variables of this function are passed in as local variables
    to the LayerHelper constructor.

    Note: Unlike `paddle.fluid.layers.data` which set shape at compile time but
       not check the shape of feeded data, this `paddle.fluid.data` checks the
       shape of data feeded by Executor/ParallelExecutor during run time.

    Args:
       name (None|str): The name/alias of the variable
       shape (list|tuple): List|Tuple of integers declaring the shape.
       dtype (np.dtype|VarType|str): The type of the data: float32, int64, etc.
       type (VarType): The output type. Default: LOD_TENSOR.

    Returns:
        Variable: The global variable that gives access to the data.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.data(name='x', shape=[784], dtype='float32')

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
