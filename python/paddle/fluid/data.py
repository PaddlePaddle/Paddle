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

import core
from .framework import convert_np_dtype_to_dtype_
from .layer_helper import LayerHelper

__all__ = ['data', 'check_feed_shape_type']


def data(name, shape, dtype='float32', type=core.VarDesc.VarType.LOD_TENSOR):
    """
    This function takes in the input and based on whether data has
    to be returned back as a minibatch, it creates the global variable by using
    the helper functions. The global variables can be accessed by all the
    following operators in the graph.

    All the input variables of this function are passed in as local variables
    to the LayerHelper constructor.

    Args:
       name(str): The name/alias of the function
       shape(list): Tuple declaring the shape. If :code:`append_batch_size` is 
                    True and there is no -1 inside :code:`shape`, it should be 
                    considered as the shape of the each sample. Otherwise, it
                    should be considered as the shape of the batched data.  
       dtype(np.dtype|VarType|str): The type of data : float32, float16, int etc
       type(VarType): The output type. By default it is LOD_TENSOR.

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


def check_feed_shape_type(var, feed):
    """Returns True if the variable doesn't require feed check or it is
       compatible with the shape and dtype of thefeed value.
    
       Args:
           var(Variable): the Variable object
           feed(list|np.array): the feed value
    """
    print(var.name + " need check feed: " + str(var.need_check_feed))
    if var.need_check_feed:
        numpy_feed = as_numpy(feed) if isinstance(
            feed, core.LodTensorArray) else numpy.array(
                feed, copy=False)
        if not dimension_is_compatible_with(numpy_feed.shape, var.shape()):
            raise ValueError('Cannot feed value of shape %r for Variable %r, '
                             'which has shape %r' %
                             (numpy_feed.shape, var.name(), var.shape()))
        if not dtype_is_compatible_with(numpy_feed.dtype, var.dtype()):
            raise ValueError('Cannot feed value of type %r for Variable %r, '
                             'which has type %r' %
                             (numpy_feed.dtype, var.name(), var.dtype()))
    return True


def dtype_is_compatible_with(first, second):
    """Returns True if the first dtype can be compatible the second one.
       Currently, we require the two dtype's have to be same.
      
    Args:
      dtype(np.dtype|VarType|str): The type of data : float32, float16, int etc
    
    Returns:
      Whether the two types are same
    """

    first = convert_np_dtype_to_dtype_(first)
    second = convert_np_dtype_to_dtype_(second)
    return first == second


def dimension_is_compatible_with(first, second):
    """Returns True if the two dimensions are compatible.

       A dimension is compatible with the other if and only if:
       1. The length of the dimensions are same.
       2. Each non-negative number of the two dimentions are same.
       3. For negative number or 'None' in a dimention, it means unknown so it
          is compatible with any number.

    Args:
      first(list/tuple): integers representing shape. "None" or negative
        number means unknown.
      second(list/tuple): integers representing shape. "None" or negative
        number means unknown.

    Returns:
       True if the two dimensions are compatible.
    """

    dim_len = len(first)
    if dim_len != len(second):
        return False

    for i in range(dim_len):
        if first[i] is None or first[i] < 0:
            continue
        if second[i] is None or second[i] < 0:
            continue
        if first[i] != second[i]:
            return False

    return True
