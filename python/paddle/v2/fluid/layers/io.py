#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from .. import core
from ..layer_helper import LayerHelper

__all__ = ['data']


def data(name,
         shape,
         append_batch_size=True,
         dtype='float32',
         lod_level=0,
         type=core.VarDesc.VarType.LOD_TENSOR,
         stop_gradient=True):
    """
    **Data Layer**

    This function takes in the input and based on whether data has
    to be returned back as a minibatch, it creates the global variable by using
    the helper functions. The global variables can be accessed by all the
    following operators in the graph.

    All the input variables of this function are passed in as local variables
    to the LayerHelper constructor.

    Args:
       name(str): The name/alias of the function
       shape(list): Tuple declaring the shape.
       append_batch_size(bool): Whether or not to append the data as a batch.
       dtype(int|float): The type of data : float32, float_16, int etc
       type(VarType): The output type. By default it is LOD_TENSOR.
       lod_level(int): The LoD Level. 0 means the input data is not a sequence.
       main_program(Program): Name of the main program that calls this
       startup_program(Program): Name of the startup program
       stop_gradient(bool): A boolean that mentions whether gradient should flow.

    Returns:
        Variable: The global variable that gives access to the data.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(name='x', shape=[784], dtype='float32')
    """
    helper = LayerHelper('data', **locals())
    shape = list(shape)
    for i in xrange(len(shape)):
        if shape[i] is None:
            shape[i] = -1
            append_batch_size = False
        elif shape[i] < 0:
            append_batch_size = False

    if append_batch_size:
        shape = [-1] + shape  # append batch size as -1

    return helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=type,
        stop_gradient=stop_gradient,
        lod_level=lod_level)
