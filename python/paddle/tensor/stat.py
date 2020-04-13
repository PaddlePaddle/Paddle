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

# TODO: define statistical functions of a tensor  
__all__ = [
    #       'mean', 
    #       'reduce_mean', 
    'std',
    #       'var'
]
from ..fluid import layers
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import Variable, OpProtoHolder, in_dygraph_mode, dygraph_only, _dygraph_tracer, default_main_program
from ..fluid import core


def std(input, axis=None, keepdim=False, unbiased=True, name=None):
    """
    Computes the standard-deviation of the input tensor's elements along the given axis.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        axis (list|int, optional): The dimension along which the mean is computed. If
            `None`, compute the mean over all elements of :attr:`input`
            and return a variable with a single element, otherwise it
            must be in the range :math:`[-rank(input), rank(input))`. If
            :math:`dim[i] < 0`, the dimension to reduce is
            :math:`rank(input) + dim[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default 
            value is False.
        unbiased(bool, optional): Whether to use the unbiased estimation. If true, standard-deviation 
            will be calculated via the Bessel's correction. Otherwise, biased estimator will be used.
            Default value is True.
        name(str,  optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    
    Returns:
        Variable: Tensor, results of average on the specified dim of input tensor,
        it's data type is the same as input's Tensor.
    
    Raises:
        TypeError, if out data type is different with the input data type.
    
    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            paddle.std(x)  # [0.28252685] 
            paddle.std(x, axis=0)  # [0.0707107, 0.07071075, 0.07071064, 0.1414217]
            paddle.std(x, axis=-1)  # [0.30956957, 0.29439208] 
    """
    check_variable_and_dtype(input, 'input',
                             ['float32', 'float64', 'int32', 'int64'], 'std')
    sqrd = layers.elementwise_mul(input, input)
    sqrd_mean = layers.reduce_mean(sqrd, dim=axis, keep_dim=keepdim)
    mean = layers.reduce_mean(input, dim=axis, keep_dim=keepdim)
    mean_sqrd = layers.elementwise_mul(mean, mean)
    var = layers.abs(layers.elementwise_sub(sqrd_mean, mean_sqrd))
    if not unbiased:
        return layers.sqrt(var)
    size_input = layers.size(input)
    size_out = layers.size(var)
    count = size_input / size_out
    count = layers.cast(count, input.dtype)
    var = layers.elementwise_div(layers.elementwise_mul(var, count), count - 1)
    return layers.sqrt(var)
