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
# __all__ = ['mean', 'reduce_mean', 'std', 'var']
__all__ = ['reduce_mean', 'std']
from ..fluid.layers import elementwise_mul, elementwise_sub, elementwise_div, abs, sqrt, size
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import Variable, OpProtoHolder, in_dygraph_mode, dygraph_only, _dygraph_tracer, default_main_program


def reduce_mean(input, dim=None, keep_dim=False, name=None):
    """
    Computes the mean of the input tensor's elements along the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimension along which the mean is computed. If
            `None`, compute the mean over all elements of :attr:`input`
            and return a variable with a single element, otherwise it
            must be in the range :math:`[-rank(input), rank(input))`. If
            :math:`dim[i] < 0`, the dimension to reduce is
            :math:`rank(input) + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default 
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    
    Returns:
        Variable: Tensor, results of average on the specified dim of input tensor,
        it's data type is the same as input's Tensor.
    
    Raises:
        TypeError, if out data type is different with the input data type.
    
    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            fluid.layers.reduce_mean(x)  # [0.4375]
            fluid.layers.reduce_mean(x, dim=0)  # [0.15, 0.25, 0.55, 0.8]
            fluid.layers.reduce_mean(x, dim=-1)  # [0.475, 0.4]
            fluid.layers.reduce_mean(x, dim=1, keep_dim=True)  # [[0.475], [0.4]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_mean(y, dim=[1, 2]) # [2.5, 6.5]
            fluid.layers.reduce_mean(y, dim=[0, 1]) # [4.0, 5.0]
    """

    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    attrs = {
        'dim': dim if dim != None and dim != [] else [0],
        'keep_dim': keep_dim,
        'reduce_all': True if dim == None or dim == [] else False
    }

    if in_dygraph_mode():
        inputs = {'X': [input]}
        outs = core.ops.reduce_mean(inputs, attrs)
        return outs['Out'][0]

    check_variable_and_dtype(
        input, 'input', ['float32', 'float64', 'int32', 'int64'], 'reduce_mean')
    helper = LayerHelper('reduce_mean', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='reduce_mean',
        inputs={'X': input},
        outputs={'Out': out},
        attrs=attrs)
    return out


def std(input, axis, keepdim=False, unbiased=True, out=None, name=None):
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

            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            fluid.tensor.std(x)  # 
            fluid.tensor.std(x, dim=0)  # 
            fluid.tensor.std(x, dim=-1)  # 
    """
    check_variable_and_dtype(input, 'input',
                             ['float32', 'float64', 'int32', 'int64'], 'std')
    sqrd = elementwise_mul(input, input)
    sqrd_mean = reduce_mean(sqrd, dim=axis, keep_dim=keepdim)
    mean = reduce_mean(input, dim=axis, keep_dim=keepdim)
    mean_sqrd = elementwise_mul(mean, mean)
    var = abs(elementwise_sub(sqrd_mean, mean_sqrd))
    if not unbiased:
        return sqrt(var)
    size_input = size(input)
    size_out = size(var)
    count = size_input / size_out
    var = var * count / (count - 1)
    return sqrt(var)
