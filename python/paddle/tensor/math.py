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

# TODO: define math functions  
# __all__ = ['abs',
#            'acos',
#            'asin',
#            'atan',
#            'ceil',
#            'cos',
#            'cumsum',
#            'elementwise_add',
#            'elementwise_div',
#            'elementwise_floordiv',
#            'elementwise_max',
#            'elementwise_min',
#            'elementwise_mod',
#            'elementwise_mul',
#            'elementwise_pow',
#            'elementwise_sub',
#            'exp',
#            'floor',
#            'increment',
#            'log',
#            'mul',
#            'multiplex',
#            'pow',
#            'reciprocal',
#            'reduce_max',
#            'reduce_min',
#            'reduce_prod',
#            'reduce_sum',
#            'round',
#            'rsqrt',
#            'scale',
#            'sign',
#            'sin',
#            'sqrt',
#            'square',
#            'stanh',
#            'sum',
#            'sums',
#            'tanh',
#            'elementwise_sum',
#            'max',
#            'min',
#            'mm',
#            'div',
#            'add',
#            'atan',
#            'logsumexp',
#            'inverse',
#            'log1p',
#            'erf',
#            'addcmul',
#            'addmm']
from paddle.common_ops_import import *


def max(input, dim=None, keep_dim=False, out=None, name=None):
    """
    Computes the maximum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimension along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, results of maximum on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            paddle.tensor.max(x)  # [0.9]
            paddle.tensor.max(x, dim=0)  # [0.2, 0.3, 0.6, 0.9]
            paddle.tensor.max(x, dim=-1)  # [0.9, 0.7]
            paddle.tensor.max(x, dim=1, keep_dim=True)  # [[0.9], [0.7]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            paddle.tensor.max(y, dim=[1, 2]) # [4.0, 8.0]
            paddle.tensor.max(y, dim=[0, 1]) # [7.0, 8.0]
    """
    helper = LayerHelper('max', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_max',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None and dim != [] else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None or dim == [] else False
        })
    return out


def min(input, dim=None, keep_dim=False, out=None, name=None):
    """
    Computes the minimum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimensions along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, result of minimum on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            paddle.tensor.min(x)  # [0.1]
            paddle.tensor.min(x, dim=0)  # [0.1, 0.2, 0.5, 0.7]
            paddle.tensor.min(x, dim=-1)  # [0.2, 0.1]
            paddle.tensor.min(x, dim=1, keep_dim=True)  # [[0.2], [0.1]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            paddle.tensor.min(y, dim=[1, 2]) # [1.0, 5.0]
            paddle.tensor.min(y, dim=[0, 1]) # [1.0, 2.0]
    """
    helper = LayerHelper('min', **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_min',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None and dim != [] else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None or dim == [] else False
        })
    return out


def log1p(x, out=None, name=None):
    """
    Calculates the natural log of the given input tensor, element-wise.

    .. math::

        Out = \\ln(x+1)

    Args:
        x (Variable): Input LoDTensor or Tensor. Must be one of the following types: float32, float64.
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`


    Returns:
        Variable: The natural log of the input LoDTensor or Tensor computed element-wise.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            # Graph Organizing
            x = fluid.layers.data(name="x", shape=[1], dtype="float32")
            res = fluid.layers.log1p(x)

            # Create an executor using CPU as an example
            exe = fluid.Executor(fluid.CPUPlace())

            # Execute
            x_i = np.array([[0], [1]]).astype(np.float32)
            res_val, = exe.run(fluid.default_main_program(), feed={'x':x_i}, fetch_list=[res])
            print(res_val) # [[0.], [0.6931472]]
    """

    inputs = {'X': [x]}
    if in_dygraph_mode():
        outs = core.ops.log1p(inputs)
        return outs['Out'][0]

    helper = LayerHelper('log1p', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    if out is None:
        out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type="log1p", inputs={"X": x}, outputs={"Out": out})
    return out
