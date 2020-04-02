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
__all__ = [#'abs',
           # 'acos',
           # 'asin',
           # 'atan',
           # 'ceil',
           # 'cos',
           # 'cumsum',
           # 'elementwise_add',
           # 'elementwise_div',
           # 'elementwise_floordiv',
           # 'elementwise_max',
           # 'elementwise_min',
           # 'elementwise_mod',
           # 'elementwise_mul',
           # 'elementwise_pow',
           # 'elementwise_sub',
           # 'exp',
           # 'floor',
           # 'increment',
           # 'log',
           # 'mul',
           # 'multiplex',
           # 'pow',
           # 'reciprocal',
           # 'reduce_max',
           # 'reduce_min',
           # 'reduce_prod',
           # 'reduce_sum',
           # 'round',
           # 'rsqrt',
           # 'scale',
           # 'sign',
           # 'sin',
           # 'sqrt',
           # 'square',
           # 'stanh',
            'sum',
           # 'sums',
           # 'tanh',
           # 'elementwise_sum',
           # 'max',
           # 'min',
           # 'mm',
           # 'div',
           # 'add',
           # 'atan',
           # 'logsumexp',
           # 'inverse',
           # 'log1p',
           # 'erf',
           # 'addcmul',
           # 'addmm'
           ]
from paddle.common_ops_import import *


def sum(input, dim=None, dtype=None, keep_dim=False, name=None):
    """
    Computes the sum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`input` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
            the dimension to reduce is :math:`rank + dim[i]`.
        dtype(str, optional): The dtype of output tensor. The default value is None, the dtype 
            of output is the same as input tensor.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, results of summation operation on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Raises:
        ValueError, the :attr:`dtype` must be float64 or int64.
    
    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            out1 = paddle.sum(x)  # [3.5]
            out2 = paddle.sum(x, dim=0)  # [0.3, 0.5, 1.1, 1.6]
            out3 = paddle.sum(x, dim=-1)  # [1.9, 1.6]
            out4 = paddle.sum(x, dim=1, keep_dim=True)  # [[1.9], [1.6]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1, 2], [3, 4]],
            #      [[5, 6], [7, 8]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            out5 = paddle.sum(y, dim=[1, 2]) # [10, 26]
            out6 = paddle.sum(y, dim=[0, 1]) # [16, 20]

    """
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    attrs = {
        'dim': dim if dim != None and dim != [] else [0],
        'keep_dim': keep_dim,
        'reduce_all': True if dim == None or dim == [] else False,
    }
    dtype_flag = False
    if dtype is not None:
        if dtype in ['float64', 'int64']:
            if (convert_dtype(input.dtype) == "float32" and dtype == "float64") or \
               (convert_dtype(input.dtype) == "int32" and dtype == "int64"):
                attrs.update({
                    'in_dtype': input.dtype,
                    'out_dtype': convert_np_dtype_to_dtype_(dtype)
                })
                dtype_flag = True
        else:
            raise ValueError(
                "The value of 'dtype' in sum op must be float64, int64, but received of {}".
                format(dtype))

    if in_dygraph_mode():
        inputs = {'X': [input]}
        outs = core.ops.reduce_sum(inputs, attrs)
        return outs['Out'][0]

    check_variable_and_dtype(
        input, 'input', ['float32', 'float64', 'int32', 'int64'], 'reduce_sum')
    helper = LayerHelper('sum', **locals())
    if dtype_flag:
        out = helper.create_variable_for_type_inference(
            dtype=convert_np_dtype_to_dtype_(dtype))
    else:
        out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='reduce_sum',
        inputs={'X': input},
        outputs={'Out': out},
        attrs=attrs)
    return out
