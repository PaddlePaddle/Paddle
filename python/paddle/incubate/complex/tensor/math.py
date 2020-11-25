#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle.common_ops_import import *
from ..helper import is_complex, is_real, complex_variable_exists
from ....fluid.framework import ComplexVariable
from ....fluid import layers
from ....tensor import math

__all__ = [
    'elementwise_add',
    'elementwise_sub',
    'elementwise_mul',
    'elementwise_div',
    'kron',
    'trace',
    'sum',
]

from ....fluid.layers import elementwise_add  #DEFINE_ALIAS
from ....fluid.layers import elementwise_sub  #DEFINE_ALIAS
from ....fluid.layers import elementwise_mul  #DEFINE_ALIAS
from ....fluid.layers import elementwise_div  #DEFINE_ALIAS
from ....tensor.math import trace  #DEFINE_ALIAS
from ....tensor.math import kron  #DEFINE_ALIAS


def sum(input, dim=None, keep_dim=False, name=None):
    """
    The layer to compute the sum for a complex number tensor elements over the given dimension. input :attr:`input` must be a ComplexVariable. 
    See the detailed description for the function and other arguments 
    in :ref:`api_tensor_math_sum` . 

    Args:
        input(ComplexVariable): The input ComplexVariable with any number of dimensions. 
            The supported data types include complex64 and complex128.
        dim (list|int, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`input` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
            the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        ComplexVariable: Results of summation operation on the specified dim of input tensor,
        it's data type is the same as input.

    Raises:
        ValueError: the :attr:`dtype` must be float64 or int64.
    
    Examples:
        .. code-block:: python

            import paddle.complex as cpx
            import paddle.fluid.dygraph as dg
            import numpy as np

            with dg.guard():
                # x is a Tensor variable with following elements:
                #    [[0.2, 0.3, 0.5, 0.9], 
                #     [0.1, 0.2, 0.6, 0.7]]
                # Each example is followed by the corresponding output tensor.
                x = np.array([[0.2, 0.3, 0.5, 0.9],[0.1, 0.2, 0.6, 0.7]]) + 1j * np.array([[0.3, 0.4, 0.5, 0.2],[0.3, 0.6, 0.8, 0.3]])
                x = dg.to_variable(x)
                out1 = cpx.sum(x)  # [3.5+3.4j]
                out2 = cpx.sum(x, dim=0)  # [0.3+0.6j, 0.5+1.j, 1.1+1.3j, 1.6+0.5j]
                out3 = cpx.sum(x, dim=-1)  # [1.9+1.4j, 1.6+2.j]
                out4 = cpx.sum(x, dim=1, keep_dim=True)  # [[1.9+1.4j], [1.6+2.j]]

                # y is a Tensor variable with shape [2, 2, 2] and elements as below:
                #      [[[1, 2], [3, 4]],
                #      [[5, 6], [7, 8]]]
                # Each example is followed by the corresponding output tensor.
                y = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) + 1j * np.array([[[4, 3], [2, 1]], [[8, 7], [6, 5]]])
                y = dg.to_variable(y)
                out5 = cpx.sum(y, dim=[1, 2]) # [10.+10.j, 26.+26.j]
                out6 = cpx.sum(y, dim=[0, 1]) # [16.+20.j, 20.+16.j]

    """
    complex_variable_exists([input], "sum")
    real = math.sum(input.real, axis=dim, keepdim=keep_dim, name=name)
    imag = math.sum(input.imag, axis=dim, keepdim=keep_dim, name=name)
    return ComplexVariable(real, imag)
