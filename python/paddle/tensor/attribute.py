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

from __future__ import print_function

from ..fluid.framework import core, in_dygraph_mode, Variable
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype

# TODO: define functions to get tensor attributes  
from ..fluid.layers import rank  #DEFINE_ALIAS
from ..fluid.layers import shape  #DEFINE_ALIAS

__all__ = ['rank', 'shape', 'real', 'imag']


def _complex_to_real_dtype(dtype):
    if dtype == core.VarDesc.VarType.COMPLEX64:
        return core.VarDesc.VarType.FP32
    elif dtype == core.VarDesc.VarType.COMPLEX128:
        return core.VarDesc.VarType.FP64
    else:
        return dtype


def real(x, name=None):
    """
    Returns a new tensor containing real values of the input tensor.

    Args:
        x (Tensor): the input tensor, its data type could be complex64 or complex128.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name` .
      
    Returns:
        Tensor: a tensor containing real values of the input tensor.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor(
                [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])
            # Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
            #        [[(1+6j), (2+5j), (3+4j)],
            #         [(4+3j), (5+2j), (6+1j)]])

            real_res = paddle.real(x)
            # Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[1., 2., 3.],
            #         [4., 5., 6.]])

            real_t = x.real()
            # Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[1., 2., 3.],
            #         [4., 5., 6.]])
    """
    if in_dygraph_mode():
        return core.ops.real(x)

    check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], 'real')
    helper = LayerHelper('real', **locals())
    out = helper.create_variable_for_type_inference(
        dtype=_complex_to_real_dtype(helper.input_dtype()))
    helper.append_op(type='real', inputs={'X': x}, outputs={'Out': out})
    return out


def imag(x, name=None):
    """
    Returns a new tensor containing imaginary values of input tensor.

    Args:
        x (Tensor): the input tensor, its data type could be complex64 or complex128.
        name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: a tensor containing imaginary values of the input tensor.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor(
                [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])
            # Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
            #        [[(1+6j), (2+5j), (3+4j)],
            #         [(4+3j), (5+2j), (6+1j)]])

            imag_res = paddle.imag(x)
            # Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[6., 5., 4.],
            #         [3., 2., 1.]])

            imag_t = x.imag()
            # Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[6., 5., 4.],
            #         [3., 2., 1.]])
    """
    if in_dygraph_mode():
        return core.ops.imag(x)

    check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], 'imag')
    helper = LayerHelper('imag', **locals())
    out = helper.create_variable_for_type_inference(
        dtype=_complex_to_real_dtype(helper.input_dtype()))
    helper.append_op(type='imag', inputs={'X': x}, outputs={'Out': out})
    return out
