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
from ..fluid.data_feeder import check_variable_and_dtype, check_type

# TODO: define functions to get tensor attributes  
from ..fluid.layers import rank  #DEFINE_ALIAS
from ..fluid.layers import shape  #DEFINE_ALIAS

__all__ = ['rank', 'shape', 'real', 'imag', 'size']


def size(input, axes=None):
    """
    Get the shape of the input.

    .. code-block:: text

        Given N-D Tensor:
            input = [ [1, 2, 3, 4], [5, 6, 7, 8] ]

        Then:
            input.shape = [2, 4]

    Args:
        input (Variable): The input can be N-D Tensor or SelectedRows with data type bool, float16, float32, float64, int32, int64.
                          If input variable is type of SelectedRows, returns the shape of it's inner tensor.

    Returns:
        Variable (Tensor): The shape of the input variable.

    Examples:
        .. code-block:: python

            import paddle
             ndim_3_tensor = paddle.to_tensor([[[1, 2, 3, 4, 5],
                                                [6, 7, 8, 9, 10]],
                                               [[11, 12, 13, 14, 15],
                                                [16, 17, 18, 19, 20]]])
             print(ndim_3_tensor.size()) # Tensor(shape=[3], dtype=int32, place=CPUPlace, stop_gradient=True, [2, 2, 5])
             print(ndim_3_tensor.size(2)) # Tensor(shape=[1], dtype=int32, place=CPUPlace, stop_gradient=True, [5])
             print(ndim_3_tensor.size([-1, 0, 1])) # Tensor(shape=[3], dtype=int32, place=CPUPlace, stop_gradient=True, [5, 2, 2])
             print(paddle.shape(ndim_3_tensor)) # Tensor(shape=[3], dtype=int32, place=CPUPlace, stop_gradient=True, [2, 2, 5])
    """
    check_variable_and_dtype(
        input, 'input',
        ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'], 'shape')
    check_type(axes, 'axes', (int, list, tuple, type(None)), 'shape')
    if isinstance(axes, int):
        axes = [axes]
    attrs = {'axes': axes if axes is not None and axes != [] else []}
    helper = LayerHelper('shape', **locals())
    out = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type='shape',
        inputs={'Input': input},
        attrs=attrs,
        outputs={'Out': out})

    return out


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
