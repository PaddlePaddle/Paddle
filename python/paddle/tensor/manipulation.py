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

from ..fluid.layers import core
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import Variable, OpProtoHolder, in_dygraph_mode, convert_np_dtype_to_dtype_
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ..fluid.layers.tensor import fill_constant
from ..fluid.layers import utils
import numpy as np
# TODO: define functions to manipulate a tensor  
from ..fluid.layers import cast  #DEFINE_ALIAS
from ..fluid.layers import expand_as  #DEFINE_ALIAS
from ..fluid.layers import slice  #DEFINE_ALIAS
from ..fluid.layers import strided_slice  #DEFINE_ALIAS
from ..fluid.layers import transpose  #DEFINE_ALIAS
from ..fluid.layers import unique  #DEFINE_ALIAS
from ..fluid.layers import unstack  #DEFINE_ALIAS

from ..fluid.layers import gather_nd  #DEFINE_ALIAS
from ..fluid.layers import scatter_nd_add  #DEFINE_ALIAS
from ..fluid.layers import scatter_nd  #DEFINE_ALIAS
from ..fluid.layers import shard_index  #DEFINE_ALIAS
from ..fluid.layers import unique_with_counts  #DEFINE_ALIAS
from ..fluid import layers
import paddle

__all__ = [
    'cast',
    'concat',
    'expand',
    'broadcast_to',
    'expand_as',
    'flatten',
    'gather',
    'gather_nd',
    'reshape',
    'reverse',
    'scatter',
    'scatter_nd_add',
    'scatter_nd',
    'shard_index',
    'slice',
    'split',
    'chunk'
    'squeeze',
    'stack',
    'strided_slice',
    'transpose',
    'unique',
    'unique_with_counts',
    'unsqueeze',
    'unstack',
    'flip',
    'unbind',
    'roll',
    'tile',
]


def concat(x, axis=0, name=None):
    """
	:alias_main: paddle.concat
	:alias: paddle.tensor.concat, paddle.tensor.manipulation.concat

    This OP concatenates the input along the axis.

    Args:
        x(list|tuple): ``x`` is a Tensor list or Tensor tuple which is with data type bool, float16, 
            float32, float64, int32, int64. All the Tensors in ``x`` must have same data type.
        axis(int|Tensor, optional): Specify the axis to operate on the input Tensors.
            It's a scalar with data type int or a Tensor with shape [1] and data type int32 
            or int64. The effective range is [-R, R), where R is Rank(x). When ``axis < 0``,
            it works the same way as ``axis+R``. Default is 0.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Raises:
        TypeError: ``x`` must be list or tuple.
        TypeError: The data type of ``x`` must be one of bool, float16, float32, float64, int32 and int64. 
        TypeError: The ``axis`` must be int or Tensor. The dtype of ``axis`` must be int32 or int64 when it's a Tensor.
        TypeError: All the Tensors in ``x`` must have the same data type.

    Returns:
        Tensor: A Tensor with the same data type as ``x``.

    Examples:
        .. code-block:: python
            
            import paddle
            import numpy as np
            
            paddle.disable_static()  # Now we are in imperative mode
            in1 = np.array([[1, 2, 3],
                            [4, 5, 6]])
            in2 = np.array([[11, 12, 13],
                            [14, 15, 16]])
            in3 = np.array([[21, 22],
                            [23, 24]])
            x1 = paddle.to_tensor(in1)
            x2 = paddle.to_tensor(in2)
            x3 = paddle.to_tensor(in3)
            zero = paddle.full(shape=[1], dtype='int32', fill_value=0)
            # When the axis is negative, the real axis is (axis + Rank(x))
            # As follow, axis is -1, Rank(x) is 2, the real axis is 1
            out1 = paddle.concat(x=[x1, x2, x3], axis=-1)
            out2 = paddle.concat(x=[x1, x2], axis=0)
            out3 = paddle.concat(x=[x1, x2], axis=zero)
            # out1
            # [[ 1  2  3 11 12 13 21 22]
            #  [ 4  5  6 14 15 16 23 24]]
            # out2 out3
            # [[ 1  2  3]
            #  [ 4  5  6]
            #  [11 12 13]
            #  [14 15 16]]
    """
    check_type(x, 'x', (list, tuple), 'concat')
    return paddle.fluid.layers.concat(input=x, axis=axis, name=name)


def flip(x, axis, name=None):
    """
	:alias_main: paddle.flip
	:alias: paddle.flip,paddle.tensor.flip,paddle.tensor.manipulation.flip


    Reverse the order of a n-D tensor along given axis in axis.

    Args:
        x (Variable): A Tensor(or LoDTensor) with shape :math:`[N_1, N_2,..., N_k]` . The data type of the input Tensor x
            should be float32, float64, int32, int64, bool.
        axis (list): The axis(axes) to flip on. Negative indices for indexing from the end are accepted.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: Tensor or LoDTensor calculated by flip layer. The data type is same with input x.

    Examples:
        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()

          image_shape=(3, 2, 2)
          x = np.arange(image_shape[0] * image_shape[1] * image_shape[2]).reshape(image_shape)
          x = x.astype('float32')
          img = paddle.to_variable(x)
          out = paddle.flip(img, [0,1])

          print(out) # [[[10,11][8, 9]],[[6, 7],[4, 5]] [[2, 3],[0, 1]]]
    """
    helper = LayerHelper("flip", **locals())
    check_type(x, 'X', (Variable), 'flip')
    dtype = helper.input_dtype('x')
    check_dtype(dtype, 'X',
                ['float16', 'float32', 'float64', 'int32', 'int64', 'bool'],
                'flip')
    check_type(axis, 'axis', (list, tuple), 'flip')
    if name is None:
        out = helper.create_variable_for_type_inference(dtype)
    else:
        out = helper.create_variable(name=name, dtype=dtype, persistable=False)

    helper.append_op(
        type="flip",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={"axis": axis})
    return out


reverse = flip  #DEFINE_ALIAS


def flatten(x, start_axis=0, stop_axis=-1, name=None):
    """
    **Flatten op**

    Flattens a contiguous range of axes in a tensor according to start_axis and stop_axis.

    For Example:

    .. code-block:: text

        Case 1:

          Given
            X.shape = (3, 100, 100, 4)

          and
            start_axis = 1
            end_axis = 2

          We get:
            Out.shape = (3, 1000 * 100, 2)

        Case 2:

          Given
            X.shape = (3, 100, 100, 4)

          and
            start_axis = 0
            stop_axis = -1

          We get:
            Out.shape = (3 * 100 * 100 * 4)

    Args:
        x (Variable): A tensor of number of dimentions >= axis. A tensor with data type float32,
                      float64, int8, int32, int64.
        start_axis (int): the start axis to flatten
        stop_axis (int): the stop axis to flatten
        name(str, Optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.

    Returns:
        Variable: A tensor with the contents of the input tensor, with input \
                  axes flattened by indicated start axis and end axis. \
                  A Tensor with data type same as input x.

    Raises:
        ValueError: If x is not a Variable.
        ValueError: If start_axis or stop_axis is illegal.

    Examples:

        .. code-block:: python

            import paddle
            import numpy as np

            paddle.disable_static()

            image_shape=(2, 3, 4, 4)
            x = np.arange(image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3]).reshape(image_shape) / 100.
            x = x.astype('float32')
            
            img = paddle.to_variable(x)
            out = paddle.flatten(img, start_axis=1, stop_axis=2)
            # out shape is [2, 12, 4]
    """
    if not (isinstance(x, Variable)):
        raise ValueError("The input x should be a Variable")

    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int8', 'int32', 'int64'], 'flatten')
    helper = LayerHelper('flatten', **locals())

    x_dim = len(x.shape)
    if not (isinstance(start_axis, int)) or (
            start_axis > x_dim - 1) or start_axis < -x_dim:
        raise ValueError(
            "The start_axis should be a int, and in range [-rank(x), rank(x))")
    if not (isinstance(stop_axis, int)) or (
            stop_axis > x_dim - 1) or stop_axis < -x_dim:
        raise ValueError(
            "The stop_axis should be a int, and in range [-rank(x), rank(x))")
    if start_axis < 0:
        start_axis = start_axis + x_dim
    if stop_axis < 0:
        stop_axis = stop_axis + x_dim
    if start_axis > stop_axis:
        raise ValueError("The stop_axis should be larger than stat_axis")

    if in_dygraph_mode():
        dy_out, _ = core.ops.flatten_contiguous_range(
            x, 'start_axis', start_axis, 'stop_axis', stop_axis)
        return dy_out

    out = helper.create_variable_for_type_inference(x.dtype)
    x_shape = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='flatten_contiguous_range',
        inputs={"X": x},
        outputs={'Out': out,
                 'XShape': x_shape},
        attrs={"start_axis": start_axis,
               "stop_axis": stop_axis})
    return out


def roll(x, shifts, axis=None, name=None):
    """
	:alias_main: paddle.roll
	:alias: paddle.roll,paddle.tensor.roll,paddle.tensor.manipulation.roll

    Roll the `x` tensor along the given axis(axes). With specific 'shifts', Elements that 
    roll beyond the last position are re-introduced at the first according to 'shifts'. 
    If a axis is not specified, 
    the tensor will be flattened before rolling and then restored to the original shape.

    Args:
        x (Variable): The x tensor variable as input.
        shifts (int|list|tuple): The number of places by which the elements
                           of the `x` tensor are shifted.
        axis (int|list|tuple|None): axis(axes) along which to roll.

    Returns:
        Variable: A Tensor with same data type as `x`.

    Examples:
        .. code-block:: python
            import numpy as np
            import paddle
            import paddle.fluid as fluid

            data = np.array([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]])
            paddle.disable_static()
            x = paddle.to_variable(data)
            out_z1 = paddle.roll(x, shifts=1)
            print(out_z1.numpy())
            #[[9. 1. 2.]
            # [3. 4. 5.]
            # [6. 7. 8.]]
            out_z2 = paddle.roll(x, shifts=1, axis=0)
            print(out_z2.numpy())
            #[[7. 8. 9.]
            # [1. 2. 3.]
            # [4. 5. 6.]]
    """
    helper = LayerHelper("roll", **locals())
    origin_shape = x.shape
    if type(shifts) == int:
        shifts = [shifts]
    if type(axis) == int:
        axis = [axis]

    len_origin_shape = len(origin_shape)
    if axis:
        for i in range(len(axis)):
            if axis[i] >= len_origin_shape or axis[i] < -len_origin_shape:
                raise ValueError(
                    "axis is out of range, it should be in range [{}, {}), but received {}".
                    format(-len_origin_shape, len_origin_shape, axis))

    if axis:
        check_type(axis, 'axis', (list, tuple), 'roll')
    check_type(shifts, 'shifts', (list, tuple), 'roll')

    if in_dygraph_mode():
        if axis is None:
            x = core.ops.reshape(x, 'shape', [-1, 1])
            axis = [0]
        out = core.ops.roll(x, 'axis', axis, 'shifts', shifts)
        return core.ops.reshape(out, 'shape', origin_shape)

    out = helper.create_variable_for_type_inference(x.dtype)

    if axis is None:
        x = reshape(x, shape=[-1, 1])
        axis = [0]

    helper.append_op(
        type='roll',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'axis': axis,
               'shifts': shifts})
    out = layers.reshape(out, shape=origin_shape, inplace=True)
    return out


def stack(x, axis=0, name=None):
    """
	:alias_main: paddle.stack
	:alias: paddle.stack, paddle.tensor.stack, paddle.tensor.manipulation.stack

    This OP stacks all the input tensors ``x`` along ``axis`` dimemsion. 
    All tensors must be of the same shape and same dtype.
    
    For example, given N tensors of shape [A, B], if ``axis == 0``, the shape of stacked 
    tensor is [N, A, B]; if ``axis == 1``, the shape of stacked 
    tensor is [A, N, B], etc.
    

    .. code-block:: text

        Case 1:

          Input:
            x[0].shape = [1, 2]
            x[0].data = [ [1.0 , 2.0 ] ]
            x[1].shape = [1, 2]
            x[1].data = [ [3.0 , 4.0 ] ]
            x[2].shape = [1, 2]
            x[2].data = [ [5.0 , 6.0 ] ]

          Attrs:
            axis = 0

          Output:
            Out.dims = [3, 1, 2]
            Out.data =[ [ [1.0, 2.0] ],
                        [ [3.0, 4.0] ],
                        [ [5.0, 6.0] ] ]


        Case 2:

          Input:
            x[0].shape = [1, 2]
            x[0].data = [ [1.0 , 2.0 ] ]
            x[1].shape = [1, 2]
            x[1].data = [ [3.0 , 4.0 ] ]
            x[2].shape = [1, 2]
            x[2].data = [ [5.0 , 6.0 ] ]


          Attrs:
            axis = 1 or axis = -2  # If axis = -2, axis = axis+ndim(x[0])+1 = -2+2+1 = 1.

          Output:
            Out.shape = [1, 3, 2]
            Out.data =[ [ [1.0, 2.0]
                          [3.0, 4.0]
                          [5.0, 6.0] ] ]

    Args:
        x (Tensor|list[Tensor]): Input ``x`` can be a single tensor, or a ``list`` of tensors.
                                     If ``x`` is a ``list``, the Tensors in ``x``
                                     must be of the same shape and dtype. Supported data types: float32, float64, int32, int64.
        axis (int, optional): The axis along which all inputs are stacked. ``axis`` range is ``[-(R+1), R+1)``,
                              where ``R`` is the number of dimensions of the first input tensor ``x[0]``. 
                              If ``axis < 0``, ``axis = axis+R+1``. The default value of axis is 0.
        name (str, optional): Please refer to :ref:`api_guide_Name`, Default None.
        
    Returns:
        Tensor: The stacked tensor with same data type as input.

    Example:    
        .. code-block:: python

            import paddle
            import numpy as np

            data1 = np.array([[1.0, 2.0]])
            data2 = np.array([[3.0, 4.0]])
            data3 = np.array([[5.0, 6.0]])

            paddle.disable_static()
            x1 = paddle.to_variable(data1)
            x2 = paddle.to_variable(data2)
            x3 = paddle.to_variable(data3)

            out = paddle.stack([x1, x2, x3], axis=0)
            print(out.shape)  # [3, 1, 2]
            print(out.numpy())
            # [[[1., 2.]],
            #  [[3., 4.]],
            #  [[5., 6.]]]
    """
    return layers.stack(x, axis, name)


def split(x, num_or_sections, axis=0, name=None):
    """
	:alias_main: paddle.split
        :alias: paddle.tensor.split, paddle.tensor.manipulation.split
    
    Split the input tensor into multiple sub-Tensors.
    
    Args:
        x (Tensor): A N-D Tensor. The data type is bool, float16, float32, float64, int32 or int64.
        num_or_sections (int|list|tuple): If ``num_or_sections`` is an int, then ``num_or_sections`` 
            indicates the number of equal sized sub-Tensors that the ``x`` will be divided into.
            If ``num_or_sections`` is a list or tuple, the length of it indicates the number of
            sub-Tensors and the elements in it indicate the sizes of sub-Tensors'  dimension orderly.
            The length of the list must not  be larger than the ``x`` 's size of specified ``axis``.
        axis (int|Tensor, optional): The axis along which to split, it can be a scalar with type 
            ``int`` or a ``Tensor`` with shape [1] and data type  ``int32`` or ``int64``.
            If :math::`axis < 0`, the axis to split along is :math:`rank(x) + axis`. Default is 0.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
    Returns:
        list(Tensor): The list of segmented Tensors.
    Raises:
        TypeError: The data type of ``x`` must be one of bool, float16, float32, float64, int32, int64.
        TypeError: ``num_or_sections`` is not int, list or tuple.
        TypeError: ``axis`` is not int or Tensor. the data type of ``axis`` must be int32 or int64 when it's a Tensor.
    Example:
        .. code-block:: python
            
            import numpy as np
            import paddle
            
            paddle.disable_static()
            # x is a Tensor which shape is [3, 9, 5]
            x_np = np.random.random([3, 9, 5]).astype("int32")
            x = paddle.to_tensor(x_np)

            out0, out1, out22 = paddle.split(x, num_or_sections=3, axis=1)
            # out0.shape [3, 3, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 3, 5]

            out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, 4], axis=1)
            # out0.shape [3, 2, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 4, 5]

            out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, -1], axis=1)
            # out0.shape [3, 2, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 4, 5]
            
            # axis is negative, the real axis is (rank(x) + axis) which real
            # value is 1.
            out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=-2)
            # out0.shape [3, 3, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 3, 5]
    """
    return paddle.fluid.layers.split(
        input=x, num_or_sections=num_or_sections, dim=axis, name=name)


def squeeze(x, axis=None, name=None):
    """
	:alias_main: paddle.squeeze
	:alias: paddle.squeeze, paddle.tensor.squeeze, paddle.tensor.manipulation.squeeze

    This OP will squeeze the dimension(s) of size 1 of input tensor x's shape. 

    If axis is provided, it will remove the dimension(s) by given axis that of size 1. 
    If the dimension of given axis is not of size 1, the dimension remain unchanged. 
    If axis is not provided, all dims equal of size 1 will be removed.

    .. code-block:: text

        Case1:

          Input:
            x.shape = [1, 3, 1, 5]  # If axis is not provided, all dims equal of size 1 will be removed.
            axis = None
          Output:
            out.shape = [3, 5]

        Case2:

          Input:
            x.shape = [1, 3, 1, 5]  # If axis is provided, it will remove the dimension(s) by given axis that of size 1.
            axis = 0
          Output:
            out.shape = [3, 1, 5]
        
        Case4:

          Input:
            x.shape = [1, 3, 1, 5]  # If the dimension of one given axis (3) is not of size 1, the dimension remain unchanged. 
            axis = [0, 2, 3]
          Output:
            out.shape = [3, 5]

        Case4:

          Input:
            x.shape = [1, 3, 1, 5]  # If axis is negative, axis = axis + ndim (number of dimensions in x). 
            axis = [-2]
          Output:
            out.shape = [1, 3, 5]

    Args:
        x (Tensor): The input Tensor. Supported data type: float32, float64, bool, int8, int32, int64.
        axis (int|list|tuple, optional): An integer or list of integers, indicating the dimensions to be squeezed. Default is None.
                          The range of axis is :math:`[-ndim(x), ndim(x))`.
                          If axis is negative, :math:`axis = axis + ndim(x)`.
                          If axis is None, all the dimensions of x of size 1 will be removed.
        name (str, optional): Please refer to :ref:`api_guide_Name`, Default None.

    Returns:
        Tensor: Squeezed Tensor with the same data type as input Tensor.

    Examples:
        .. code-block:: python

            import paddle

            paddle.disable_static()
            
            x = paddle.rand([5, 1, 10])
            output = paddle.squeeze(x, axis=1)
            # output.shape [5, 10]

    """
    if axis is None:
        axis = []
    elif isinstance(axis, int):
        axis = [axis]
    elif isinstance(axis, tuple):
        axis = list(axis)

    return layers.squeeze(x, axis, name)


def unsqueeze(x, axis, name=None):
    """
	:alias_main: paddle.unsqueeze
	:alias: paddle.unsqueeze, paddle.tensor.unsqueeze, paddle.tensor.manipulation.unsqueeze

    Insert single-dimensional entries to the shape of input Tensor ``x``. Takes one
    required argument axis, a dimension or list of dimensions that will be inserted.
    Dimension indices in axis are as seen in the output tensor.

    Args:
        x (Tensor): The input Tensor to be unsqueezed. Supported data type: float32, float64, bool, int8, int32, int64.
        axis (int|list|tuple|Tensor): Indicates the dimensions to be inserted. The data type is ``int32`` . 
                                    If ``axis`` is a list or tuple, the elements of it should be integers or Tensors with shape [1]. 
                                    If ``axis`` is a Tensor, it should be an 1-D Tensor .
                                    If ``axis`` is negative, ``axis = axis + ndim(x) + 1``.
        name (str|None): Name for this layer. Please refer to :ref:`api_guide_Name`, Default None.

    Returns:
        Tensor: Unsqueezed Tensor with the same data type as input Tensor.

    Examples:
        .. code-block:: python

            import paddle

            paddle.disable_static()
            x = paddle.rand([5, 10])
            print(x.shape)  # [5, 10]
            
            out1 = paddle.unsqueeze(x, axis=0)
            print(out1.shape)  # [1, 5, 10]
            
            out2 = paddle.unsqueeze(x, axis=[0, 2]) 
            print(out2.shape)  # [1, 5, 1, 10]

            axis = paddle.fluid.dygraph.to_variable([0, 1, 2])
            out3 = paddle.unsqueeze(x, axis=axis) 
            print(out3.shape)  # [1, 1, 1, 5, 10]
            
    """
    if isinstance(axis, int):
        axis = [axis]

    return layers.unsqueeze(x, axis, name)


def gather(x, index, axis=None, name=None):
    """

    **Gather Layer**

    Output is obtained by gathering entries of the outer-most dimension
    of X indexed by `index` and concatenate them together.

    .. math::

        Out = X[Index]


    .. code-block:: text


                Given:

                x = [[1, 2],
                     [3, 4],
                     [5, 6]]

                index = [1, 2]
                axis=[0]

                Then:

                out = [[3, 4],
                       [5, 6]]
    Args:
        x (Tensor): The source input tensor with rank>=1. Supported data type is
            int32, int64, float32, float64 and uint8 (only for CPU),
            float16 (only for GPU).
        index (Tensor): The index input tensor with rank=1. Data type is int32 or int64.
        axis (Tensor|int, optional): The axis of input to be gathered, it's can be int or a Tensor with data type is int32 or int64. Default: if None, the axis is 0.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        output (Tensor): The output is a tensor with the same rank as ``x``.
    
    Raises:
        TypeError: ``x`` must be a Tensor and the data type of ``x`` must to be one of float16, float32, float64, int32, int64, uint8.
        TypeError: ``index`` must be a Tensor and the data type of ``index`` must be int32 or int64.
        TypeError: ``axis`` must be a Tensor or int and the data type of ``index`` must be int32 or int64 when it's a Tensor.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            paddle.disable_static()
            input_1 = np.array([[1,2],[3,4],[5,6]])
            index_1 = np.array([0,1])
            input = fluid.to_tensor(input_1)
            index = fluid.to_tensor(index_1)
            output = paddle.gather(input, index, axis=0)
            # expected output: [[1,2],[3,4]]
    """
    if axis is None:
        axis = 0
    axis_tensor = axis
    if not isinstance(axis, Variable):
        axis_tensor = fill_constant(shape=[1], dtype='int64', value=axis)
    if in_dygraph_mode():
        return core.ops.gather(x, index, axis_tensor)

    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64', 'uint8'],
        'gather')
    check_variable_and_dtype(index, 'index', ['int32', 'int64'], 'gather')
    if isinstance(axis, Variable):
        check_variable_and_dtype(axis, 'axis', ['int32', 'int64'], 'gather')
    else:
        check_type(axis, 'axis', (int), 'gather')

    helper = LayerHelper('gather', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="gather",
        inputs={"X": x,
                "Index": index,
                "Axis": axis_tensor},
        outputs={"Out": out})
    return out


def unbind(input, axis=0):
    """
	:alias_main: paddle.tensor.unbind
	:alias: paddle.tensor.unbind,paddle.tensor.manipulation.unbind

    Removes a tensor dimension, then split the input tensor into multiple sub-Tensors.
    Args:
        input (Variable): The input variable which is an N-D Tensor, data type being float32, float64, int32 or int64.
       
        axis (int32|int64, optional): A scalar with type ``int32|int64`` shape [1]. The dimension along which to unbind. If :math:`axis < 0`, the
            dimension to unbind along is :math:`rank(input) + axis`. Default is 0.
    Returns:
        list(Variable): The list of segmented Tensor variables.

    Example:
        .. code-block:: python
            import paddle
            # input is a variable which shape is [3, 4, 5]
            input = paddle.fluid.data(
                 name="input", shape=[3, 4, 5], dtype="float32")
            [x0, x1, x2] = paddle.tensor.unbind(input, axis=0)
            # x0.shape [4, 5]
            # x1.shape [4, 5]
            # x2.shape [4, 5]
            [x0, x1, x2, x3] = paddle.tensor.unbind(input, axis=1)
            # x0.shape [3, 5]
            # x1.shape [3, 5]
            # x2.shape [3, 5]
            # x3.shape [3, 5]

    """
    helper = LayerHelper("unbind", **locals())
    check_type(input, 'input', (Variable), 'unbind')
    dtype = helper.input_dtype()
    check_dtype(dtype, 'unbind', ['float32', 'float64', 'int32', 'int64'],
                'unbind')
    if not isinstance(axis, (int)):
        raise TypeError("The type of 'axis'  must be int, but received %s." %
                        (type(axis)))
    if isinstance(axis, np.generic):
        axis = np.asscalar(axis)
    input_shape = input.shape
    axis_ = axis if axis >= 0 else len(input_shape) + axis
    num = input_shape[axis_]
    outs = [
        helper.create_variable_for_type_inference(dtype=helper.input_dtype())
        for i in range(num)
    ]

    helper.append_op(
        type="unbind",
        inputs={"X": input},
        outputs={"Out": outs},
        attrs={"axis": axis})
    return outs


def scatter(x, index, updates, overwrite=True, name=None):
    """
    **Scatter Layer**
    Output is obtained by updating the input on selected indices based on updates.
    
    .. code-block:: python
        import numpy as np
        #input:
        x = np.array([[1, 1], [2, 2], [3, 3]])
        index = np.array([2, 1, 0, 1])
        # shape of updates should be the same as x
        # shape of updates with dim > 1 should be the same as input
        updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        overwrite = False
        # calculation:
        if not overwrite:
            for i in range(len(index)):
                x[index[i]] = np.zeros((2))
        for i in range(len(index)):
            if (overwrite):
                x[index[i]] = updates[i]
            else:
                x[index[i]] += updates[i]
        # output:
        out = np.array([[3, 3], [6, 6], [1, 1]])
        out.shape # [3, 2]

    **NOTICE**: The order in which updates are applied is nondeterministic, 
    so the output will be nondeterministic if index contains duplicates.

    Args:
        x (Tensor): The input N-D Tensor with ndim>=1. Data type can be float32, float64.
        index (Tensor): The index 1-D Tensor. Data type can be int32, int64. The length of index cannot exceed updates's length, and the value in index cannot exceed input's length.
        updates (Tensor): update input with updates parameter based on index. shape should be the same as input, and dim value with dim > 1 should be the same as input.
        overwrite (bool): The mode that updating the output when there are same indices. 
          If True, use the overwrite mode to update the output of the same index,
	      if False, use the accumulate mode to update the output of the same index.Default value is True.
        name(str, optional): The default value is None. Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .
 
    Returns:
        Tensor: The output is a Tensor with the same shape as x.

    Examples:
        .. code-block:: python
            
            import paddle
            import numpy as np
            paddle.disable_static()

            x_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float32)
            index_data = np.array([2, 1, 0, 1]).astype(np.int64)
            updates_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(np.float32)
            
            x = paddle.to_tensor(x_data)
            index = paddle.to_tensor(index_data)
            updates = paddle.to_tensor(updates_data)
  
            output1 = paddle.scatter(x, index, updates, overwrite=False)
            # [[3., 3.],
            #  [6., 6.],
            #  [1., 1.]]

            output2 = paddle.scatter(x, index, updates, overwrite=True)
            # CPU device:
            # [[3., 3.],
            #  [4., 4.],
            #  [1., 1.]]
            # GPU device maybe have two results because of the repeated numbers in index
            # result 1:
            # [[3., 3.],
            #  [4., 4.],
            #  [1., 1.]]
            # result 2:
            # [[3., 3.],
            #  [2., 2.],
            #  [1., 1.]]
    """
    if in_dygraph_mode():
        return core.ops.scatter(x, index, updates, 'overwrite', overwrite)

    check_variable_and_dtype(x, 'dtype', ['float32', 'float64'], 'scatter')
    check_type(overwrite, 'overwrite', bool, 'scatter')
    helper = LayerHelper('scatter', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type="scatter",
        inputs={"X": x,
                "Ids": index,
                "Updates": updates},
        attrs={'overwrite': overwrite},
        outputs={"Out": out})
    return out


def chunk(x, chunks, axis=0, name=None):
    """
    Split the input tensor into multiple sub-Tensors.
    
    Args:
        x (Tensor): A N-D Tensor. The data type is bool, float16, float32, float64, int32 or int64.
        chunks(int): The number of tensor to be split along the certain axis.
        axis (int|Tensor, optional): The axis along which to split, it can be a scalar with type 
            ``int`` or a ``Tensor`` with shape [1] and data type  ``int32`` or ``int64``.
            If :math::`axis < 0`, the axis to split along is :math:`rank(x) + axis`. Default is 0.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
    Returns:
        list(Tensor): The list of segmented Tensors.
    Raises:
        TypeError: The data type of ``x`` must be one of bool, float16, float32, float64, int32, int64.
        TypeError: ``chunks`` is not int.
        TypeError: ``axis`` is not int or Tensor. the data type of ``axis`` must be int32 or int64 when it's a Tensor.
    Example:
        .. code-block:: python
            
            import numpy as np
            import paddle
            
            paddle.disable_static()
            # x is a Tensor which shape is [3, 9, 5]
            x_np = np.random.random([3, 9, 5]).astype("int32")
            x = paddle.to_tensor(x_np)

            out0, out1, out22 = paddle.chunk(x, chunks=3, axis=1)
            # out0.shape [3, 3, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 3, 5]

            
            # axis is negative, the real axis is (rank(x) + axis) which real
            # value is 1.
            out0, out1, out2 = paddle.chunk(x, chunks=3, axis=-2)
            # out0.shape [3, 3, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 3, 5]
    """
    check_type(chunks, 'chunks', (int), 'chunk')
    return paddle.fluid.layers.split(
        input=x, num_or_sections=chunks, dim=axis, name=name)


def tile(x, repeat_times, name=None):
    """

    Construct a new Tensor by repeating ``x`` the number of times given by ``repeat_times``.
    After tiling, the value of the i'th dimension of the output is equal to ``x.shape[i]*repeat_times[i]``.

    Both the number of dimensions of ``x`` and the number of elements in ``repeat_times`` should be less than or equal to 6.

    Args:
        x (Tensor): The input tensor, its data type should be bool, float32, float64, int32 or int64.
        repeat_times (Tensor|tuple|list): The number of repeating times. If repeat_times is a list or tuple, all its elements
            should be integers or 1-D Tensors with the data type int32. If repeat_times is a Tensor, it should be an 1-D Tensor with the data type int32.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor. The data type is the same as ``x``.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.disable_static()
            np_data = np.array([1, 2, 3]).astype('int32')
            data = paddle.to_tensor(np_data)
            out = paddle.tile(data, repeat_times=[2, 1])
            np_out = out.numpy()
            # [[1, 2, 3], [1, 2, 3]]

            out = paddle.tile(data, repeat_times=[2, 2])
            np_out = out.numpy()
            # [[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]]

            np_repeat_times = np.array([2, 1]).astype("int32")
            repeat_times = paddle.to_tensor(np_repeat_times)
            out = paddle.tile(data, repeat_times=repeat_times)
            np_out = out.numpy()
            # [[1, 2, 3], [1, 2, 3]]
    """
    check_variable_and_dtype(
        x, 'x', ['bool', 'float32', 'float64', 'int32', 'int64'], 'tile')
    check_type(repeat_times, 'repeat_times', (list, tuple, Variable), 'tile')
    if convert_dtype(x.dtype) == 'bool' and x.stop_gradient == False:
        raise ValueError(
            "When the date type is bool for the input 'x' of tile op, you "
            "must set its stop_gradient to be True by "
            "some_var.stop_gradient == True supporting some_var is the input.")

    if in_dygraph_mode():
        return core.ops.tile(x, 'repeat_times', repeat_times)

    helper = LayerHelper('tile', **locals())

    inputs = {"X": [x]}
    attrs = {}

    def get_attr_repeat_times(list_repeat_times):
        attrs_repeat_times = []
        for idx, times in enumerate(list_repeat_times):
            if isinstance(times, Variable):
                attrs_repeat_times.append(-1)
            else:
                attrs_repeat_times.append(times)
                assert times > 0, (
                    "All elements in repeat_times must be positive for tile.")
        return attrs_repeat_times

    if isinstance(repeat_times, Variable):
        repeat_times.stop_gradient = True
        inputs['RepeatTimes'] = repeat_times
        attrs['repeat_times'] = [-1]
    elif isinstance(repeat_times, (list, tuple)):
        attrs['repeat_times'] = get_attr_repeat_times(repeat_times)
        if utils._contain_var(repeat_times):
            inputs['repeat_times_tensor'] = utils._convert_to_tensor_list(
                repeat_times)

    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='tile', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


def expand_as(x, y, name=None):
    """

    Expand the input tensor ``x`` to the same shape as the input tensor ``y``.

    Both the number of dimensions of ``x`` and ``y`` must be less than or equal to 6, and the number of dimensions of ``y`` must be greather than or equal to that of ``x``. The dimension to expand must have a value of 1.

    Args:
        x (Tensor): The input tensor, its data type is bool, float32, float64, int32 or int64.
        y (Tensor): The input tensor that gives the shape to expand to.
        name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        N-D Tensor: A Tensor with the same shape as ``y``. The data type is the same as ``x``.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            paddle.disable_static()

            np_data_x = np.array([1, 2, 3]).astype('int32')
            np_data_y = np.array([[1, 2, 3], [4, 5, 6]]).astype('int32')
            data_x = paddle.to_tensor(np_data_x)
            data_y = paddle.to_tensor(np_data_y)
            out = paddle.expand_as(data_x, data_y)
            np_out = out.numpy()
            # [[1, 2, 3], [1, 2, 3]]
    """
    check_variable_and_dtype(
        x, 'x', ['bool', 'float32', 'float64', 'int32', 'int64'], 'expand_as')
    check_type(y, 'y', Variable, 'expand_as')

    if convert_dtype(x.dtype) == 'bool' and x.stop_gradient == False:
        raise ValueError(
            "When the data type of input 'x' for expand_as is bool, "
            "you must set its stop_gradient to be False by "
            "some_var.stop_gradient = True, supporting "
            "some_var as the input 'x'.")
    inputs = {"X": [x], "target_tensor": [y]}

    if in_dygraph_mode():
        return core.ops.expand_as_v2(x, y)

    helper = LayerHelper('expand_as', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type='expand_as_v2', inputs=inputs, outputs={'Out': out})
    return out


def expand(x, shape, name=None):
    """

    Expand the input tensor to a given shape.

    Both the number of dimensions of ``x`` and the number of elements in ``shape`` should be less than or equal to 6. The dimension to expand must have a value 1.


    Args:
        x (Tensor): The input tensor, its data type is bool, float32, float64, int32 or int64.
        shape (list|tuple|Tensor): The result shape after expanding. The data type is int32. If shape is a list or tuple, all its elements
            should be integers or 1-D Tensors with the data type int32. If shape is a Tensor, it should be an 1-D Tensor with the data type int32. 
            The value -1 in shape means keeping the corresponding dimension unchanged.
        name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        N-D Tensor: A Tensor with the given shape. The data type is the same as ``x``.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            paddle.disable_static()
            np_data = np.array([1, 2, 3]).astype('int32')
            data = paddle.to_tensor(np_data)
            out = paddle.expand(data, shape=[2, 3])
            out = out.numpy()
            # [[1, 2, 3], [1, 2, 3]]
    """
    check_variable_and_dtype(
        x, 'x', ['bool', 'float32', 'float64', 'int32', 'int64'], 'expand')
    check_type(shape, 'shape', (list, tuple, Variable), 'expand')

    inputs = {"X": [x]}
    attrs = {}
    if convert_dtype(x.dtype) == 'bool' and x.stop_gradient == False:
        raise ValueError("When the data type of input 'x' for expand is bool, "
                         "you must set its stop_gradient to be False by "
                         "some_var.stop_gradient = True, supporting "
                         "some_var as the input.")

    if in_dygraph_mode():
        return core.ops.expand_v2(x, 'shape', shape)

    helper = LayerHelper('expand', **locals())

    def get_attr_expand_shape(list_expand_shape):
        attrs_expand_shape = []
        for idx, shape in enumerate(list_expand_shape):
            if isinstance(shape, Variable):
                attrs_expand_shape.append(-1)
            else:
                attrs_expand_shape.append(shape)
                assert shape > 0 or shape == -1, (
                    "All elements in shape of expand must be positive or -1.")
        return attrs_expand_shape

    if isinstance(shape, Variable):
        shape.stop_gradient = True
        inputs['Shape'] = shape
    elif isinstance(shape, (list, tuple)):
        attrs['shape'] = get_attr_expand_shape(shape)
        if utils._contain_var(shape):
            inputs['expand_shapes_tensor'] = utils._convert_to_tensor_list(
                shape)

    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='expand_v2', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


broadcast_to = expand


def reshape(x, shape, name=None):
    """
    :alias_main: paddle.reshape
	:alias: paddle.reshape,paddle.tensor.reshape,paddle.tensor.manipulation.reshape

    This operator changes the shape of ``x`` without changing its data.

    Some tricks exist when specifying the target shape.

    1. -1 means the value of this dimension is inferred from the total element
    number of x and remaining dimensions. Thus one and only one dimension can
    be set -1.

    2. 0 means the actual dimension value is going to be copied from the
    corresponding dimension of x. The index of 0s in shape can not exceed
    the dimension of x.

    Here are some examples to explain it.

    1. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    is [6, 8], the reshape operator will transform x into a 2-D tensor with
    shape [6, 8] and leaving x's data unchanged.

    2. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    specified is [2, 3, -1, 2], the reshape operator will transform x into a
    4-D tensor with shape [2, 3, 4, 2] and leaving x's data unchanged. In this
    case, one dimension of the target shape is set to -1, the value of this
    dimension is inferred from the total element number of x and remaining
    dimensions.

    3. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    is [-1, 0, 3, 2], the reshape operator will transform x into a 4-D tensor
    with shape [2, 4, 3, 2] and leaving x's data unchanged. In this case,
    besides -1, 0 means the actual dimension value is going to be copied from
    the corresponding dimension of x.

    Args:
        x(Tensor): An N-D Tensor. The data type is ``float32``, ``float64``, ``int32`` or ``int64``.
        shape(list|tuple|Tensor): Define the target shape. At most one dimension of the target shape can be -1.
                        The data type is ``int32`` . If ``shape`` is a list or tuple, the elements of it should be integers or Tensors with shape [1].
                        If ``shape`` is an Tensor, it should be an 1-D Tensor .
        name(str, optional): The default value is None. Normally there is no need for user to set this property.
                            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: A reshaped Tensor with the same data type as ``x``.

    Raises:
        ValueError: If more than one elements of ``shape`` is -1.
        ValueError: If the element of ``shape`` is 0, the corresponding dimension should be less than or equal to the dimension of ``x``.
        ValueError: If the elements in ``shape`` is negative except -1.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            paddle.disable_static()

            data = np.random.random([2, 4, 6]).astype("float32")
            x = paddle.to_tensor(data)

            positive_four = paddle.fill_constant([1], "int32", 4)

            out_1 = paddle.reshape(x, [-1, 0, 3, 2])
            # the shape of out_1 is [2,4,3,2].

            out_2 = paddle.reshape(x, shape=[positive_four, 12])
            # the shape of out_2 is [4, 12].

            shape_tensor = paddle.to_tensor(np.array([8, 6]).astype("int32"))
            out_3 = paddle.reshape(x, shape=shape_tensor)
            # the shape of out_2 is [8, 6].
    """
    return paddle.fluid.layers.reshape(x=x, shape=shape, name=name)


def gather_nd(x, index, name=None):
    """
    **Gather Nd Layer**

    This function is actually a high-dimensional extension of :code:`gather`
    and supports for simultaneous indexing by multiple axes. :attr:`index` is a
    K-dimensional integer tensor, which is regarded as a (K-1)-dimensional
    tensor of :attr:`index` into :attr:`input`, where each element defines
    a slice of params:

    .. math::

        output[(i_0, ..., i_{K-2})] = input[index[(i_0, ..., i_{K-2})]]

    Obviously, :code:`index.shape[-1] <= input.rank` . And, the output tensor has
    shape :code:`index.shape[:-1] + input.shape[index.shape[-1]:]` .

    .. code-block:: text

            Given:
                input = [[[ 0,  1,  2,  3],
                          [ 4,  5,  6,  7],
                          [ 8,  9, 10, 11]],
                         [[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]]]
                input.shape = (2, 3, 4)

            * Case 1:
                index = [[1]]

                gather_nd(input, index)
                         = [input[1, :, :]]
                         = [[12, 13, 14, 15],
                            [16, 17, 18, 19],
                            [20, 21, 22, 23]]

            * Case 2:
                index = [[0,2]]

                gather_nd(input, index)
                         = [input[0, 2, :]]
                         = [8, 9, 10, 11]

            * Case 3:
                index = [[1, 2, 3]]

                gather_nd(input, index)
                         = [input[1, 2, 3]]
                         = [23]

    Args:
        x (Tensor): The input Tensor which it's data type should be bool, float32, float64, int32, int64.
        index (Tensor): The index input with rank > 1, index.shape[-1] <= input.rank.
                        Its dtype should be int32, int64.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.
                        For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        output (Tensor): A tensor with the shape index.shape[:-1] + input.shape[index.shape[-1]:]
    
    Raises:
        TypeError: ``x`` must be a Tensor and the data type of ``x`` must be one of float32, float64, int32 and int64.
        TypeError: ``index`` must be a Tensor and the data type of ``index`` must be one of int32 and int64.

    Examples:

        .. code-block:: python
            import paddle
            import numpy as np
            
            paddle.disable_static()
            np_x = np.array([[[1, 2], [3, 4], [5, 6]],
                             [[7, 8], [9, 10], [11, 12]]])
            np_index = [[0, 1]]
            x = paddle.to_tensor(np_x)
            index = paddle.to_tensor(np_index)
            
            output = paddle.gather_nd(x, index) #[[3, 4]]

    """

    return paddle.fluid.layers.gather_nd(input=x, index=index, name=name)
