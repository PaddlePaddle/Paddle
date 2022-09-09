#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle
from ..framework import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype
from ..fluid import layers
from ..framework import core, in_dygraph_mode, _non_static_mode
from ..fluid.framework import _in_legacy_dygraph
from paddle.common_ops_import import convert_np_dtype_to_dtype_
from paddle.common_ops_import import Variable
from paddle.common_ops_import import VarDesc
from paddle import _C_ops, _legacy_C_ops
from .logic import logical_not

# TODO: define searching & indexing functions of a tensor
# from ..fluid.layers import has_inf  #DEFINE_ALIAS
# from ..fluid.layers import has_nan  #DEFINE_ALIAS

__all__ = []


def argsort(x, axis=-1, descending=False, name=None):
    """
    Sorts the input along the given axis, and returns the corresponding index tensor for the sorted output values. The default sort algorithm is ascending, if you want the sort algorithm to be descending, you must set the :attr:`descending` as True.

    Args:
        x(Tensor): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is -1.
        descending(bool, optional) : Descending is a flag, if set to true,
            algorithm will sort by descending order, else sort by
            ascending order. Default is false.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: sorted indices(with the same shape as ``x``
        and with data type int64).

    Examples:

        .. code-block:: python

            import paddle
            
            x = paddle.to_tensor([[[5,8,9,5],
                                   [0,0,1,7],
                                   [6,9,2,4]],
                                  [[5,2,4,2],
                                   [4,7,7,9],
                                   [1,7,0,6]]], 
                                dtype='float32')
            out1 = paddle.argsort(x, axis=-1)
            out2 = paddle.argsort(x, axis=0)
            out3 = paddle.argsort(x, axis=1)
            
            print(out1)
            #[[[0 3 1 2]
            #  [0 1 2 3]
            #  [2 3 0 1]]
            # [[1 3 2 0]
            #  [0 1 2 3]
            #  [2 0 3 1]]]
            
            print(out2)
            #[[[0 1 1 1]
            #  [0 0 0 0]
            #  [1 1 1 0]]
            # [[1 0 0 0]
            #  [1 1 1 1]
            #  [0 0 0 1]]]
            
            print(out3)
            #[[[1 1 1 2]
            #  [0 0 2 0]
            #  [2 2 0 1]]
            # [[2 0 2 0]
            #  [1 1 0 2]
            #  [0 2 1 1]]]
    """
    if in_dygraph_mode():
        _, ids = _C_ops.argsort(x, axis, descending)
        return ids

    if _in_legacy_dygraph():
        _, ids = _legacy_C_ops.argsort(x, 'axis', axis, 'descending',
                                       descending)
        return ids
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int16', 'int32', 'int64', 'uint8'],
        'argsort')

    helper = LayerHelper("argsort", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype,
                                                    stop_gradient=True)
    ids = helper.create_variable_for_type_inference(VarDesc.VarType.INT64,
                                                    stop_gradient=True)
    helper.append_op(type='argsort',
                     inputs={'X': x},
                     outputs={
                         'Out': out,
                         'Indices': ids
                     },
                     attrs={
                         'axis': axis,
                         'descending': descending
                     })
    return ids


def argmax(x, axis=None, keepdim=False, dtype="int64", name=None):
    """
    Computes the indices of the max elements of the input tensor's
    element along the provided axis.

    Args:
        x(Tensor): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. Default is None, the input `x` will be into the flatten tensor, and selecting the min value index.
        keepdim(bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimentions is one fewer than x since the axis is squeezed. Default is False.
        dtype(str|np.dtype, optional): Data type of the output tensor which can
                    be int32, int64. The default value is ``int64`` , and it will
                    return the int64 indices.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, return the tensor of int32 if set :attr:`dtype` is int32, otherwise return the tensor of int64.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[5,8,9,5],
                                 [0,0,1,7],
                                 [6,9,2,4]])
            out1 = paddle.argmax(x)
            print(out1) # 2
            out2 = paddle.argmax(x, axis=0)
            print(out2) 
            # [2, 2, 0, 1]
            out3 = paddle.argmax(x, axis=-1)
            print(out3) 
            # [2, 3, 1]
            out4 = paddle.argmax(x, axis=0, keepdim=True)
            print(out4)
            # [[2, 2, 0, 1]]
    """
    if axis is not None and not isinstance(axis, (int, Variable)):
        raise TypeError(
            "The type of 'axis'  must be int or Tensor or None in argmax, but received %s."
            % (type(axis)))

    if dtype is None:
        raise ValueError(
            "the value of 'dtype' in argmax could not be None, but received None"
        )

    var_dtype = convert_np_dtype_to_dtype_(dtype)
    flatten = False
    if axis is None:
        flatten = True
        axis = 0

    if in_dygraph_mode():
        return _C_ops.argmax(x, axis, keepdim, flatten, var_dtype)
    if _in_legacy_dygraph():
        out = _legacy_C_ops.arg_max(x, 'axis', axis, 'dtype', var_dtype,
                                    'keepdims', keepdim, 'flatten', flatten)
        return out

    helper = LayerHelper("argmax", **locals())
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int16', 'int32', 'int64', 'uint8'],
        'paddle.argmax')
    check_dtype(var_dtype, 'dtype', ['int32', 'int64'], 'argmin')
    attrs = {}
    out = helper.create_variable_for_type_inference(var_dtype)
    attrs['keepdims'] = keepdim
    attrs['axis'] = axis
    attrs['flatten'] = flatten
    attrs['dtype'] = var_dtype
    helper.append_op(type='arg_max',
                     inputs={'X': x},
                     outputs={'Out': [out]},
                     attrs=attrs)
    out.stop_gradient = True
    return out


def argmin(x, axis=None, keepdim=False, dtype="int64", name=None):
    """
    Computes the indices of the min elements of the input tensor's
    element along the provided axis.

    Args:
        x(Tensor): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. Default is None, the input `x` will be into the flatten tensor, and selecting the min value index.
        keepdim(bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimentions is one fewer than x since the axis is squeezed. Default is False.
        dtype(str, optional): Data type of the output tensor which can
                    be int32, int64. The default value is 'int64', and it will
                    return the int64 indices.
        name(str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.
        
    Returns:
        Tensor, return the tensor of `int32` if set :attr:`dtype` is `int32`, otherwise return the tensor of `int64`.

    Examples:
        .. code-block:: python

            import paddle

            x =  paddle.to_tensor([[5,8,9,5],
                                     [0,0,1,7],
                                     [6,9,2,4]])
            out1 = paddle.argmin(x)
            print(out1) # 4
            out2 = paddle.argmin(x, axis=0)
            print(out2) 
            # [1, 1, 1, 2]
            out3 = paddle.argmin(x, axis=-1)
            print(out3) 
            # [0, 0, 2]
            out4 = paddle.argmin(x, axis=0, keepdim=True)
            print(out4)
            # [[1, 1, 1, 2]]
    """
    if axis is not None and not isinstance(axis, (int, Variable)):
        raise TypeError(
            "The type of 'axis'  must be int or Tensor or None in argmin, but received %s."
            % (type(axis)))

    if dtype is None:
        raise ValueError(
            "the value of 'dtype' in argmin could not be None, but received None"
        )

    var_dtype = convert_np_dtype_to_dtype_(dtype)
    flatten = False
    if axis is None:
        flatten = True
        axis = 0

    if in_dygraph_mode():
        return _C_ops.argmin(x, axis, keepdim, flatten, var_dtype)
    if _in_legacy_dygraph():
        out = _legacy_C_ops.arg_min(x, 'axis', axis, 'dtype', var_dtype,
                                    'keepdims', keepdim, 'flatten', flatten)
        return out

    helper = LayerHelper("argmin", **locals())
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int16', 'int32', 'int64', 'uint8'],
        'paddle.argmin')
    check_dtype(var_dtype, 'dtype', ['int32', 'int64'], 'argmin')
    out = helper.create_variable_for_type_inference(var_dtype)
    attrs = {}
    attrs['keepdims'] = keepdim
    attrs['axis'] = axis
    attrs['flatten'] = flatten
    attrs['dtype'] = var_dtype
    helper.append_op(type='arg_min',
                     inputs={'X': x},
                     outputs={'Out': [out]},
                     attrs=attrs)
    out.stop_gradient = True
    return out


def index_select(x, index, axis=0, name=None):
    """

    Returns a new tensor which indexes the ``input`` tensor along dimension ``axis`` using 
    the entries in ``index`` which is a Tensor. The returned tensor has the same number 
    of dimensions as the original ``x`` tensor. The dim-th dimension has the same 
    size as the length of ``index``; other dimensions have the same size as in the ``x`` tensor. 

    Args:
        x (Tensor): The input Tensor to be operated. The data of ``x`` can be one of float32, float64, int32, int64.
        index (Tensor): The 1-D Tensor containing the indices to index. The data type of ``index`` must be int32 or int64.
        axis (int, optional): The dimension in which we index. Default: if None, the ``axis`` is 0.
        name(str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: A Tensor with same data type as ``x``.
    
    Examples:
        .. code-block:: python
            
            import paddle

            x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                                  [5.0, 6.0, 7.0, 8.0],
                                  [9.0, 10.0, 11.0, 12.0]])
            index = paddle.to_tensor([0, 1, 1], dtype='int32')
            out_z1 = paddle.index_select(x=x, index=index)
            #[[1. 2. 3. 4.]
            # [5. 6. 7. 8.]
            # [5. 6. 7. 8.]]
            out_z2 = paddle.index_select(x=x, index=index, axis=1)
            #[[ 1.  2.  2.]
            # [ 5.  6.  6.]
            # [ 9. 10. 10.]]
    """

    if in_dygraph_mode():
        return _C_ops.index_select(x, index, axis)

    if _in_legacy_dygraph():
        return _legacy_C_ops.index_select(x, index, 'dim', axis)

    helper = LayerHelper("index_select", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'paddle.tensor.search.index_select')
    check_variable_and_dtype(index, 'index', ['int32', 'int64'],
                             'paddle.tensor.search.index_select')

    out = helper.create_variable_for_type_inference(x.dtype)

    helper.append_op(type='index_select',
                     inputs={
                         'X': x,
                         'Index': index
                     },
                     outputs={'Out': out},
                     attrs={'dim': axis})
    return out


def nonzero(x, as_tuple=False):
    """
    Return a tensor containing the indices of all non-zero elements of the `input` 
    tensor. If as_tuple is True, return a tuple of 1-D tensors, one for each dimension 
    in `input`, each containing the indices (in that dimension) of all non-zero elements 
    of `input`. Given a n-Dimensional `input` tensor with shape [x_1, x_2, ..., x_n], If 
    as_tuple is False, we can get a output tensor with shape [z, n], where `z` is the 
    number of all non-zero elements in the `input` tensor. If as_tuple is True, we can get 
    a 1-D tensor tuple of length `n`, and the shape of each 1-D tensor is [z, 1].

    Args:
        x (Tensor): The input tensor variable.
        as_tuple (bool): Return type, Tensor or tuple of Tensor.

    Returns:
        Tensor. The data type is int64.

    Examples:

        .. code-block:: python

            import paddle

            x1 = paddle.to_tensor([[1.0, 0.0, 0.0],
                                   [0.0, 2.0, 0.0],
                                   [0.0, 0.0, 3.0]])
            x2 = paddle.to_tensor([0.0, 1.0, 0.0, 3.0])
            out_z1 = paddle.nonzero(x1)
            print(out_z1)
            #[[0 0]
            # [1 1]
            # [2 2]]
            out_z1_tuple = paddle.nonzero(x1, as_tuple=True)
            for out in out_z1_tuple:
                print(out)
            #[[0]
            # [1]
            # [2]]
            #[[0]
            # [1]
            # [2]]
            out_z2 = paddle.nonzero(x2)
            print(out_z2)
            #[[1]
            # [3]]
            out_z2_tuple = paddle.nonzero(x2, as_tuple=True)
            for out in out_z2_tuple:
                print(out)
            #[[1]
            # [3]]

    """
    list_out = []
    shape = x.shape
    rank = len(shape)

    if in_dygraph_mode():
        outs = _C_ops.where_index(x)
    elif paddle.in_dynamic_mode():
        outs = _legacy_C_ops.where_index(x)
    else:
        helper = LayerHelper("where_index", **locals())

        outs = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.INT64)

        helper.append_op(type='where_index',
                         inputs={'Condition': x},
                         outputs={'Out': [outs]})

    if not as_tuple:
        return outs
    elif rank == 1:
        return tuple([outs])
    else:
        for i in range(rank):
            list_out.append(
                paddle.slice(outs, axes=[1], starts=[i], ends=[i + 1]))
        return tuple(list_out)


def sort(x, axis=-1, descending=False, name=None):
    """

    Sorts the input along the given axis, and returns the sorted output tensor. The default sort algorithm is ascending, if you want the sort algorithm to be descending, you must set the :attr:`descending` as True.

    Args:
        x(Tensor): An input N-D Tensor with type float32, float64, int16,
            int32, int64, uint8.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is Rank(x). when axis<0, it works the same way
            as axis+R. Default is -1.
        descending(bool, optional) : Descending is a flag, if set to true,
            algorithm will sort by descending order, else sort by
            ascending order. Default is false.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.
        
    Returns:
        Tensor: sorted tensor(with the same shape and data type as ``x``).
    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[[5,8,9,5],
                                   [0,0,1,7],
                                   [6,9,2,4]],
                                  [[5,2,4,2],
                                   [4,7,7,9],
                                   [1,7,0,6]]], 
                                 dtype='float32')
            out1 = paddle.sort(x=x, axis=-1)
            out2 = paddle.sort(x=x, axis=0)
            out3 = paddle.sort(x=x, axis=1)
            print(out1)
            #[[[5. 5. 8. 9.]
            #  [0. 0. 1. 7.]
            #  [2. 4. 6. 9.]]
            # [[2. 2. 4. 5.]
            #  [4. 7. 7. 9.]
            #  [0. 1. 6. 7.]]]
            print(out2)
            #[[[5. 2. 4. 2.]
            #  [0. 0. 1. 7.]
            #  [1. 7. 0. 4.]]
            # [[5. 8. 9. 5.]
            #  [4. 7. 7. 9.]
            #  [6. 9. 2. 6.]]]
            print(out3)
            #[[[0. 0. 1. 4.]
            #  [5. 8. 2. 5.]
            #  [6. 9. 9. 7.]]
            # [[1. 2. 0. 2.]
            #  [4. 7. 4. 6.]
            #  [5. 7. 7. 9.]]]
    """
    if in_dygraph_mode():
        outs, _ = _C_ops.argsort(x, axis, descending)
        return outs

    if _in_legacy_dygraph():
        outs, _ = _legacy_C_ops.argsort(x, 'axis', axis, 'descending',
                                        descending)
        return outs
    helper = LayerHelper("sort", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype,
                                                    stop_gradient=False)
    ids = helper.create_variable_for_type_inference(VarDesc.VarType.INT64,
                                                    stop_gradient=True)
    helper.append_op(type='argsort',
                     inputs={'X': x},
                     outputs={
                         'Out': out,
                         'Indices': ids
                     },
                     attrs={
                         'axis': axis,
                         'descending': descending
                     })
    return out


def mode(x, axis=-1, keepdim=False, name=None):
    """
    Used to find values and indices of the modes at the optional axis.

    Args:
        x(Tensor): Tensor, an input N-D Tensor with type float32, float64, int32, int64.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. Default is -1.
        keepdim(bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimentions is one fewer than x since the axis is squeezed. Default is False.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        tuple(Tensor), return the values and indices. The value data type is the same as the input `x`. The indices data type is int64.

    Examples:

        .. code-block:: python

           import paddle
           
           tensor = paddle.to_tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]], dtype=paddle.float32)
           res = paddle.mode(tensor, 2)
           print(res)
           # (Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
           #   [[2., 3.],
           #    [5., 9.]]), Tensor(shape=[2, 2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
           #   [[1, 1],
           #    [1, 0]]))
           
    """
    if in_dygraph_mode():
        return _C_ops.mode(x, axis, keepdim)
    if _in_legacy_dygraph():
        return _legacy_C_ops.mode(x, "axis", axis, "keepdim", keepdim)

    helper = LayerHelper("mode", **locals())
    inputs = {"X": [x]}
    attrs = {}
    attrs['axis'] = axis
    attrs['keepdim'] = keepdim

    values = helper.create_variable_for_type_inference(dtype=x.dtype)
    indices = helper.create_variable_for_type_inference(dtype="int64")

    helper.append_op(type="mode",
                     inputs=inputs,
                     outputs={
                         "Out": [values],
                         "Indices": [indices]
                     },
                     attrs=attrs)
    indices.stop_gradient = True
    return values, indices


def where(condition, x=None, y=None, name=None):
    r"""
    Return a Tensor of elements selected from either :attr:`x` or :attr:`y` according to corresponding elements of :attr:`condition`. Concretely,

    .. math::

        out_i =
        \begin{cases}
        x_i, & \text{if}  \ condition_i \  \text{is} \ True \\
        y_i, & \text{if}  \ condition_i \  \text{is} \ False \\
        \end{cases}.

    Notes:
        ``numpy.where(condition)`` is identical to ``paddle.nonzero(condition, as_tuple=True)``, please refer to :ref:`api_tensor_search_nonzero`.

    Args:
        condition (Tensor): The condition to choose x or y. When True (nonzero), yield x, otherwise yield y.
        x (Tensor|scalar, optional): A Tensor or scalar to choose when the condition is True with data type of float32, float64, int32 or int64. Either both or neither of x and y should be given.
        y (Tensor|scalar, optional): A Tensor or scalar to choose when the condition is False with data type of float32, float64, int32 or int64. Either both or neither of x and y should be given.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor: A Tensor with the same shape as :attr:`condition` and same data type as :attr:`x` and :attr:`y`.

    Examples:
        
        .. code-block:: python

            import paddle
            
            x = paddle.to_tensor([0.9383, 0.1983, 3.2, 1.2])
            y = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
            
            out = paddle.where(x>1, x, y)
            print(out)
            #out: [1.0, 1.0, 3.2, 1.2]
            
            out = paddle.where(x>1)
            print(out)
            #out: (Tensor(shape=[2, 1], dtype=int64, place=CPUPlace, stop_gradient=True,
            #            [[2],
            #             [3]]),)
    """
    if np.isscalar(x):
        x = paddle.full([1], x, np.array([x]).dtype.name)

    if np.isscalar(y):
        y = paddle.full([1], y, np.array([y]).dtype.name)

    if x is None and y is None:
        return nonzero(condition, as_tuple=True)

    if x is None or y is None:
        raise ValueError("either both or neither of x and y should be given")

    if not paddle.in_dynamic_mode():
        check_variable_and_dtype(condition, 'condition', ['bool'], 'where')
        check_variable_and_dtype(x, 'x',
                                 ['float32', 'float64', 'int32', 'int64'],
                                 'where')
        check_variable_and_dtype(y, 'y',
                                 ['float32', 'float64', 'int32', 'int64'],
                                 'where')

    condition_shape = list(condition.shape)
    x_shape = list(x.shape)
    y_shape = list(y.shape)

    if x_shape == y_shape and condition_shape == x_shape:
        broadcast_condition = condition
        broadcast_x = x
        broadcast_y = y
    else:
        zeros_like_x = paddle.zeros_like(x)
        zeros_like_y = paddle.zeros_like(y)
        zeros_like_condition = paddle.zeros_like(condition)
        zeros_like_condition = paddle.cast(zeros_like_condition, x.dtype)
        cast_cond = paddle.cast(condition, x.dtype)

        broadcast_zeros = paddle.add(zeros_like_x, zeros_like_y)
        broadcast_zeros = paddle.add(broadcast_zeros, zeros_like_condition)
        broadcast_x = paddle.add(x, broadcast_zeros)
        broadcast_y = paddle.add(y, broadcast_zeros)
        broadcast_condition = paddle.add(cast_cond, broadcast_zeros)
        broadcast_condition = paddle.cast(broadcast_condition, 'bool')

    if in_dygraph_mode():
        return _C_ops.where(broadcast_condition, broadcast_x, broadcast_y)
    else:
        if _in_legacy_dygraph():
            return _legacy_C_ops.where(broadcast_condition, broadcast_x,
                                       broadcast_y)
        else:
            helper = LayerHelper("where", **locals())
            out = helper.create_variable_for_type_inference(dtype=x.dtype)

            helper.append_op(type='where',
                             inputs={
                                 'Condition': broadcast_condition,
                                 'X': broadcast_x,
                                 'Y': broadcast_y
                             },
                             outputs={'Out': [out]})

            return out


def index_sample(x, index):
    """
    **IndexSample Layer**

    IndexSample OP returns the element of the specified location of X, 
    and the location is specified by Index. 

    .. code-block:: text


                Given:

                X = [[1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10]]

                Index = [[0, 1, 3],
                         [0, 2, 4]]

                Then:

                Out = [[1, 2, 4],
                       [6, 8, 10]]

    Args:
        x (Tensor): The source input tensor with 2-D shape. Supported data type is 
            int32, int64, float32, float64.
        index (Tensor): The index input tensor with 2-D shape, first dimension should be same with X. 
            Data type is int32 or int64.

    Returns:
        output (Tensor): The output is a tensor with the same shape as index.

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                                  [5.0, 6.0, 7.0, 8.0],
                                  [9.0, 10.0, 11.0, 12.0]], dtype='float32')
            index = paddle.to_tensor([[0, 1, 2],
                                      [1, 2, 3],
                                      [0, 0, 0]], dtype='int32')
            target = paddle.to_tensor([[100, 200, 300, 400],
                                       [500, 600, 700, 800],
                                       [900, 1000, 1100, 1200]], dtype='int32')
            out_z1 = paddle.index_sample(x, index)
            print(out_z1)
            #[[1. 2. 3.]
            # [6. 7. 8.]
            # [9. 9. 9.]]

            # Use the index of the maximum value by topk op
            # get the value of the element of the corresponding index in other tensors
            top_value, top_index = paddle.topk(x, k=2)
            out_z2 = paddle.index_sample(target, top_index)
            print(top_value)
            #[[ 4.  3.]
            # [ 8.  7.]
            # [12. 11.]]

            print(top_index)
            #[[3 2]
            # [3 2]
            # [3 2]]

            print(out_z2)
            #[[ 400  300]
            # [ 800  700]
            # [1200 1100]]

    """
    if in_dygraph_mode():
        return _C_ops.index_sample(x, index)
    else:
        if _in_legacy_dygraph():
            return _legacy_C_ops.index_sample(x, index)
        else:
            helper = LayerHelper("index_sample", **locals())
            check_variable_and_dtype(x, 'x',
                                     ['float32', 'float64', 'int32', 'int64'],
                                     'paddle.tensor.search.index_sample')
            check_variable_and_dtype(index, 'index', ['int32', 'int64'],
                                     'paddle.tensor.search.index_sample')
            out = helper.create_variable_for_type_inference(dtype=x.dtype)

            helper.append_op(type='index_sample',
                             inputs={
                                 'X': x,
                                 'Index': index
                             },
                             outputs={'Out': out})
            return out


def masked_select(x, mask, name=None):
    """
    Returns a new 1-D tensor which indexes the input tensor according to the ``mask``
    which is a tensor with data type of bool.

    Args:
        x (Tensor): The input Tensor, the data type can be int32, int64, float32, float64. 
        mask (Tensor): The Tensor containing the binary mask to index with, it's data type is bool.
        name(str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns: 
        A 1-D Tensor which is the same data type  as ``x``.
    
    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                                  [5.0, 6.0, 7.0, 8.0],
                                  [9.0, 10.0, 11.0, 12.0]])
            mask = paddle.to_tensor([[True, False, False, False],
                                     [True, True, False, False],
                                     [True, False, False, False]])
            out = paddle.masked_select(x, mask)
            #[1.0 5.0 6.0 9.0]
    """

    if in_dygraph_mode():
        return _C_ops.masked_select(x, mask)

    if _in_legacy_dygraph():
        return _legacy_C_ops.masked_select(x, mask)

    helper = LayerHelper("masked_select", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'paddle.tensor.search.mask_select')
    check_variable_and_dtype(mask, 'mask', ['bool'],
                             'paddle.tensor.search.masked_select')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='masked_select',
                     inputs={
                         'X': x,
                         'Mask': mask
                     },
                     outputs={'Y': out})
    return out


def topk(x, k, axis=None, largest=True, sorted=True, name=None):
    """
    Return values and indices of the k largest or smallest at the optional axis.
    If the input is a 1-D Tensor, finds the k largest or smallest values and indices.
    If the input is a Tensor with higher rank, this operator computes the top k values and indices along the :attr:`axis`.

    Args:
        x(Tensor): Tensor, an input N-D Tensor with type float32, float64, int32, int64.
        k(int, Tensor): The number of top elements to look for along the axis.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. Default is -1.
        largest(bool, optional) : largest is a flag, if set to true,
            algorithm will sort by descending order, otherwise sort by
            ascending order. Default is True.
        sorted(bool, optional): controls whether to return the elements in sorted order, default value is True. In gpu device, it always return the sorted value. 
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        tuple(Tensor), return the values and indices. The value data type is the same as the input `x`. The indices data type is int64.

    Examples:

        .. code-block:: python

            import paddle

            data_1 = paddle.to_tensor([1, 4, 5, 7])
            value_1, indices_1 = paddle.topk(data_1, k=1)
            print(value_1) # [7]
            print(indices_1) # [3]

            data_2 = paddle.to_tensor([[1, 4, 5, 7], [2, 6, 2, 5]])
            value_2, indices_2 = paddle.topk(data_2, k=1)
            print(value_2) # [[7], [6]]
            print(indices_2) # [[3], [1]]

            value_3, indices_3 = paddle.topk(data_2, k=1, axis=-1)
            print(value_3) # [[7], [6]]
            print(indices_3) # [[3], [1]]

            value_4, indices_4 = paddle.topk(data_2, k=1, axis=0)
            print(value_4) # [[2, 6, 5, 7]]
            print(indices_4) # [[1, 1, 0, 0]]


    """

    if in_dygraph_mode():
        if axis == None:
            axis = -1
        out, indices = _C_ops.top_k(x, k, axis, largest, sorted)
        return out, indices

    if _non_static_mode():
        if axis is None:
            out, indices = _legacy_C_ops.top_k_v2(x, 'k', int(k), 'largest',
                                                  largest, 'sorted', sorted)
        else:
            out, indices = _legacy_C_ops.top_k_v2(x, 'k', int(k), 'axis', axis,
                                                  'largest', largest, 'sorted',
                                                  sorted)
        return out, indices

    helper = LayerHelper("top_k_v2", **locals())
    inputs = {"X": [x]}
    attrs = {}
    if isinstance(k, Variable):
        inputs['K'] = [k]
    else:
        attrs = {'k': k}
    attrs['largest'] = largest
    attrs['sorted'] = sorted
    if axis is not None:
        attrs['axis'] = axis

    values = helper.create_variable_for_type_inference(dtype=x.dtype)
    indices = helper.create_variable_for_type_inference(dtype="int64")

    helper.append_op(type="top_k_v2",
                     inputs=inputs,
                     outputs={
                         "Out": [values],
                         "Indices": [indices]
                     },
                     attrs=attrs)
    indices.stop_gradient = True
    return values, indices


def bucketize(x, sorted_sequence, out_int32=False, right=False, name=None):
    """
    This API is used to find the index of the corresponding 1D tensor `sorted_sequence` in the innermost dimension based on the given `x`.

    Args:
        x(Tensor): An input N-D tensor value with type int32, int64, float32, float64.
        sorted_sequence(Tensor): An input 1-D tensor with type int32, int64, float32, float64. The value of the tensor monotonically increases in the innermost dimension. 
        out_int32(bool, optional): Data type of the output tensor which can be int32, int64. The default value is False, and it indicates that the output data type is int64.
        right(bool, optional): Find the upper or lower bounds of the sorted_sequence range in the innermost dimension based on the given `x`. If the value of the sorted_sequence is nan or inf, return the size of the innermost dimension.
                               The default value is False and it shows the lower bounds.  
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.
        
    Returns:
        Tensor（the same sizes of the `x`）, return the tensor of int32 if set :attr:`out_int32` is True, otherwise return the tensor of int64.  
    
    Examples:

        .. code-block:: python
    
            import paddle

            sorted_sequence = paddle.to_tensor([2, 4, 8, 16], dtype='int32')
            x = paddle.to_tensor([[0, 8, 4, 16], [-1, 2, 8, 4]], dtype='int32')
            out1 = paddle.bucketize(x, sorted_sequence)
            print(out1)
            # Tensor(shape=[2, 4], dtype=int64, place=CPUPlace, stop_gradient=True,
            #        [[0, 2, 1, 3],
            #         [0, 0, 2, 1]])
            out2 = paddle.bucketize(x, sorted_sequence, right=True)
            print(out2)
            # Tensor(shape=[2, 4], dtype=int64, place=CPUPlace, stop_gradient=True,
            #        [[0, 3, 2, 4],
            #         [0, 1, 3, 2]])
            out3 = x.bucketize(sorted_sequence)
            print(out3)
            # Tensor(shape=[2, 4], dtype=int64, place=CPUPlace, stop_gradient=True,
            #        [[0, 2, 1, 3],
            #         [0, 0, 2, 1]])
            out4 = x.bucketize(sorted_sequence, right=True)
            print(out4)
            # Tensor(shape=[2, 4], dtype=int64, place=CPUPlace, stop_gradient=True,
            #        [[0, 3, 2, 4],
            #         [0, 1, 3, 2]])
            
    """
    check_variable_and_dtype(sorted_sequence, 'SortedSequence',
                             ['float32', 'float64', 'int32', 'int64'],
                             'paddle.searchsorted')
    if sorted_sequence.dim() != 1:
        raise ValueError(
            f"sorted_sequence tensor must be 1 dimension, but got dim {sorted_sequence.dim()}"
        )
    return searchsorted(sorted_sequence, x, out_int32, right, name)


def searchsorted(sorted_sequence,
                 values,
                 out_int32=False,
                 right=False,
                 name=None):
    """
    Find the index of the corresponding `sorted_sequence` in the innermost dimension based on the given `values`.

    Args:
        sorted_sequence(Tensor): An input N-D or 1-D tensor with type int32, int64, float32, float64. The value of the tensor monotonically increases in the innermost dimension. 
        values(Tensor): An input N-D tensor value with type int32, int64, float32, float64.
        out_int32(bool, optional): Data type of the output tensor which can be int32, int64. The default value is False, and it indicates that the output data type is int64.
        right(bool, optional): Find the upper or lower bounds of the sorted_sequence range in the innermost dimension based on the given `values`. If the value of the sorted_sequence is nan or inf, return the size of the innermost dimension.
                               The default value is False and it shows the lower bounds.  
        name(str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.
        
    Returns:
        Tensor（the same sizes of the `values`）, return the tensor of int32 if set :attr:`out_int32` is True, otherwise return the tensor of int64.  
    
    Examples:

        .. code-block:: python
    
            import paddle

            sorted_sequence = paddle.to_tensor([[1, 3, 5, 7, 9, 11],
                                                [2, 4, 6, 8, 10, 12]], dtype='int32')
            values = paddle.to_tensor([[3, 6, 9, 10], [3, 6, 9, 10]], dtype='int32')
            out1 = paddle.searchsorted(sorted_sequence, values)
            print(out1)
            # Tensor(shape=[2, 4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #        [[1, 3, 4, 5],
            #         [1, 2, 4, 4]])
            out2 = paddle.searchsorted(sorted_sequence, values, right=True)
            print(out2)
            # Tensor(shape=[2, 4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #        [[2, 3, 5, 5],
            #         [1, 3, 4, 5]])
            sorted_sequence_1d = paddle.to_tensor([1, 3, 5, 7, 9, 11, 13])
            out3 = paddle.searchsorted(sorted_sequence_1d, values)     
            print(out3)
            # Tensor(shape=[2, 4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #        [[1, 3, 4, 5],
            #         [1, 3, 4, 5]])
            
    """
    if in_dygraph_mode():
        return _C_ops.searchsorted(sorted_sequence, values, out_int32, right)

    if _in_legacy_dygraph():
        return _legacy_C_ops.searchsorted(sorted_sequence, values, "out_int32",
                                          out_int32, "right", right)

    check_variable_and_dtype(sorted_sequence, 'SortedSequence',
                             ['float32', 'float64', 'int32', 'int64'],
                             'paddle.searchsorted')
    check_variable_and_dtype(values, 'Values',
                             ['float32', 'float64', 'int32', 'int64'],
                             'paddle.searchsorted')

    helper = LayerHelper('searchsorted', **locals())
    out_type = 'int32' if out_int32 else 'int64'
    out = helper.create_variable_for_type_inference(dtype=out_type)
    helper.append_op(type='searchsorted',
                     inputs={
                         'SortedSequence': sorted_sequence,
                         "Values": values
                     },
                     outputs={'Out': out},
                     attrs={
                         "out_int32": out_int32,
                         "right": right
                     })

    return out


def kthvalue(x, k, axis=None, keepdim=False, name=None):
    """
    Find values and indices of the k-th smallest at the axis.

    Args:
        x(Tensor): A N-D Tensor with type float32, float64, int32, int64.
        k(int): The k for the k-th smallest number to look for along the axis.
        axis(int, optional): Axis to compute indices along. The effective range
            is [-R, R), where R is x.ndim. when axis < 0, it works the same way
            as axis + R. The default is None. And if the axis is None, it will computed as -1 by default.
        keepdim(bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimentions is one fewer than x since the axis is squeezed. Default is False.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        tuple(Tensor), return the values and indices. The value data type is the same as the input `x`. The indices data type is int64.
   
    Examples:

        .. code-block:: python
    
            import paddle
            
            x = paddle.randn((2,3,2))
            # Tensor(shape=[2, 3, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [[[ 0.22954939, -0.01296274],
            #         [ 1.17135799, -0.34493217],
            #         [-0.19550551, -0.17573971]],
            #
            #        [[ 0.15104349, -0.93965352],
            #         [ 0.14745511,  0.98209465],
            #         [ 0.10732264, -0.55859774]]])           
            y = paddle.kthvalue(x, 2, 1)    
            # (Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            # [[ 0.22954939, -0.17573971],
            #  [ 0.14745511, -0.55859774]]), Tensor(shape=[2, 2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
            #  [[0, 2],
            #  [1, 2]]))
    """
    if _non_static_mode():
        if axis is not None:
            if _in_legacy_dygraph():
                return _legacy_C_ops.kthvalue(x, 'k', k, "axis", axis,
                                              "keepdim", keepdim)
            return _C_ops.kthvalue(x, k, axis, keepdim)
        else:
            if _in_legacy_dygraph():
                return _legacy_C_ops.kthvalue(x, 'k', k, "keepdim", keepdim)
            return _C_ops.kthvalue(x, k, -1, keepdim)

    helper = LayerHelper("kthvalue", **locals())
    inputs = {"X": [x]}
    attrs = {'k': k}
    if axis is not None:
        attrs['axis'] = axis
    values = helper.create_variable_for_type_inference(dtype=x.dtype)
    indices = helper.create_variable_for_type_inference(dtype="int64")

    helper.append_op(type="kthvalue",
                     inputs=inputs,
                     outputs={
                         "Out": [values],
                         "Indices": [indices]
                     },
                     attrs=attrs)
    indices.stop_gradient = True
    return values, indices
