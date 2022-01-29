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

import numpy as np
from ..fluid.framework import Variable
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import core, in_dygraph_mode
from ..fluid import layers
from .search import where
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
import paddle
from paddle import _C_ops

__all__ = []


def mean(x, axis=None, keepdim=False, name=None):
    """
    Computes the mean of the input tensor's elements along ``axis``.

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        axis (int|list|tuple, optional): The axis along which to perform mean
            calculations. ``axis`` should be int, list(int) or tuple(int). If
            ``axis`` is a list/tuple of dimension(s), mean is calculated along
            all element(s) of ``axis`` . ``axis`` or element(s) of ``axis``
            should be in range [-D, D), where D is the dimensions of ``x`` . If
            ``axis`` or element(s) of ``axis`` is less than 0, it works the
            same way as :math:`axis + D` . If ``axis`` is None, mean is
            calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of average along ``axis`` of ``x``, with the same data
        type as ``x``.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[[1., 2., 3., 4.],
                                   [5., 6., 7., 8.],
                                   [9., 10., 11., 12.]],
                                  [[13., 14., 15., 16.],
                                   [17., 18., 19., 20.],
                                   [21., 22., 23., 24.]]])
            out1 = paddle.mean(x)
            # [12.5]
            out2 = paddle.mean(x, axis=-1)
            # [[ 2.5  6.5 10.5]
            #  [14.5 18.5 22.5]]
            out3 = paddle.mean(x, axis=-1, keepdim=True)
            # [[[ 2.5]
            #   [ 6.5]
            #   [10.5]]
            #  [[14.5]
            #   [18.5]
            #   [22.5]]]
            out4 = paddle.mean(x, axis=[0, 2])
            # [ 8.5 12.5 16.5]
    """

    if isinstance(axis, int):
        axis = [axis]
    reduce_all = True if axis is None \
        or len(axis)==0 \
        or len(axis) == len(x.shape) else False
    if axis is None or len(axis) == 0:
        axis = [0]

    if in_dygraph_mode():
        return _C_ops.reduce_mean(x, 'dim', axis, 'keep_dim', keepdim,
                                  'reduce_all', reduce_all)

    check_variable_and_dtype(x, 'x/input',
                             ['uint16', 'float16', 'float32', 'float64'],
                             'mean/reduce_mean')
    check_type(axis, 'axis/dim', (int, list, tuple), 'mean/reduce_mean')
    if isinstance(axis, (list, tuple)):
        for item in axis:
            check_type(item, 'elements of axis/dim', (int), 'mean/reduce_mean')

    helper = LayerHelper('mean', **locals())
    attrs = {'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all}
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='reduce_mean', inputs={'X': x}, outputs={'Out': out}, attrs=attrs)
    return out


def var(x, axis=None, unbiased=True, keepdim=False, name=None):
    """
    Computes the variance of ``x`` along ``axis`` .

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        axis (int|list|tuple, optional): The axis along which to perform
            variance calculations. ``axis`` should be int, list(int) or
            tuple(int). If ``axis`` is a list/tuple of dimension(s), variance
            is calculated along all element(s) of ``axis`` . ``axis`` or
            element(s) of ``axis`` should be in range [-D, D), where D is the
            dimensions of ``x`` . If ``axis`` or element(s) of ``axis`` is less
            than 0, it works the same way as :math:`axis + D` . If ``axis`` is
            None, variance is calculated over all elements of ``x``. Default
            is None.
        unbiased (bool, optional): Whether to use the unbiased estimation. If
            ``unbiased`` is True, the divisor used in the computation is
            :math:`N - 1`, where :math:`N` represents the number of elements
            along ``axis`` , otherwise the divisor is :math:`N`. Default is True.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of variance along ``axis`` of ``x``, with the same data
        type as ``x``.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
            out1 = paddle.var(x)
            # [2.66666667]
            out2 = paddle.var(x, axis=1)
            # [1.         4.33333333]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'var')

    u = mean(x, axis, True, name)
    out = paddle.sum((x - u)**2, axis, keepdim=keepdim, name=name)

    n = paddle.cast(paddle.numel(x), x.dtype) \
        / paddle.cast(paddle.numel(out), x.dtype)
    if unbiased:
        one_const = paddle.ones([1], x.dtype)
        n = where(n > one_const, n - 1., one_const)
    out /= n
    return out


def std(x, axis=None, unbiased=True, keepdim=False, name=None):
    """
    Computes the standard-deviation of ``x`` along ``axis`` .

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        axis (int|list|tuple, optional): The axis along which to perform
            standard-deviation calculations. ``axis`` should be int, list(int)
            or tuple(int). If ``axis`` is a list/tuple of dimension(s),
            standard-deviation is calculated along all element(s) of ``axis`` .
            ``axis`` or element(s) of ``axis`` should be in range [-D, D),
            where D is the dimensions of ``x`` . If ``axis`` or element(s) of
            ``axis`` is less than 0, it works the same way as :math:`axis + D` .
            If ``axis`` is None, standard-deviation is calculated over all
            elements of ``x``. Default is None.
        unbiased (bool, optional): Whether to use the unbiased estimation. If
            ``unbiased`` is True, the standard-deviation is calculated via the
            unbiased estimator. If ``unbiased`` is True,  the divisor used in
            the computation is :math:`N - 1`, where :math:`N` represents the
            number of elements along ``axis`` , otherwise the divisor is
            :math:`N`. Default is True.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of standard-deviation along ``axis`` of ``x``, with the
        same data type as ``x``.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
            out1 = paddle.std(x)
            # [1.63299316]
            out2 = paddle.std(x, axis=1)
            # [1.       2.081666]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'std')

    out = var(**locals())
    return paddle.sqrt(out)


def numel(x, name=None):
    """
    Returns the number of elements for a tensor, which is a int64 Tensor with shape [1] in static mode
    or a scalar value in imperative mode

    Args:
        x (Tensor): The input Tensor, it's data type can be bool, float16, float32, float64, int32, int64.

    Returns:
        Tensor: The number of elements for the input Tensor.

    Examples:
        .. code-block:: python

            import paddle
            
            x = paddle.full(shape=[4, 5, 7], fill_value=0, dtype='int32')
            numel = paddle.numel(x) # 140


    """
    if in_dygraph_mode():
        return _C_ops.size(x)

    if not isinstance(x, Variable):
        raise TypeError("x must be a Tensor in numel")
    helper = LayerHelper('numel', **locals())
    out = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.INT64)
    helper.append_op(type='size', inputs={'Input': x}, outputs={'Out': out})
    return out


def median(x, axis=None, keepdim=False, name=None):
    """
    Compute the median along the specified axis.

    Args:
        x (Tensor): The input Tensor, it's data type can be bool, float16, float32, float64, int32, int64.
        axis (int, optional): The axis along which to perform median calculations ``axis`` should be int.
            ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
            If ``axis`` is less than 0, it works the same way as :math:`axis + D`.
            If ``axis`` is None, median is calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of median along ``axis`` of ``x``. If data type of ``x`` is float64, data type of results will be float64, otherwise data type will be float32.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.arange(12).reshape([3, 4])
            # x is [[0 , 1 , 2 , 3 ],
            #       [4 , 5 , 6 , 7 ],
            #       [8 , 9 , 10, 11]]

            y1 = paddle.median(x)
            # y1 is [5.5]

            y2 = paddle.median(x, axis=0)
            # y2 is [4., 5., 6., 7.]

            y3 = paddle.median(x, axis=1)
            # y3 is [1.5, 5.5, 9.5]

            y4 = paddle.median(x, axis=0, keepdim=True)
            # y4 is [[4., 5., 6., 7.]]

    """
    if not isinstance(x, Variable):
        raise TypeError("In median, the input x should be a Tensor.")
    is_flatten = axis is None
    dims = len(x.shape)
    if is_flatten:
        x = paddle.flatten(x)
        axis = 0
    else:
        if not isinstance(axis, int) or not (axis < dims and axis >= -dims):
            raise ValueError(
                "In median, axis should be none or an integer in range [-rank(x), rank(x))."
            )
        if axis < 0:
            axis += dims
    sz = x.shape[axis]
    kth = sz >> 1
    tensor_topk, idx = paddle.topk(x, kth + 1, axis=axis, largest=False)
    dtype = 'float64' if x.dtype == core.VarDesc.VarType.FP64 else 'float32'
    if sz & 1 == 0:
        out_tensor = paddle.slice(
            tensor_topk, axes=[axis], starts=[kth - 1],
            ends=[kth]) + paddle.slice(
                tensor_topk, axes=[axis], starts=[kth], ends=[kth + 1])
        out_tensor = paddle.cast(out_tensor, dtype=dtype) / 2
    else:
        out_tensor = paddle.cast(
            paddle.slice(
                tensor_topk, axes=[axis], starts=[kth], ends=[kth + 1]),
            dtype=dtype)
    if not keepdim or is_flatten:
        if not is_flatten:
            newshape = x.shape[:axis] + x.shape[axis + 1:]
        elif not keepdim:
            newshape = [1]
        else:
            newshape = [1] * dims
    else:
        newshape = out_tensor.shape
    out_tensor = out_tensor.reshape(newshape, name=name)
    return out_tensor


def quantile(x, q, axis=None, keepdim=False):
    """
    Compute the quantile of the input along the specified axis.

    Args:
        x (Tensor): The input Tensor, it's data type can be float32, float64.
        q (int|float|list): The q for calculate quantile, which should be in range [0, 1]. If q is a list, 
            each q will be calculated and the first dimension of output is same to the number of ``q`` .
        axis (int|list, optional): The axis along which to calculate quantile. ``axis`` should be int or list of int.
            ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
            If ``axis`` is less than 0, it works the same way as :math:`axis + D`.
            If ``axis`` is a list, quantile is calculated over all elements of given axises.
            If ``axis`` is None, quantile is calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of quantile along ``axis`` of ``x``. If data type of ``x`` is float64, data type of results will be float64, otherwise data type will be float32.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.randn((2,3))
            #[[-1.28740597,  0.49533170, -1.00698614],
            # [-1.11656201, -1.01010525, -2.23457789]])

            y1 = paddle.quantile(x, q=0.5, axis=[0, 1])
            # y1 = -1.06333363

            y2 = paddle.quantile(x, q=0.5, axis=1)
            # y2 = [-1.00698614, -1.11656201]

            y3 = paddle.quantile(x, q=[0.3, 0.5], axis=1)
            # y3 =[[-1.11915410, -1.56376839],
            #      [-1.00698614, -1.11656201]]

            y4 = paddle.quantile(x, q=0.8, axis=1, keepdim=True)
            # y4 = [[-0.10559537],
            #       [-1.05268800]])
    """
    if not isinstance(x, Variable):
        raise TypeError("input x should be a Tensor.")
    dims = len(x.shape)
    out_shape = x.shape
    if axis is None:
        x = paddle.flatten(x)
        axis = 0
        out_shape = [1] * dims
    else:
        if isinstance(axis, list):
            if (len(axis) <= 0):
                raise ValueError("axis should not be empty")
            axis_src, axis_dst = [], []
            for axis_single in axis:
                if not isinstance(axis_single, int) or not (
                        axis_single < dims and axis_single >= -dims):
                    raise ValueError(
                        "Axis should be None, int, or a list, element should in range [-rank(x), rank(x))."
                    )
                if axis_single < 0:
                    axis_single = axis_single + dims
                axis_src.append(axis_single)
                out_shape[axis_single] = 1
            axis_dst = list(range(-len(axis), 0))
            x = paddle.moveaxis(x, axis_src, axis_dst)
            x = paddle.flatten(x, axis_dst[0], axis_dst[-1])
            axis = axis_dst[0]
        else:
            if not isinstance(axis, int) or not (axis < dims and axis >= -dims):
                raise ValueError(
                    "Axis should be None, int, or a list, element should in range [-rank(x), rank(x))."
                )
            if axis < 0:
                axis += dims
            out_shape[axis] = 1
    indices = []
    if isinstance(q, (int, float)):
        if q < 0 or q > 1:
            raise ValueError("q should be in range [0, 1]")
        indices.append(q * (x.shape[axis] - 1))
    elif isinstance(q, (list, tuple)):
        if len(q) <= 0:
            raise ValueError("q should not be empty")
        for q_num in q:
            if q_num < 0 or q_num > 1:
                raise ValueError("q should be in range [0, 1]")
            indices.append(q_num * (x.shape[axis] - 1))
    else:
        raise TypeError("Type of q should be int, float, list or tuple.")
    indices = paddle.to_tensor(indices).astype(paddle.float32)
    sorted_tensor = paddle.sort(x, axis)
    indices_below = paddle.floor(indices).astype(paddle.int32)
    indices_upper = paddle.ceil(indices).astype(paddle.int32)
    outputs = []

    def expand_dim(indices, sorted_tensor_shape, axis):
        assert axis < len(list(sorted_tensor_shape))
        expanded_shape = [1] * len(list(sorted_tensor_shape))
        expanded_shape[axis] = len(indices)
        expanded_shape = tuple(expanded_shape)
        indices = indices.reshape(expanded_shape)
        return indices

    # TODO(chenjianye): replace the for-loop to directly take elements.
    for i in range(len(indices)):
        if (indices_upper[i] != indices_below[i]):
            tensor_below = paddle.take_along_axis(
                sorted_tensor,
                expand_dim(indices_below[i], sorted_tensor.shape, axis), axis)
            tensor_upper = paddle.take_along_axis(
                sorted_tensor,
                expand_dim(indices_upper[i], sorted_tensor.shape, axis), axis)
            weights = (indices[i] - indices_below[i]).astype(x.dtype)
            out = paddle.lerp(tensor_below, tensor_upper, weights)
        else:
            out = paddle.take_along_axis(
                sorted_tensor,
                expand_dim(indices_below[i], sorted_tensor.shape, axis), axis)
        if not keepdim:
            out = paddle.squeeze(out, axis=axis)
        else:
            out = out.reshape(out_shape)
        outputs.append(out)
    if isinstance(q, (list, tuple)):
        return paddle.stack(outputs, 0)
    else:
        return outputs[0]
