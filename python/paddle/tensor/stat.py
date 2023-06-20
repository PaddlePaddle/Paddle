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

import paddle
from paddle import _C_ops
from paddle.framework import in_dynamic_mode

from ..common_ops_import import Variable
from ..fluid.data_feeder import check_type, check_variable_and_dtype
from ..framework import LayerHelper, core
from .math import _get_reduce_axis_with_tensor
from .search import where

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
            # 12.5
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
    if in_dynamic_mode():
        return _C_ops.mean(x, axis, keepdim)
    else:
        reduce_all, axis = _get_reduce_axis_with_tensor(axis, x)
        check_variable_and_dtype(
            x,
            'x/input',
            ['uint16', 'float16', 'float32', 'float64'],
            'mean/reduce_mean',
        )
        check_type(
            axis, 'axis/dim', (int, list, tuple, Variable), 'mean/reduce_mean'
        )
        if isinstance(axis, (list, tuple)):
            for item in axis:
                check_type(
                    item,
                    'elements of axis/dim',
                    (int, Variable),
                    'mean/reduce_mean',
                )

        helper = LayerHelper('mean', **locals())

        attrs = {'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all}
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='reduce_mean',
            inputs={'X': x},
            outputs={'Out': out},
            attrs=attrs,
        )
        return out


def var(x, axis=None, unbiased=True, keepdim=False, name=None):
    """
    Computes the variance of ``x`` along ``axis`` .

    Args:
        x (Tensor): The input Tensor with data type float16, float32, float64.
        axis (int|list|tuple, optional): The axis along which to perform variance calculations. ``axis`` should be int, list(int) or tuple(int).

            - If ``axis`` is a list/tuple of dimension(s), variance is calculated along all element(s) of ``axis`` . ``axis`` or element(s) of ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
            - If ``axis`` or element(s) of ``axis`` is less than 0, it works the same way as :math:`axis + D` .
            - If ``axis`` is None, variance is calculated over all elements of ``x``. Default is None.

        unbiased (bool, optional): Whether to use the unbiased estimation. If ``unbiased`` is True, the divisor used in the computation is :math:`N - 1`, where :math:`N` represents the number of elements along ``axis`` , otherwise the divisor is :math:`N`. Default is True.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the output Tensor. The result tensor will have one fewer dimension than the input unless keep_dim is true. Default is False.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of variance along ``axis`` of ``x``, with the same data type as ``x``.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
            out1 = paddle.var(x)
            # 2.66666667
            out2 = paddle.var(x, axis=1)
            # [1.         4.33333333]
    """
    if not in_dynamic_mode():
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], 'var'
        )

    u = mean(x, axis, True, name)
    out = paddle.sum(paddle.pow((x - u), 2), axis, keepdim=keepdim, name=name)

    dtype = x.dtype
    n = paddle.cast(paddle.numel(x), paddle.int64) / paddle.cast(
        paddle.numel(out), paddle.int64
    )
    n = n.astype(dtype)
    if unbiased:
        one_const = paddle.ones([], x.dtype)
        n = where(n > one_const, n - 1.0, one_const)
    n.stop_gradient = True
    out /= n
    return out


def std(x, axis=None, unbiased=True, keepdim=False, name=None):
    """
    Computes the standard-deviation of ``x`` along ``axis`` .

    Args:
        x (Tensor): The input Tensor with data type float16, float32, float64.
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
            # 1.63299316
            out2 = paddle.std(x, unbiased=False)
            # 1.49071205
            out3 = paddle.std(x, axis=1)
            # [1.       2.081666]

    """
    if not in_dynamic_mode():
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], 'std'
        )
    out = var(**locals())
    return paddle.sqrt(out)


def numel(x, name=None):
    """
    Returns the number of elements for a tensor, which is a 0-D int64 Tensor with shape [].

    Args:
        x (Tensor): The input Tensor, it's data type can be bool, float16, float32, float64, int32, int64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The number of elements for the input Tensor, whose shape is [].

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.full(shape=[4, 5, 7], fill_value=0, dtype='int32')
            numel = paddle.numel(x) # 140


    """
    if in_dynamic_mode():
        return _C_ops.numel(x)
    else:
        if not isinstance(x, Variable):
            raise TypeError("x must be a Tensor in numel")
        helper = LayerHelper('numel', **locals())
        out = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.INT64
        )
        helper.append_op(type='size', inputs={'Input': x}, outputs={'Out': out})
        return out


def nanmedian(x, axis=None, keepdim=False, name=None):
    r"""
    Compute the median along the specified axis, while ignoring NaNs.

    If the valid count of elements is a even number,
    the average value of both elements in the middle is calculated as the median.

    Args:
        x (Tensor): The input Tensor, it's data type can be int32, int64, float16, float32, float64.
        axis (None|int|list|tuple, optional):
            The axis along which to perform median calculations ``axis`` should be int or list of int.
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
        Tensor, results of median along ``axis`` of ``x``. The output dtype is the same as `x`.

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([[float('nan'), 2. , 3. ], [0. , 1. , 2. ]])

            y1 = x.nanmedian()
            # y1 is 2.

            y2 = x.nanmedian(0)
            # y2 is [0., 1.5, 2.5]

            y3 = x.nanmedian(0, keepdim=True)
            # y3 is [[0.,  1.5, 2.5]]

            y4 = x.nanmedian((0, 1))
            # y4 is 2.
    """
    if not isinstance(x, Variable):
        raise TypeError("In median, the input x should be a Tensor.")

    if isinstance(axis, (list, tuple)) and len(axis) == 0:
        raise ValueError("Axis list should not be empty.")

    if axis is None:
        axis = []
    elif isinstance(axis, tuple):
        axis = list(axis)
    elif isinstance(axis, int):
        axis = [axis]

    if in_dynamic_mode():
        return _C_ops.nanmedian(x, axis, keepdim)
    else:
        check_variable_and_dtype(
            x,
            'X',
            ['int32', 'int64', 'float16', 'float32', 'float64'],
            'nanmedian',
        )

        helper = LayerHelper('nanmedian', **locals())
        attrs = {'axis': axis, 'keepdim': keepdim}
        out = helper.create_variable_for_type_inference(x.dtype)
        medians = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='nanmedian',
            inputs={'X': x},
            outputs={'Out': out, 'MedianIndex': medians},
            attrs=attrs,
        )
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
            # Tensor(shape=[3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
            #        [[0 , 1 , 2 , 3 ],
            #         [4 , 5 , 6 , 7 ],
            #         [8 , 9 , 10, 11]])

            y1 = paddle.median(x)
            # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            #        5.50000000)

            y2 = paddle.median(x, axis=0)
            # Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
            #        [4., 5., 6., 7.])

            y3 = paddle.median(x, axis=1)
            # Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            #        [1.50000000, 5.50000000, 9.50000000])

            y4 = paddle.median(x, axis=0, keepdim=True)
            # Tensor(shape=[1, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            #        [[4., 5., 6., 7.]])

    """
    if not isinstance(x, Variable):
        raise TypeError("In median, the input x should be a Tensor.")

    if x.size == 0:
        raise ValueError("In median, the size of input x should not be 0.")

    is_flatten = False
    dims = len(x.shape)
    if dims == 0:
        assert axis in [
            -1,
            0,
            None,
        ], 'when input 0-D, axis can only be [-1, 0] or default None'
        is_flatten = True

    if axis is None:
        is_flatten = True

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
            tensor_topk, axes=[axis], starts=[kth - 1], ends=[kth]
        ) + paddle.slice(tensor_topk, axes=[axis], starts=[kth], ends=[kth + 1])
        out_tensor = paddle.cast(out_tensor, dtype=dtype) / 2
    else:
        out_tensor = paddle.cast(
            paddle.slice(
                tensor_topk, axes=[axis], starts=[kth], ends=[kth + 1]
            ),
            dtype=dtype,
        )
    out_tensor = out_tensor + paddle.sum(
        paddle.cast(paddle.isnan(x), dtype=dtype) * x, axis=axis, keepdim=True
    )
    if is_flatten:
        if keepdim:
            out_tensor = out_tensor.reshape([1] * dims)
        else:
            out_tensor = out_tensor.reshape([])
    else:
        if not keepdim:
            out_tensor = out_tensor.squeeze(axis)
    return out_tensor


def _compute_quantile(x, q, axis=None, keepdim=False, ignore_nan=False):
    """
    Compute the quantile of the input along the specified axis.

    Args:
        x (Tensor): The input Tensor, it's data type can be float32, float64, int32, int64.
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
        ignore_nan: (bool, optional): Whether to ignore NaN of input Tensor.
            If ``ignore_nan`` is True, it will calculate nanquantile.
            Otherwise it will calculate quantile. Default is False.

    Returns:
        Tensor, results of quantile along ``axis`` of ``x``.
        In order to obtain higher precision, data type of results will be float64.
    """
    # Validate x
    if not isinstance(x, Variable):
        raise TypeError("input x should be a Tensor.")

    # Validate q
    if isinstance(q, (int, float)):
        q = [q]
    elif isinstance(q, (list, tuple)):
        if len(q) <= 0:
            raise ValueError("q should not be empty")
    else:
        raise TypeError("Type of q should be int, float, list or tuple.")

    # Validate axis
    dims = len(x.shape)
    out_shape = list(x.shape)
    if axis is None:
        x = paddle.flatten(x)
        axis = 0
        out_shape = [1] * dims
    else:
        if isinstance(axis, list):
            axis_src, axis_dst = [], []
            for axis_single in axis:
                if not isinstance(axis_single, int) or not (
                    axis_single < dims and axis_single >= -dims
                ):
                    raise ValueError(
                        "Axis should be None, int, or a list, element should in range [-rank(x), rank(x))."
                    )
                if axis_single < 0:
                    axis_single = axis_single + dims
                axis_src.append(axis_single)
                out_shape[axis_single] = 1

            axis_dst = list(range(-len(axis), 0))
            x = paddle.moveaxis(x, axis_src, axis_dst)
            if len(axis_dst) == 0:
                x = paddle.flatten(x)
                axis = 0
            else:
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

    mask = x.isnan()
    valid_counts = mask.logical_not().sum(
        axis=axis, keepdim=True, dtype='float64'
    )

    indices = []

    for q_num in q:
        if q_num < 0 or q_num > 1:
            raise ValueError("q should be in range [0, 1]")
        if in_dynamic_mode():
            q_num = paddle.to_tensor(q_num, dtype='float64')
        if ignore_nan:
            indices.append(q_num * (valid_counts - 1))
        else:
            # TODO: Use paddle.index_fill instead of where
            index = q_num * (valid_counts - 1)
            last_index = x.shape[axis] - 1
            nums = paddle.full_like(index, fill_value=last_index)
            index = paddle.where(mask.any(axis=axis, keepdim=True), nums, index)
            indices.append(index)

    sorted_tensor = paddle.sort(x, axis)

    outputs = []

    # TODO(chenjianye): replace the for-loop to directly take elements.
    for index in indices:
        indices_below = paddle.floor(index).astype(paddle.int32)
        indices_upper = paddle.ceil(index).astype(paddle.int32)
        tensor_upper = paddle.take_along_axis(
            sorted_tensor, indices_upper, axis=axis
        )
        tensor_below = paddle.take_along_axis(
            sorted_tensor, indices_below, axis=axis
        )
        weights = index - indices_below.astype('float64')
        out = paddle.lerp(
            tensor_below.astype('float64'),
            tensor_upper.astype('float64'),
            weights,
        )
        if not keepdim:
            out = paddle.squeeze(out, axis=axis)
        else:
            out = out.reshape(out_shape)
        outputs.append(out)

    if len(q) > 1:
        outputs = paddle.stack(outputs, 0)
    else:
        outputs = outputs[0]

    return outputs


def quantile(x, q, axis=None, keepdim=False):
    """
    Compute the quantile of the input along the specified axis.
    If any values in a reduced row are NaN, then the quantiles for that reduction will be NaN.

    Args:
        x (Tensor): The input Tensor, it's data type can be float32, float64, int32, int64.
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
        Tensor, results of quantile along ``axis`` of ``x``.
        In order to obtain higher precision, data type of results will be float64.

    Examples:
        .. code-block:: python

            import paddle

            y = paddle.arange(0, 8 ,dtype="float32").reshape([4, 2])
            # Tensor(shape=[4, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            #        [[0., 1.],
            #         [2., 3.],
            #         [4., 5.],
            #         [6., 7.]])

            y1 = paddle.quantile(y, q=0.5, axis=[0, 1])
            # Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        3.50000000)

            y2 = paddle.quantile(y, q=0.5, axis=1)
            # Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        [0.50000000, 2.50000000, 4.50000000, 6.50000000])

            y3 = paddle.quantile(y, q=[0.3, 0.5], axis=0)
            # Tensor(shape=[2, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        [[1.80000000, 2.80000000],
            #         [3.        , 4.        ]])

            y[0,0] = float("nan")
            y4 = paddle.quantile(y, q=0.8, axis=1, keepdim=True)
            # Tensor(shape=[4, 1], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        [[nan       ],
            #         [2.80000000],
            #         [4.80000000],
            #         [6.80000000]])

    """
    return _compute_quantile(x, q, axis=axis, keepdim=keepdim, ignore_nan=False)


def nanquantile(x, q, axis=None, keepdim=False):
    """
    Compute the quantile of the input as if NaN values in input did not exist.
    If all values in a reduced row are NaN, then the quantiles for that reduction will be NaN.

    Args:
        x (Tensor): The input Tensor, it's data type can be float32, float64, int32, int64.
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
        Tensor, results of quantile along ``axis`` of ``x``.
        In order to obtain higher precision, data type of results will be float64.

    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor(
                [[0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9]],
                dtype="float32")
            x[0,0] = float("nan")

            y1 = paddle.nanquantile(x, q=0.5, axis=[0, 1])
            # Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        5.)

            y2 = paddle.nanquantile(x, q=0.5, axis=1)
            # Tensor(shape=[2], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        [2.50000000, 7.        ])

            y3 = paddle.nanquantile(x, q=[0.3, 0.5], axis=0)
            # Tensor(shape=[2, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        [[5.        , 2.50000000, 3.50000000, 4.50000000, 5.50000000],
            #         [5.        , 3.50000000, 4.50000000, 5.50000000, 6.50000000]])

            y4 = paddle.nanquantile(x, q=0.8, axis=1, keepdim=True)
            # Tensor(shape=[2, 1], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        [[3.40000000],
            #         [8.20000000]])

            nan = paddle.full(shape=[2, 3], fill_value=float("nan"))
            y5 = paddle.nanquantile(nan, q=0.8, axis=1, keepdim=True)
            # Tensor(shape=[2, 1], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        [[nan],
            #         [nan]])

    """
    return _compute_quantile(x, q, axis=axis, keepdim=keepdim, ignore_nan=True)
