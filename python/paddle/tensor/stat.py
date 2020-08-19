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
from ..fluid.layers import reduce_mean  #DEFINE_ALIAS

__all__ = ['mean', 'reduce_mean', 'std', 'var']

import numpy as np
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import core, in_dygraph_mode
from ..fluid import layers
from .search import where
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
import paddle


def mean(x, axis=None, keepdim=False, name=None):
    """
    Computes the mean of the input tensor's elements along ``axis``.

    Args:
        x (Tensor): The input Tensor with data type float32, float64, int32,
            int64.
        axis (int|list|tuple, optional): The axis along which to perform mean
            calculations. ``axis`` should be int, list(int) or tuple(int). If
            ``axis`` is a list/tuple of dimension(s), mean is calculated along
            all element(s) of ``axis`` . ``axis`` or element(s) of ``axis``
            should be in range [-D, D), where D is the dimensions of ``x`` . If
            ``axis`` or element(s) of ``axis`` is less than 0, it works the
            same way as :math:`axis + D` . If ``axis`` is None, mean is
            calculated along all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keep_dim`` is True, the dimensions of
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
            import numpy as np

            paddle.disable_static()

            x = np.array([[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]],
                          [[13, 14, 15, 16],
                           [17, 18, 19, 20],
                           [21, 22, 23, 24]]], 'float32')
            x = paddle.to_variable(x)
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
        return core.ops.reduce_mean(x, 'dim', axis, 'keep_dim', keepdim,
                                    'reduce_all', reduce_all)

    check_variable_and_dtype(x, 'x/input',
                             ['float32', 'float64', 'int32', 'int64'],
                             'mean/reduce_mean')

    helper = LayerHelper('mean', **locals())
    attrs = {'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all}
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='reduce_mean', inputs={'X': x}, outputs={'Out': out}, attrs=attrs)
    return out


def var(input, axis=None, keepdim=False, unbiased=True, out=None, name=None):
    """
	:alias_main: paddle.var
	:alias: paddle.var,paddle.tensor.var,paddle.tensor.stat.var

    Computes the variance of the input Variable's elements along the specified 
    axis.

    Args:
        input (Variable): The input Variable to be computed variance, with data 
            type float32 and float64 supported.
        axis (list|int, optional): The axis along which the variance is computed. 
            If `None`, compute the variance over all elements of :attr:`input`
            and return a Variable with a single element, otherwise it must be in 
            the range :math:`[-rank(input), rank(input))`. If :math:`axis[i] < 0`, 
            the axis to compute is :math:`rank(input) + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimensions in 
            the output Variable. The dimensions in :attr:`axis` will be squeezed 
            and the result Variable will have :attr:`len(axis)` fewer dimensions 
            than the :attr:`input` unless :attr:`keepdim` is true, default False.
        unbiased (bool, optional): Whether to compute variance via the unbiased 
            estimator, in which the divisor used in the computation is 
            :math:`N - 1`, where :math:`N` represents the number of elements 
            along :attr:`axis`, otherwise the divisor is :math:`N`. Default True.
        out (Variable, optional): Alternate output Variable to store the result
            variance. Default None.
        name (str, optional): The name for this layer. Normally there is no 
            need for user to set this property.  For more information, please 
            refer to :ref:`api_guide_Name`. Default None.

    Returns:
        Variable: The result variance with the same dtype as :attr:`input`. 
            If :attr:`out = None`, returns a new Variable containing the 
            variance, otherwise returns a reference to the output Variable.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.fluid.dygraph as dg

            a = np.array([[1.0, 2.0], [3.0, 4.0]]).astype("float32")
            with dg.guard():
                data = dg.to_variable(a)
                variance = paddle.var(data, axis=[1])
                print(variance.numpy())   
                # [0.5 0.5]
    """
    dtype = convert_dtype(input.dtype)
    if dtype not in ["float32", "float64"]:
        raise ValueError("Layer tensor.var() only supports floating-point "
                         "dtypes, but received {}.".format(dtype))
    rank = len(input.shape)
    axes = axis if axis != None and axis != [] else range(rank)
    axes = [e if e >= 0 else e + rank for e in axes]
    inp_shape = input.shape if in_dygraph_mode() else layers.shape(input)
    mean = layers.reduce_mean(input, dim=axis, keep_dim=True, name=name)
    tmp = layers.reduce_mean(
        (input - mean)**2, dim=axis, keep_dim=keepdim, name=name)

    if unbiased:
        n = 1
        for i in axes:
            n *= inp_shape[i]
        if not in_dygraph_mode():
            n = layers.cast(n, dtype)
            zero_const = layers.fill_constant(shape=[1], dtype=dtype, value=0.0)
            factor = where(n > 1.0, n / (n - 1.0), zero_const)
        else:
            factor = n / (n - 1.0) if n > 1.0 else 0.0
        tmp *= factor
    if out:
        layers.assign(input=tmp, output=out)
        return out
    else:
        return tmp


def std(input, axis=None, keepdim=False, unbiased=True, out=None, name=None):
    """
	:alias_main: paddle.std
	:alias: paddle.std,paddle.tensor.std,paddle.tensor.stat.std

    Computes the standard-deviation  of the input Variable's elements along the specified 
    axis.

    Args:
        input (Variable): The input Variable to be computed standard-deviation, with data 
            type float32 and float64 supported.
        axis (list|int, optional): The axis along which the standard-deviation is computed. 
            If `None`, compute the standard-deviation over all elements of :attr:`input`
            and return a Variable with a single element, otherwise it must be in 
            the range :math:`[-rank(input), rank(input))`. If :math:`axis[i] < 0`, 
            the axis to compute is :math:`rank(input) + axis[i]`.
        keepdim (bool, optional): Whether to reserve the reduced dimensions in 
            the output Variable. The dimensions in :attr:`axis` will be squeezed 
            and the result Variable will have :attr:`len(axis)` fewer dimensions 
            than the :attr:`input` unless :attr:`keepdim` is true, default False.
        unbiased (bool, optional): Whether to compute standard-deviation via the unbiased 
            estimator, in which the divisor used in the computation is 
            :math:`N - 1`, where :math:`N` represents the number of elements 
            along :attr:`axis`, otherwise the divisor is :math:`N`. Default True.
        out (Variable, optional): Alternate output Variable to store the result
            standard-deviation . Default None.
        name (str, optional): The name for this layer. Normally there is no 
            need for user to set this property.  For more information, please 
            refer to :ref:`api_guide_Name`. Default None.

    Returns:
        Variable: The result standard-deviation  with the same dtype as :attr:`input`. 
            If :attr:`out = None`, returns a new Variable containing the 
            standard-deviation , otherwise returns a reference to the output Variable.
    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            paddle.std(x)  # [0.28252685] 
            paddle.std(x, axis=[0])  # [0.0707107, 0.07071075, 0.07071064, 0.1414217]
            paddle.std(x, axis=[-1])  # [0.30956957, 0.29439208] 
    """
    check_variable_and_dtype(input, 'input', ['float32', 'float64'], 'std')

    tmp = var(input, axis=axis, keepdim=keepdim, unbiased=unbiased, name=name)
    tmp = layers.sqrt(tmp)
    if out is not None:
        layers.assign(input=tmp, output=out)
        return out
    else:
        return tmp
