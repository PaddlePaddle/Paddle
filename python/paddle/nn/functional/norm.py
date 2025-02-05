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

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import paddle
from paddle import _C_ops, in_dynamic_mode
from paddle.base.framework import (
    in_dygraph_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)

from ...base.data_feeder import check_type, check_variable_and_dtype
from ...base.layer_helper import LayerHelper

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from paddle import Tensor
    from paddle._typing import (
        DataLayout1D,
        DataLayout2D,
        DataLayout3D,
        DataLayoutND,
    )

__all__ = []


def normalize(
    x: Tensor,
    p: float = 2,
    axis: int = 1,
    epsilon: float = 1e-12,
    name: str | None = None,
) -> Tensor:
    r"""
    Normalize ``x`` along dimension ``axis`` using :math:`L_p` norm. This layer computes

    .. math::

        y = \frac{x}{ \max\left( \lvert \lvert x \rvert \rvert_p, epsilon\right) }

    .. math::
        \lvert \lvert x \rvert \rvert_p = \left( \sum_i {\lvert x_i \rvert^p}  \right)^{1/p}

    where, :math:`\sum_i{\lvert x_i \rvert^p}` is calculated along the ``axis`` dimension.


    Parameters:
        x (Tensor): The input tensor could be N-D tensor, and the input data type could be float32 or float64.
        p (float|int, optional): The exponent value in the norm formulation. Default: 2.
        axis (int, optional): The axis on which to apply normalization. If `axis < 0`, the dimension to normalization is `x.ndim + axis`. -1 is the last dimension.
        epsilon (float, optional): Small float added to denominator to avoid dividing by zero. Default is 1e-12.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the output has the same shape and data type with ``x``.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> paddle.disable_static()
            >>> x = paddle.arange(6, dtype="float32").reshape([2,3])
            >>> y = F.normalize(x)
            >>> print(y)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.        , 0.44721359, 0.89442718],
             [0.42426404, 0.56568539, 0.70710671]])

            >>> y = F.normalize(x, p=1.5)
            >>> print(y)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.        , 0.40862012, 0.81724024],
             [0.35684016, 0.47578689, 0.59473360]])

            >>> y = F.normalize(x, axis=0)
            >>> print(y)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.        , 0.24253564, 0.37139067],
             [1.        , 0.97014254, 0.92847669]])

    """

    if in_dygraph_mode():
        eps = paddle.full(shape=[1], fill_value=epsilon, dtype=x.dtype)
        out = _C_ops.p_norm(x, float(p), axis, epsilon, True, False)
        return x / _C_ops.maximum(out, eps)

    elif in_pir_mode():
        eps = paddle.full(shape=[1], fill_value=epsilon, dtype=x.dtype)
        out = _C_ops.p_norm(x, float(p), axis, epsilon, True, False)
        return paddle.divide(x, _C_ops.maximum(out, eps), name=name)

    else:
        check_type(p, 'p', (float, int), 'normalize')
        check_type(axis, 'axis', (int), 'normalize')
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], 'normalize'
        )
        if len(x.shape) == 1 and axis != 0 and axis != -1:
            raise ValueError(
                f"Axis must be 0 or -1 when x is a 1-D tensor, but received axis = {axis}"
            )

        attrs = {
            'axis': axis,
            'porder': float(p),
            'keepdim': True,
            'epsilon': epsilon,
        }
        helper = LayerHelper('p_norm', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='p_norm', inputs={'X': x}, outputs={'Out': out}, attrs=attrs
        )
        eps = out.block.create_var(dtype=out.dtype)
        eps = paddle.full(shape=[1], fill_value=epsilon, dtype=out.dtype)
        return paddle.divide(x, paddle.maximum(out, eps), name=name)


def batch_norm(
    x,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.9,
    epsilon: float = 1e-05,
    data_format: DataLayoutND = 'NCHW',
    use_global_stats: bool | None = None,
    name: str | None = None,
) -> Tensor:
    """
    Applies Batch Normalization as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .

    nn.functional.batch_norm is used for nn.BatchNorm1D, nn.BatchNorm2D, nn.BatchNorm3D. Please use above API for BatchNorm.

    Parameters:
        x(Tensor): input value. It's data type should be float32, float64.
        running_mean(Tensor): running mean.
        running_var(Tensor): running variance.
        weight(Tensor, optional): The weight tensor of batch_norm. Default: None.
        bias(Tensor, optional): The bias tensor of batch_norm. Default: None.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        training(bool, optional): True means train mode which compute by batch data and track global mean and var during train period. False means inference mode which compute by global mean and var which calculated by train period. Default False.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        data_format(str, optional): Specify the input data format, may be "NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC" or "NDHWC", where `N` is batch size, `C` is the number of the feature map, `D` is the depth of the feature, `H` is the height of the feature map, `W` is the width of the feature map, `L` is the length of the feature map. Default "NCHW".
        use_global_stats(bool|None, optional): Whether to use global mean and variance. If set to False, use the statistics of one mini-batch, if set to True, use the global statistics, if set to None, use global statistics in the test phase and use the statistics of one mini-batch in the training phase. Default: None.
        name(str|None, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.arange(12, dtype="float32").reshape([2, 1, 2, 3])
            >>> print(x)
            Tensor(shape=[2, 1, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0. , 1. , 2. ],
               [3. , 4. , 5. ]]],
             [[[6. , 7. , 8. ],
               [9. , 10., 11.]]]])
            >>> running_mean = paddle.to_tensor([0], dtype="float32")
            >>> running_variance = paddle.to_tensor([1], dtype="float32")
            >>> weight = paddle.to_tensor([2], dtype="float32")
            >>> bias = paddle.to_tensor([1], dtype="float32")

            >>> batch_norm_out = paddle.nn.functional.batch_norm(x, running_mean,
            ...                                             running_variance, weight, bias)
            >>> print(batch_norm_out)
            Tensor(shape=[2, 1, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[1.         , 2.99998999 , 4.99997997 ],
               [6.99996948 , 8.99995995 , 10.99994946]]],
             [[[12.99993896, 14.99992943, 16.99991989],
               [18.99990845, 20.99989891, 22.99988937]]]])

    """
    assert len(x.shape) >= 2, "input dim must be larger than 1"

    # input ad out must share the memory
    mean_out = running_mean
    variance_out = running_var

    true_data_format = ['NC', 'NCL', 'NCHW', 'NCDHW', 'NLC', 'NHWC', 'NDHWC']
    if data_format not in true_data_format:
        raise ValueError(
            "data_format must be one of 'NC', 'NCL', 'NCHW', 'NCDHW', "
            f"'NLC', 'NHWC', 'NDHWC' but receive {data_format}"
        )

    data_format = 'NCHW' if data_format[1] == 'C' else 'NHWC'

    if use_global_stats is None:
        use_global_stats = not training
        trainable_statistics = False
    else:
        trainable_statistics = not use_global_stats

    if in_dynamic_mode():
        batch_norm_out, _, _, _, _, _ = _C_ops.batch_norm(
            x,
            running_mean,
            running_var,
            weight,
            bias,
            not training,
            momentum,
            epsilon,
            data_format,
            use_global_stats,
            trainable_statistics,
        )
        return batch_norm_out

    elif in_pir_mode():
        batch_norm_out, t1, t2, t3, t4, _ = _C_ops.batch_norm_(
            x,
            running_mean,
            running_var,
            weight,
            bias,
            not training,
            momentum,
            epsilon,
            data_format,
            use_global_stats,
            trainable_statistics,
        )
        return batch_norm_out

    else:
        check_variable_and_dtype(
            x, 'input', ['float16', 'uint16', 'float32', 'float64'], 'BatchNorm'
        )

        # for static need dict
        attrs = {
            "momentum": momentum,
            "epsilon": epsilon,
            "is_test": not training,
            "data_layout": data_format,
            "fuse_with_relu": False,
            "use_global_stats": use_global_stats,
            "trainable_statistics": trainable_statistics,
        }

        inputs = {
            "X": [x],
            "Mean": [running_mean],
            "Variance": [running_var],
        }

        if weight:
            inputs['Scale'] = [weight]
        if bias:
            inputs['Bias'] = [bias]
        helper = LayerHelper('batch_norm', **locals())
        from paddle.base.data_feeder import convert_dtype

        param_dtype = (
            x.dtype
            if convert_dtype(x.dtype) not in ['float16', 'uint16']
            else 'float32'
        )
        saved_mean = helper.create_variable_for_type_inference(
            dtype=param_dtype, stop_gradient=True
        )
        saved_variance = helper.create_variable_for_type_inference(
            dtype=param_dtype, stop_gradient=True
        )
        batch_norm_out = helper.create_variable_for_type_inference(x.dtype)

        outputs = {
            "Y": [batch_norm_out],
            "MeanOut": [running_mean],
            "VarianceOut": [running_var],
            "SavedMean": [saved_mean],
            "SavedVariance": [saved_variance],
        }

        if training or trainable_statistics:
            # reserve_space is only used for training.
            reserve_space = helper.create_variable_for_type_inference(
                dtype=x.dtype, stop_gradient=True
            )
            outputs["ReserveSpace"] = [reserve_space]

        helper.append_op(
            type="batch_norm", inputs=inputs, outputs=outputs, attrs=attrs
        )

        return helper.append_activation(batch_norm_out)


def layer_norm(
    x: Tensor,
    normalized_shape: int | Sequence[int],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    epsilon: float = 1e-05,
    name: str | None = None,
) -> Tensor:
    """
    nn.LayerNorm is recommended.
    For more information, please refer to :ref:`api_paddle_nn_LayerNorm` .

    Parameters:
        x(Tensor): Input Tensor. It's data type should be bfloat16, float16, float32, float64.
        normalized_shape(int|list|tuple): Input shape from an expected input of
            size :math:`[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`.
            If it is a single integer, this module will normalize over the last dimension
            which is expected to be of that specific size.
        weight(Tensor, optional): The weight tensor of layer_norm. Default: None.
        bias(Tensor, optional): The bias tensor of layer_norm. Default: None.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        name(str, optional): Name for the LayerNorm, default is None. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        None

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> x = paddle.rand((2, 2, 2, 3))
            >>> layer_norm_out = paddle.nn.functional.layer_norm(x, x.shape[1:])
            >>> print(layer_norm_out)
            Tensor(shape=[2, 2, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[ 0.87799639, -0.32706568, -1.23529339],
               [ 1.01540327, -0.66222906, -0.72354043]],
              [[ 1.24183702,  0.45458138, -0.33506915],
               [ 0.41468468,  1.26852870, -1.98983312]]],
             [[[ 0.02837803,  1.27684665, -0.90110683],
               [-0.94709367, -0.15110941, -1.16546965]],
              [[-0.82010198,  0.11218392, -0.86506516],
               [ 1.09489357,  0.19107464,  2.14656854]]]])

    """
    input_shape = list(x.shape)
    input_ndim = len(input_shape)
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = [normalized_shape]
    elif isinstance(normalized_shape, tuple):
        normalized_shape = list(normalized_shape)
    elif not isinstance(normalized_shape, list):
        raise ValueError(
            "`normalized_shape` should be int, list of ints or tuple of ints."
        )

    normalized_ndim = len(normalized_shape)
    begin_norm_axis = input_ndim - normalized_ndim
    if input_ndim < normalized_ndim or (
        not paddle.utils.is_same_shape(
            input_shape[begin_norm_axis:], normalized_shape
        )
    ):
        str_normalized_shape = str(normalized_shape)
        raise ValueError(
            'Given normalized_shape is '
            + str_normalized_shape
            + ', expected input with shape [*, '
            + str_normalized_shape[1:]
            + ', but got input shape '
            + str(input_shape)
        )

    if in_dynamic_or_pir_mode():
        out = _C_ops.layer_norm(x, weight, bias, epsilon, begin_norm_axis)
        return out

    else:
        check_variable_and_dtype(
            x, 'input', ['uint16', 'float16', 'float32', 'float64'], 'LayerNorm'
        )

        inputs = {}
        inputs['X'] = [x]
        if weight:
            inputs['Scale'] = [weight]
        if bias:
            inputs['Bias'] = [bias]
        attrs = {"epsilon": epsilon, "begin_norm_axis": begin_norm_axis}

        # create output
        helper = LayerHelper('layer_norm', **locals())
        from paddle.base.data_feeder import convert_dtype

        param_dtype = (
            x.dtype if convert_dtype(x.dtype) != 'float16' else 'float32'
        )
        mean_out = helper.create_variable_for_type_inference(
            dtype=param_dtype, stop_gradient=True
        )
        variance_out = helper.create_variable_for_type_inference(
            dtype=param_dtype, stop_gradient=True
        )
        layer_norm_out = helper.create_variable_for_type_inference(x.dtype)

        helper.append_op(
            type="layer_norm",
            inputs=inputs,
            outputs={
                "Y": layer_norm_out,
                "Mean": mean_out,
                "Variance": variance_out,
            },
            attrs={"epsilon": epsilon, "begin_norm_axis": begin_norm_axis},
        )

        return helper.append_activation(layer_norm_out)


def instance_norm(
    x: Tensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.9,
    eps: float = 1e-05,
    data_format: Literal['NC', 'NCL', 'NCHW', 'NCDHW'] = 'NCHW',
    name: str | None = None,
) -> Tensor:
    """
    It is recommended to use :ref:`api_paddle_nn_InstanceNorm1D` , :ref:`api_paddle_nn_InstanceNorm2D` , :ref:`api_paddle_nn_InstanceNorm3D` to call this method internally.

    Parameters:
        x (Tensor): Input Tensor. It's data type should be float32, float64.
        running_mean (Tensor, optional): running mean. Default None. Obsolete (that is, no longer usable).
        running_var (Tensor, optional): running variance. Default None. Obsolete (that is, no longer usable).
        weight (Tensor, optional): The weight tensor of instance_norm. Default: None.
            If its value is None, this parameter will be initialized by one.
        bias (Tensor, optional): The bias tensor of instance_norm. Default: None.
            If its value is None, this parameter will be initialized by zero.
        eps (float, optional): A value added to the denominator for numerical stability. Default is 1e-5.
        momentum (float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        use_input_stats (bool, optional): Default True. Obsolete (that is, no longer usable).
        data_format (str, optional): Specify the input data format, may be "NC", "NCL", "NCHW" or "NCDHW". Default "NCHW".
        name (str|None, optional): Name for the InstanceNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Returns:
        None.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> x = paddle.rand((2, 2, 2, 3))
            >>> instance_norm_out = paddle.nn.functional.instance_norm(x)

            >>> print(instance_norm_out)
            Tensor(shape=[2, 2, 2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[ 1.25768495, -0.18054862, -1.26451230],
               [ 1.42167914, -0.58056390, -0.65373862]],
              [[ 0.95882601,  0.25075224, -0.45947552],
               [ 0.21486834,  0.98283297, -1.94780385]]],
             [[[ 0.40697321,  1.90885782, -0.71117985],
               [-0.76650119,  0.19105314, -1.02920341]],
              [[-1.06926346, -0.18710862, -1.11180890],
               [ 0.74275863, -0.11246002,  1.73788261]]]])

    """
    if in_dynamic_or_pir_mode():
        out = _C_ops.instance_norm(x, weight, bias, eps)
        return out
    else:
        check_variable_and_dtype(
            x,
            'input',
            ['float32', 'float64', 'float16', 'uint16'],
            "InstanceNorm",
        )

        attrs = {
            "epsilon": eps,
            "momentum": momentum,
            "data_format": data_format,
        }

        if weight and bias:
            inputs = {"X": [x], "Scale": [weight], "Bias": [bias]}
        else:
            inputs = {"X": [x]}

        helper = LayerHelper('instance_norm', **locals())
        saved_mean = helper.create_variable_for_type_inference(
            dtype=x.dtype, stop_gradient=True
        )
        saved_variance = helper.create_variable_for_type_inference(
            dtype=x.dtype, stop_gradient=True
        )
        instance_norm_out = helper.create_variable_for_type_inference(x.dtype)

        outputs = {
            "Y": [instance_norm_out],
            "SavedMean": [saved_mean],
            "SavedVariance": [saved_variance],
        }

        helper.append_op(
            type="instance_norm", inputs=inputs, outputs=outputs, attrs=attrs
        )
        return instance_norm_out


def local_response_norm(
    x: Tensor,
    size: int,
    alpha: float = 0.0001,
    beta: float = 0.75,
    k: float = 1.0,
    data_format: DataLayout1D | DataLayout2D | DataLayout3D = 'NCHW',
    name: str | None = None,
) -> Tensor:
    r"""
    Local Response Normalization performs a type of "lateral inhibition" by normalizing over local input regions.
    For more information, please refer to `ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

    The formula is as follows:

    .. math::

        Output(i, x, y) = Input(i, x, y) / \left(k + \alpha \sum\limits^{\min(C-1, i + size/2)}_{j = \max(0, i - size/2)}(Input(j, x, y))^2\right)^{\beta}

    In the above equation:

    - :math:`size` : The number of channels to sum over.
    - :math:`k` : The offset (avoid being divided by 0).
    - :math:`\\alpha` : The scaling parameter.
    - :math:`\\beta` : The exponent parameter.


    Args:
        x (Tensor): The input 3-D/4-D/5-D tensor. The data type is float16 or float32.
        size (int): The number of channels to sum over.
        alpha (float, optional): The scaling parameter, positive. Default:1e-4
        beta (float, optional): The exponent, positive. Default:0.75
        k (float, optional): An offset, positive. Default: 1.0
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:
            If x is 3-D Tensor, the string could be `"NCL"` or `"NLC"` . When it is `"NCL"`,
            the data is stored in the order of: `[batch_size, input_channels, feature_length]`.
            If x is 4-D Tensor, the string could be  `"NCHW"`, `"NHWC"`. When it is `"NCHW"`,
            the data is stored in the order of: `[batch_size, input_channels, input_height, input_width]`.
            If x is 5-D Tensor, the string could be  `"NCDHW"`, `"NDHWC"` . When it is `"NCDHW"`,
            the data is stored in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name (str|None, optional): Name for the operation (optional, default is None). For more information,
            please refer to :ref:`api_guide_Name`.

    Returns:
        A tensor storing the transformation result with the same shape and data type as input.


    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.rand(shape=(3, 3, 112, 112), dtype="float32")
            >>> y = paddle.nn.functional.local_response_norm(x, size=5)
            >>> print(y.shape)
            [3, 3, 112, 112]

    """
    if not in_dynamic_mode():
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32'], 'local_response_norm'
        )
    if data_format not in ['NCL', 'NLC', 'NCHW', 'NHWC', 'NCDHW', 'NDHWC']:
        raise ValueError(
            "data_format should be in one of [NCL, NCHW, NCDHW, NLC, NHWC, NDHWC], "
            f"but got {data_format}"
        )

    sizes = x.shape
    dim = len(sizes)
    if dim < 3:
        raise ValueError(
            f'Expected 3D or higher dimensionality input, but got {dim} dimensions'
        )

    for i, sz in enumerate(sizes):
        if not sz > 0 and i > 0:
            raise ValueError(
                "Expected every dim's size to be larger than 0, "
                f"but the size of the {i}-th dim is {sz}"
            )

    channel_last = True if data_format[-1] == "C" else False

    from functools import reduce

    sum_sizes = reduce(lambda x, y: x * y, sizes[1:], 1)

    div = paddle.unsqueeze(paddle.multiply(x, x), axis=1)
    if not channel_last:
        pad4d_shape = [0, 0, size // 2, (size - 1) // 2]
        pool2d_shape = (size, 1)
        reshape_shape = [
            sizes[0],
            1,
            sizes[1],
            sizes[2],
            int(sum_sizes / (sizes[1] * sizes[2])),
        ]
        pad5d_shape = [0, 0, 0, 0, size // 2, (size - 1) // 2]
        pool3d_shape = (size, 1, 1)
    else:
        pad4d_shape = [size // 2, (size - 1) // 2, 0, 0]
        pool2d_shape = (1, size)
        reshape_shape = [
            sizes[0],
            1,
            sizes[1],
            int(sum_sizes / (sizes[1] * sizes[-1])),
            sizes[-1],
        ]
        pad5d_shape = [size // 2, (size - 1) // 2, 0, 0, 0, 0]
        pool3d_shape = (1, 1, size)

    if dim == 3:
        div = paddle.nn.functional.pad(div, pad=pad4d_shape)
        div = paddle.nn.functional.avg_pool2d(
            div, kernel_size=pool2d_shape, stride=1
        )
        div = paddle.squeeze(div, axis=1)
    else:
        div = paddle.reshape(div, shape=reshape_shape)
        div = paddle.nn.functional.pad(
            div, pad=pad5d_shape, data_format='NCDHW'
        )
        div = paddle.nn.functional.avg_pool3d(
            div, kernel_size=pool3d_shape, stride=1
        )
        div = paddle.reshape(paddle.squeeze(div, axis=1), sizes)

    div = paddle.scale(div, scale=alpha, bias=k)
    div = paddle.pow(div, beta)
    res = paddle.divide(x, div, name=name)
    return res


def group_norm(
    x: Tensor,
    num_groups: int,
    epsilon: float = 1e-05,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    data_format: DataLayout1D | DataLayout2D | DataLayout3D = 'NCHW',
    name: str | None = None,
) -> Tensor:
    """
    nn.GroupNorm is recommended.
    For more information, please refer to :ref:`api_paddle_nn_GroupNorm` .

    Parameters:
        x(Tensor): Input Tensor with shape: attr:`(batch, num_features, *)`.
        num_groups(int): The number of groups that divided from channels.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        weight(Tensor, optional): The weight Tensor of group_norm, with shape: attr:`[num_channels]`.
            Default: None.
        bias(Tensor, optional): The bias Tensor of group_norm, with shape: attr:`[num_channels]`.
            Default: None.
        data_format(str, optional): Specify the input data format. Support "NCL", "NCHW", "NCDHW", "NLC", "NHWC" or "NDHWC". Default: "NCHW".
        name(str|None, optional): Name for the GroupNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Returns:
        Tensor, the output has the same shape with ``x``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(100)
            >>> x = paddle.arange(48, dtype="float32").reshape((2, 6, 2, 2))
            >>> group_norm_out = paddle.nn.functional.group_norm(x, num_groups=6)

            >>> print(group_norm_out)
            Tensor(shape=[2, 6, 2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[-1.34163547, -0.44721183],
               [ 0.44721183,  1.34163547]],
              [[-1.34163547, -0.44721183],
               [ 0.44721183,  1.34163547]],
              [[-1.34163547, -0.44721183],
               [ 0.44721183,  1.34163547]],
              [[-1.34163547, -0.44721183],
               [ 0.44721183,  1.34163547]],
              [[-1.34163547, -0.44721183],
               [ 0.44721183,  1.34163547]],
              [[-1.34163547, -0.44721183],
               [ 0.44721183,  1.34163547]]],
             [[[-1.34163547, -0.44721183],
               [ 0.44721183,  1.34163547]],
              [[-1.34163547, -0.44721183],
               [ 0.44721183,  1.34163547]],
              [[-1.34163547, -0.44721183],
               [ 0.44721183,  1.34163547]],
              [[-1.34163547, -0.44721183],
               [ 0.44721183,  1.34163547]],
              [[-1.34163547, -0.44721183],
               [ 0.44721183,  1.34163547]],
              [[-1.34163547, -0.44721183],
               [ 0.44721183,  1.34163547]]]])
    """
    if data_format not in ['NCL', 'NCHW', 'NCDHW', 'NLC', 'NHWC', 'NDHWC']:
        raise ValueError("unsupported data layout:" + data_format)

    data_format = 'NCHW' if data_format[1] == 'C' else 'NHWC'

    if in_dynamic_or_pir_mode():
        return _C_ops.group_norm(
            x,
            weight,
            bias,
            epsilon,
            num_groups,
            data_format,
        )
    else:
        helper = LayerHelper('group_norm', **locals())
        mean_out = helper.create_variable_for_type_inference(
            dtype=x.dtype, stop_gradient=True
        )
        variance_out = helper.create_variable_for_type_inference(
            dtype=x.dtype, stop_gradient=True
        )

        inputs = {'X': x}
        if bias is not None:
            inputs['Bias'] = bias
        if weight is not None:
            inputs['Scale'] = weight

        # create output
        group_norm_out = helper.create_variable_for_type_inference(
            dtype=x.dtype
        )

        helper.append_op(
            type="group_norm",
            inputs=inputs,
            outputs={
                "Y": group_norm_out,
                "Mean": mean_out,
                "Variance": variance_out,
            },
            attrs={
                "epsilon": epsilon,
                "groups": num_groups,
                "data_layout": data_format,
            },
        )

        return helper.append_activation(group_norm_out)
