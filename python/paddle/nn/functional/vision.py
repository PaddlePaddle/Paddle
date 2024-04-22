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

import paddle
from paddle import _C_ops, _legacy_C_ops, in_dynamic_mode
from paddle.base.framework import (
    in_dygraph_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)

from ...base.data_feeder import check_variable_and_dtype
from ...base.layer_helper import LayerHelper
from ...common_ops_import import Variable
from ...device import get_cudnn_version, is_compiled_with_rocm

__all__ = []


def affine_grid(theta, out_shape, align_corners=True, name=None):
    """
    It generates a grid of (x,y) or (x,y,z) coordinates using the parameters of
    the affine transformation that correspond to a set of points where
    the input feature map should be sampled to produce the transformed
    output feature map.

    Args:
        theta (Tensor): A tensor with shape [N, 2, 3] or [N, 3, 4]. It contains a batch of affine transform parameters.
                           The data type can be float32 or float64.
        out_shape (Tensor | list | tuple): Type can be a 1-D Tensor, list, or tuple. It is used to represent the shape of the output in an affine transformation, in the format ``[N, C, H, W]`` or ``[N, C, D, H, W]``.
                                           When the format is ``[N, C, H, W]``, it represents the batch size, number of channels, height and width. When the format is ``[N, C, D, H, W]``, it represents the batch size, number of channels, depth, height and width.
                                           The data type must be int32.
        align_corners(bool, optional): if True, aligns the centers of the 4 (4D) or 8 (5D) corner pixels of the input and output tensors, and preserves the value of the corner pixels. Default: True
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A Tensor with shape [batch_size, H, W, 2] or [batch, D, H, W, 3] while ('D')'H', 'W' are the (depth)height, width of feature map in affine transformation. The data type is the same as `theta`.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F
            >>> # theta.shape = [1, 2, 3]
            >>> theta = paddle.to_tensor([[[-0.7, -0.4, 0.3],
            ...                            [ 0.6,  0.5, 1.5]]], dtype="float32")
            >>> y_t = F.affine_grid(
            ...     theta,
            ...     [1, 2, 3, 3],
            ...     align_corners=False
            ... )
            >>> print(y_t)
            Tensor(shape=[1, 3, 3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[ 1.03333330,  0.76666665],
               [ 0.56666672,  1.16666663],
               [ 0.10000002,  1.56666672]],
              [[ 0.76666665,  1.09999990],
               [ 0.30000001,  1.50000000],
               [-0.16666666,  1.90000010]],
              [[ 0.50000000,  1.43333328],
               [ 0.03333333,  1.83333337],
               [-0.43333334,  2.23333335]]]])
    """
    if not isinstance(theta, (Variable, paddle.pir.Value)):
        raise TypeError("The theta should be a Tensor.")

    cudnn_version = get_cudnn_version()
    if cudnn_version is not None and cudnn_version >= 6000 and align_corners:
        use_cudnn = True
    else:
        use_cudnn = False
    if theta.shape[1] == 3:
        use_cudnn = False
    if is_compiled_with_rocm():
        use_cudnn = (
            False  # ROCM platform do not have MIOPEN kernel for affine_grid
        )

    if in_dynamic_mode():
        _out_shape = (
            out_shape.tolist() if isinstance(out_shape, Variable) else out_shape
        )
        theta = theta._use_gpudnn(use_cudnn)
        return _C_ops.affine_grid(theta, _out_shape, align_corners)
    elif in_pir_mode():
        return _C_ops.affine_grid(
            theta,
            out_shape,
            align_corners,
        )
    else:
        helper = LayerHelper('affine_grid', **locals())
        check_variable_and_dtype(
            theta, 'theta', ['float32', 'float64'], 'affine_grid'
        )
        out = helper.create_variable_for_type_inference(dtype=theta.dtype)
        ipts = {'Theta': theta}
        attrs = {"align_corners": align_corners, "use_cudnn": use_cudnn}
        if isinstance(out_shape, Variable):
            ipts['OutputShape'] = out_shape
            check_variable_and_dtype(
                out_shape, 'out_shape', ['int32'], 'affine_grid'
            )
        else:
            attrs['output_shape'] = out_shape

        helper.append_op(
            type='affine_grid',
            inputs=ipts,
            outputs={'Output': out},
            attrs=None if len(attrs) == 0 else attrs,
        )
        return out


def grid_sample(
    x,
    grid,
    mode='bilinear',
    padding_mode='zeros',
    align_corners=True,
    name=None,
):
    """
    Sample input X by using bilinear interpolation or
    nearest interpolation based on flow field grid, which is usually
    generated by :code:`affine_grid` . When the input X is 4-D Tensor,
    the grid of shape [N, H, W, 2] is the concatenation of (x, y)
    coordinates with shape [N, H, W] each, where x is indexing the 4th
    dimension (in width dimension) of input data x and y is indexing
    the 3rd dimension (in height dimension), finally results is the
    bilinear interpolation or nearest value of 4 nearest corner
    points. The output tensor shape will be [N, C, H, W]. When the input X
    is 5-D Tensor, the grid of shape [N, D, H, W, 3] is the concatenation
    of (x, y, z) coordinates with shape [N, D, H, W] each, where x is
    indexing the 5th dimension (in width dimension) of input data x, y is
    indexing the 4th dimension (in height dimension) and z is indexing the
    3rd dimension (in depth dimension) finally results is the bilinear
    interpolation or nearest value of 8 nearest corner points. The output
    tensor shape will be [N, C, D, H, W].



    Step 1:

    Get (x, y) grid coordinates and scale to [0, H-1/W-1].

    .. code-block:: text

        grid_x = 0.5 * (grid[:, :, :, 0] + 1) * (W - 1)
        grid_y = 0.5 * (grid[:, :, :, 1] + 1) * (H - 1)

    Step 2:

    Indices input data X with grid (x, y) in each [H, W] area, and bilinear
    interpolate point value by 4 nearest points or nearest interpolate point value
    by nearest point.

    .. code-block:: text

        wn ------- y_n ------- en
        |           |           |
        |          d_n          |
        |           |           |
        x_w --d_w-- grid--d_e-- x_e
        |           |           |
        |          d_s          |
        |           |           |
        ws ------- y_s ------- wn

        For bilinear interpolation:
        x_w = floor(x)              // west side x coord
        x_e = x_w + 1               // east side x coord
        y_n = floor(y)              // north side y coord
        y_s = y_s + 1               // south side y coord
        d_w = grid_x - x_w          // distance to west side
        d_e = x_e - grid_x          // distance to east side
        d_n = grid_y - y_n          // distance to north side
        d_s = y_s - grid_y          // distance to south side
        wn = X[:, :, y_n, x_w]      // north-west point value
        en = X[:, :, y_n, x_e]      // north-east point value
        ws = X[:, :, y_s, x_w]      // south-east point value
        es = X[:, :, y_s, x_w]      // north-east point value

        output = wn * d_e * d_s + en * d_w * d_s
                + ws * d_e * d_n + es * d_w * d_n

    Args:
        x(Tensor): The input tensor, which is a 4-d tensor with shape
                     [N, C, H, W] or a 5-d tensor with shape [N, C, D, H, W],
                     N is the batch size, C is the channel number,
                     D, H and W is the feature depth, height and width.
                     The data type is float32 or float64.
        grid(Tensor): Input grid tensor, which is a 4-d tensor with shape [N, grid_H,
                        grid_W, 2] or a 5-d tensor with shape [N, grid_D, grid_H,
                        grid_W, 3]. The data type is float32 or float64.
        mode(str, optional): The interpolation method which can be 'bilinear' or 'nearest'.
                         Default: 'bilinear'.
        padding_mode(str, optional) The padding method used when source index
                   is out of input images. It can be 'zeros', 'reflection' and 'border'.
                   Default: zeros.
        align_corners(bool, optional): If `align_corners` is true, it will projects
                   -1 and 1 to the centers of the corner pixels. Otherwise, it will
                   projects -1 and 1 to the image edges.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:

        Tensor, The shape of output is [N, C, grid_H, grid_W] or [N, C, grid_D, grid_H, grid_W] in which `grid_D` is the depth of grid,
                `grid_H` is the height of grid and `grid_W` is the width of grid. The data type is same as input tensor.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> # x shape=[1, 1, 3, 3]
            >>> x = paddle.to_tensor([[[[-0.6,  0.8, -0.5],
            ...                         [-0.5,  0.2,  1.2],
            ...                         [ 1.4,  0.3, -0.2]]]], dtype='float64')
            >>> # grid.shape = [1, 3, 4, 2]
            >>> grid = paddle.to_tensor([[[[ 0.2,  0.3],
            ...                            [-0.4, -0.3],
            ...                            [-0.9,  0.3],
            ...                            [-0.9, -0.6]],
            ...                           [[ 0.4,  0.1],
            ...                            [ 0.9, -0.8],
            ...                            [ 0.4,  0.5],
            ...                            [ 0.5, -0.2]],
            ...                           [[ 0.1, -0.8],
            ...                            [-0.3, -1. ],
            ...                            [ 0.7,  0.4],
            ...                            [ 0.2,  0.8]]]], dtype='float64')
            >>> y_t = F.grid_sample(
            ...     x,
            ...     grid,
            ...     mode='bilinear',
            ...     padding_mode='border',
            ...     align_corners=True
            ... )
            >>> print(y_t)
            Tensor(shape=[1, 1, 3, 4], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[[[ 0.34000000,  0.01600000,  0.08600000, -0.44800000],
               [ 0.55000000, -0.07600000,  0.35000000,  0.59000000],
               [ 0.59600000,  0.38000000,  0.52000000,  0.24000000]]]])
    """

    _modes = ['bilinear', 'nearest']
    _padding_modes = ['zeros', 'reflection', 'border']
    if mode not in _modes:
        raise ValueError(
            f"The mode of grid sample function should be in {_modes}, but got: {mode}"
        )
    if padding_mode not in _padding_modes:
        raise ValueError(
            f"The padding mode of grid sample function should be in {_padding_modes}, but got: {padding_mode}"
        )

    if not isinstance(align_corners, bool):
        raise ValueError(
            f"The align corners should be bool, but got: {align_corners}"
        )

    cudnn_version = get_cudnn_version()
    use_cudnn = False
    if (
        not is_compiled_with_rocm()
        and (cudnn_version is not None)
        and align_corners
        and mode == 'bilinear'
        and padding_mode == 'zeros'
    ):
        use_cudnn = True
        # CUDNN always computes gradients for all inputs
        x.stop_gradient = False
        grid.stop_gradient = False

    if len(grid.shape) == 5:
        use_cudnn = False

    if in_dynamic_or_pir_mode():
        return _C_ops.grid_sample(x, grid, mode, padding_mode, align_corners)
    else:
        helper = LayerHelper("grid_sample", **locals())
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'grid_sample')
        check_variable_and_dtype(
            grid, 'grid', ['float32', 'float64'], 'grid_sample'
        )
        ipts = {'X': x, 'Grid': grid}
        attrs = {
            'mode': mode,
            'padding_mode': padding_mode,
            'align_corners': align_corners,
            'use_cudnn': use_cudnn,
        }
        out = helper.create_variable_for_type_inference(x.dtype)
        helper.append_op(
            type='grid_sampler',
            inputs=ipts,
            attrs=attrs,
            outputs={'Output': out},
        )
    return out


def pixel_shuffle(x, upscale_factor, data_format="NCHW", name=None):
    """
    This API implements pixel shuffle operation.
    See more details in :ref:`PixelShuffle <api_paddle_nn_PixelShuffle>` .


    Parameters:
        x(Tensor): 4-D tensor, the data type should be float32 or float64.
        upscale_factor(int): factor to increase spatial resolution.
        data_format (str, optional): The data format of the input and output data. An optional string from: ``"NCHW"``, ``"NHWC"``. When it is ``"NCHW"``, the data is stored in the order of: [batch_size, input_channels, input_height, input_width]. Default: ``"NCHW"``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Out(tensor): Reshaped tensor according to the new dimension.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.randn(shape=[2,9,4,4])
            >>> out_var = F.pixel_shuffle(x, 3)
            >>> print(out_var.shape)
            [2, 1, 12, 12]
    """
    if not isinstance(upscale_factor, int):
        raise TypeError("upscale factor must be int type")

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'."
            f"But receive Attr(data_format): {data_format} "
        )
    if in_dynamic_or_pir_mode():
        return _C_ops.pixel_shuffle(x, upscale_factor, data_format)
    else:
        helper = LayerHelper("pixel_shuffle", **locals())
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], 'pixel_shuffle'
        )
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type="pixel_shuffle",
            inputs={"X": x},
            outputs={"Out": out},
            attrs={
                "upscale_factor": upscale_factor,
                "data_format": data_format,
            },
        )
        return out


def pixel_unshuffle(x, downscale_factor, data_format="NCHW", name=None):
    """
    This API implements pixel unshuffle operation.
    See more details in :ref:`PixelUnShuffle <api_paddle_nn_PixelUnshuffle>` .

    Parameters:
        x (Tensor): 4-D tensor, the data type should be float32 or float64.
        downscale_factor (int): Factor to decrease spatial resolution.
        data_format (str, optional): The data format of the input and output data. An optional string of ``'NCHW'`` or ``'NHWC'``. When it is ``'NCHW'``, the data is stored in the order of [batch_size, input_channels, input_height, input_width]. Default: ``'NCHW'``.
        name (str, optional): Name for the operation (optional, default is None). Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Out (Tensor): Reshaped tensor according to the new dimension.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F
            >>> x = paddle.randn([2, 1, 12, 12])
            >>> out = F.pixel_unshuffle(x, 3)
            >>> print(out.shape)
            [2, 9, 4, 4]
    """
    if len(x.shape) != 4:
        raise ValueError(
            f"Input x should be 4D tensor, but received x with the shape of {x.shape}"
        )

    if not isinstance(downscale_factor, int):
        raise TypeError("Downscale factor must be int type")

    if downscale_factor <= 0:
        raise ValueError("Downscale factor must be positive")

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'."
            f"But receive Attr(data_format): {data_format} "
        )

    if in_dygraph_mode():
        return _legacy_C_ops.pixel_unshuffle(
            x, "downscale_factor", downscale_factor, "data_format", data_format
        )

    helper = LayerHelper("pixel_unshuffle", **locals())
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64', 'uint16'], 'pixel_unshuffle'
    )
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="pixel_unshuffle",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={
            "downscale_factor": downscale_factor,
            "data_format": data_format,
        },
    )
    return out


def channel_shuffle(x, groups, data_format="NCHW", name=None):
    """
    This API implements channel shuffle operation.
    See more details in :ref:`api_paddle_nn_ChannelShuffle`.

    Parameters:
        x (Tensor): 4-D tensor, the data type should be float32 or float64.
        groups (int): Number of groups to divide channels in.
        data_format (str, optional): The data format of the input and output data. An optional string of NCHW or NHWC. The default is NCHW. When it is NCHW, the data is stored in the order of [batch_size, input_channels, input_height, input_width].
        name (str, optional): Name for the operation (optional, default is None). Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Out (Tensor): Rearranged tensor keeping the original tensor shape.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F
            >>> x = paddle.arange(0, 0.6, 0.1, 'float32')
            >>> x = paddle.reshape(x, [1, 6, 1, 1])
            >>> print(x)
            Tensor(shape=[1, 6, 1, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.        ]],
              [[0.10000000]],
              [[0.20000000]],
              [[0.30000001]],
              [[0.40000001]],
              [[0.50000000]]]])
            >>> y = F.channel_shuffle(x, 3)
            >>> print(y)
            Tensor(shape=[1, 6, 1, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.        ]],
              [[0.20000000]],
              [[0.40000001]],
              [[0.10000000]],
              [[0.30000001]],
              [[0.50000000]]]])
    """
    if len(x.shape) != 4:
        raise ValueError(
            f"Input x should be 4D tensor, but received x with the shape of {x.shape}"
        )

    if not isinstance(groups, int):
        raise TypeError("groups must be int type")

    if groups <= 0:
        raise ValueError("groups must be positive")

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'."
            f"But receive Attr(data_format): {data_format} "
        )

    if in_dynamic_or_pir_mode():
        return _C_ops.channel_shuffle(x, groups, data_format)

    helper = LayerHelper("channel_shuffle", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'channel_shuffle')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="channel_shuffle",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={"groups": groups, "data_format": data_format},
    )
    return out
