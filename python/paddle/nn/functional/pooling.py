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

# TODO: define pooling functions
from ...fluid.layers import pool2d  #DEFINE_ALIAS
from ...fluid.layers import pool3d  #DEFINE_ALIAS
from ...fluid.layers import adaptive_pool2d  #DEFINE_ALIAS
from ...fluid.layers import adaptive_pool3d  #DEFINE_ALIAS
from ...fluid import core
from ...fluid.framework import in_dygraph_mode, convert_np_dtype_to_dtype_
from ...fluid.layers import utils, LayerHelper
from ...fluid.data_feeder import check_type, check_variable_and_dtype

__all__ = [
    'pool2d', 'pool3d', 'adaptive_pool2d', 'adaptive_pool3d', 'max_pool2d',
    'avg_pool2d', 'max_pool3d', 'avg_pool3d'
]


def update_padding2d(padding, data_format):
    def is_list_or_tuple(ele):
        if isinstance(ele, list) or isinstance(ele, tuple):
            return True
        return False

    if is_list_or_tuple(padding) and len(padding) == 4:
        if is_list_or_tuple(padding[0]) and (data_format == "NCHW"):
            if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                raise ValueError(
                    "Non-zero pool_padding(%s) in the batch or channel dimensions "
                    "is not supported." % str(padding))
            padding = padding[2:4]
            padding = [ele for a_list in padding for ele in a_list]
        elif is_list_or_tuple(padding[0]) and (data_format == "NHWC"):
            if not (padding[0] == [0, 0] and padding[3] == [0, 0]):
                raise ValueError(
                    "Non-zero pool_padding(%s) in the batch or channel dimensions "
                    "is not supported." % str(padding))
            padding = padding[1:3]
            padding = [ele for a_list in padding for ele in a_list]
        padding = utils.convert_to_list(padding, 4, 'padding')

        if utils._is_symmetric_padding(padding, 2):
            padding = [padding[0], padding[2]]
    else:
        padding = utils.convert_to_list(padding, 2, 'padding')

    return padding


def update_padding3d(padding, data_format):
    def is_list_or_tuple(ele):
        if isinstance(ele, (list, tuple)):
            return True
        return False

    if is_list_or_tuple(padding) and len(padding) == 5:
        if is_list_or_tuple(padding[0]) and (data_format == "NCDHW"):
            if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                raise ValueError(
                    "Non-zero pool_padding(%s) in the batch or channel dimensions "
                    "is not supported." % str(padding))
            padding = padding[2:5]
            padding = [ele for a_list in padding for ele in a_list]
        elif is_list_or_tuple(padding[0]) and (data_format == "NDHWC"):
            if not (padding[0] == [0, 0] and padding[4] == [0, 0]):
                raise ValueError(
                    "Non-zero pool_padding(%s) in the batch or channel dimensions "
                    "is not supported." % str(padding))
            padding = padding[1:4]
            padding = [ele for a_list in padding for ele in a_list]
        padding = utils.convert_to_list(padding, 6, 'padding')
        if utils._is_symmetric_padding(padding, 3):
            padding = [padding[0], padding[2], padding[4]]

    elif is_list_or_tuple(padding) and len(padding) == 6:
        padding = utils.convert_to_list(padding, 6, 'padding')
        if utils._is_symmetric_padding(padding, 3):
            padding = [padding[0], padding[2], padding[4]]
    else:
        padding = utils.convert_to_list(padding, 3, 'padding')

    return padding


def max_pool2d(x,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               return_indices=False,
               name=None,
               data_format="NCHW"):
    """
    This operation applies 2D max pooling over input feature based on the input,
    and kernel_size, stride, padding parameters. Input(X) and Output(Out) are
    in NCHW format, where N is batch size, C is the number of channels,
    H is the height of the feature, and W is the width of the feature.

    Example:
      Input:
           X shape: $(N, C, H_{in}, W_{in})$
      Attr:
           kernel_size: ksize

      Output:
           Out shape: $(N, C, H_{out}, W_{out})$
           $$
           out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, ksize[0] -1} \max_{n=0, \ldots, ksize[1]-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
           $$

    Args:
        x (Variable): The input tensor of pooling operator which is a 4-D tensor with
                          shape [N, C, H, W]. The format of input tensor is `"NCHW"` or
                          `"NHWC"`, where `N` is batch size, `C` is the number of channels,
                          `H` is the height of the feature, and `W` is the width of the
                          feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain two integers, (pool_stride_Height, pool_stride_Width).
            Otherwise, the pool stride size will be a square of an int.
        padding (string|int|list|tuple): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and when `data_format` is `"NCHW"`,
            `pool_padding` can be in the form `[[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Otherwise, the pool padding size will be a square of an int.
        ceil_mode (bool): when True, will use `ceil` instead of `floor` to compute the output shape
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
        return_indices (bool): Whether to return the max indices along with the outputs.
        data_format (string): The data format of the input and output data. An optional string from: `"NCHW"`, `"NDHW"`.
                        The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        Variable: The output tensor of pooling result. The data type is same as input tensor.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple, but the elements in the batch or channel dimensions are non-zero.
        ShapeError: If the input is not a 4-D or 5-D Tensor.
        ShapeError: If the dimension of input minus the size of `stride` is not 2.
        ShapeError: If the size of `pool_size` and `stride` is not equal.
        ShapeError: If the output's shape calculated is not greater than 0.
    Examples:
        .. code-block:: python
          import paddle
          import paddle.nn.functional as F
          import numpy as np
          paddle.disable_static()

          # max pool2d
          input = paddle.to_variable(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
          output = F.max_pool2d(input,
                                kernel_size=2,
                                stride=2, padding=0)
          # output.shape [1, 3, 16, 16]

          # for return_indices=True
          output, max_indices = F.max_pool2d(input,
                                             kernel_size=2,
                                             stride=2,
                                             padding=0,
                                             return_indices=True)
          # output.shape [1, 3, 16, 16], max_indices.shape [1, 3, 16, 16],
    """
    check_variable_and_dtype(x, 'input', ['float16', 'float32', 'float64'],
                             'conv2d')
    kernel_size = utils.convert_to_list(kernel_size, 2, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 2, 'pool_stride')

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format))
    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(padding) is \"VALID\", Attr(ceil_mode) must be False. "
                    "Received ceil_mode: True.")
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0]

    padding = update_padding2d(padding, data_format)

    if in_dygraph_mode():
        return core.ops.max_pool2d_with_index(
            x, 'ksize', kernel_size, 'global_pooling', False, 'strides', stride,
            'paddings', padding, 'padding_algorithm', padding_algorithm,
            'use_cudnn', True, 'ceil_mode', ceil_mode, 'use_mkldnn', False,
            'exclusive', True, 'data_format', data_format)

    op_type = 'max_pool2d_with_index'
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": True,
            "data_format": data_format,
        })

    return (pool_out, mask) if return_indices else pool_out


def avg_pool2d(x,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               count_include_pad=True,
               name=None,
               data_format="NCHW"):
    """
    This operation applies 2D average pooling over input features based on the input,
    and kernel_size, stride, padding parameters. Input(X) and Output(Out) are
    in NCHW format, where N is batch size, C is the number of channels,
    H is the height of the feature, and W is the width of the feature.

    Example:
      Input:
           X shape: $(N, C, H_{in}, W_{in})$
      Attr:
           kernel_size: ksize

      Output:
           Out shape: $(N, C, H_{out}, W_{out})$
           $$
           out(N_i, C_j, h, w)  = \frac{1}{ksize[0] * ksize[1]} \sum_{m=0}^{ksize[0]-1} \sum_{n=0}^{ksize[1]-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)
           $$

    Args:
        x (Variable): The input tensor of pooling operator which is a 4-D tensor with
                          shape [N, C, H, W]. The format of input tensor is `"NCHW"` or
                          `"NHWC"`, where `N` is batch size, `C` is the number of channels,
                          `H` is the height of the feature, and `W` is the width of the
                          feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain two integers, (pool_stride_Height, pool_stride_Width).
            Otherwise, the pool stride size will be a square of an int.
        padding (string|int|list|tuple): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and when `data_format` is `"NCHW"`,
            `pool_padding` can be in the form `[[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Otherwise, the pool padding size will be a square of an int.
        ceil_mode (bool): when True, will use `ceil` instead of `floor` to compute the output shape
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
        count_include_pad (bool): Whether to exclude padding points in average pooling
                          mode, default is `true`.
        data_format (string): The data format of the input and output data. An optional string from: `"NCHW"`, `"NDHW"`.
                        The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        Variable: The output tensor of pooling result. The data type is same as input tensor.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple, but the elements in the batch or channel dimensions are non-zero.
        ShapeError: If the input is not a 4-D or 5-D Tensor.
        ShapeError: If the dimension of input minus the size of `kernel_stride` is not 2.
        ShapeError: If the size of `kernel_size` and `pool_stride` is not equal.
        ShapeError: If the output's shape calculated is not greater than 0.
    Examples:
        .. code-block:: python
          import paddle
          import paddle.nn.functional as F
          import numpy as np
          paddle.disable_static()

          # max pool2d
          input = paddle.to_variable(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
          output = F.avg_pool2d(input,
                                kernel_size=2,
                                stride=2, padding=0)
          # output.shape [1, 3, 16, 16]

    """
    check_variable_and_dtype(x, 'input', ['float16', 'float32', 'float64'],
                             'conv2d')
    kernel_size = utils.convert_to_list(kernel_size, 2, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 2, 'pool_stride')

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(pool_padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(pool_padding) is \"VALID\", Attr(ceil_mode) must be False. "
                    "Received ceil_mode: True.")
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0]

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format))
    pool_padding = update_padding2d(padding, data_format)

    if in_dygraph_mode():
        return core.ops.pool2d(
            x, 'pooling_type', 'avg', 'ksize', kernel_size, 'global_pooling',
            False, 'padding_algorithm', padding_algorithm, 'strides', stride,
            'paddings', pool_padding, 'use_cudnn', True, 'ceil_mode', ceil_mode,
            'use_mkldnn', False, 'exclusive', not count_include_pad,
            'data_format', data_format)

    op_type = 'pool2d'
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": "avg",
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": pool_padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": not count_include_pad,
            "data_format": data_format,
        })

    return pool_out


def max_pool3d(x,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               return_indices=False,
               name=None,
               data_format="NCDHW"):
    """
    This operation applies 3D max pooling over input features based on the input,
    and kernel_size, stride, padding parameters. Input(X) and Output(Out) are
    in NCDHW format, where N is batch size, C is the number of channels,
    H is the height of the feature,  D is the depth of the feature, and W is the width of the feature.

    Example:
      Input:
           X shape: $(N, C, D_{in}, H_{in}, W_{in})$
      Attr:
           kernel_size: ksize

      Output:
           Out shape: $(N, C, D_{out}, H_{out}, W_{out})$
           $$
           \text{out}(N_i, C_j, d, h, w) ={} & \max_{k=0, \ldots, ksize[0]-1} \max_{m=0, \ldots, ksize[1]-1} \max_{n=0, \ldots, ksize[2]-1} \\
                                              & \text{input}(N_i, C_j, \text{stride[0]} \times d + k,
                                                             \text{stride[1]} \times h + m, \text{stride[2]} \times w + n)
           $$

    Args:
        x (Variable): The input tensor of pooling operator, which is a 5-D tensor with
                          shape [N, C, D, H, W]. The format of
                          input tensor is `"NCDHW"` or `"NDHWC"`, where `N` is batch size, `C` is
                          the number of channels, `D` is the depth of the feature,
                          `H` is the height of the feature, and `W` is the width
                          of the feature.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size
            is a tuple or list, it must contain three integers,
            (pool_size_Depth, pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        stride (string|int|list|tuple)): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool stride size is a tuple or list,
            it must contain three integers, `[stride_Depth, stride_Height, stride_Width]`.
            Otherwise, the pool stride size will be a cube of an int.
        padding (int|list|tuple): The pool padding size. If pool padding size is a tuple or list,
            it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCDHW"`, `pool_padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NDHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
        ceil_mode (bool): ${ceil_mode_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
        return_indices (bool): Whether to return the max indices along with the outputs.
        data_format (string): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
                        The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_depth, input_height, input_width]`.

    Returns:
        Variable: The output tensor of pooling result. The data type is same as input tensor.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple, but the elements in the batch or channel dimensions are non-zero.
        ShapeError: If the input is not a 4-D or 5-D Tensor.
        ShapeError: If the dimension of input minus the size of `stride` is not 2.
        ShapeError: If the size of `kernel_size` and `stride` is not equal.
        ShapeError: If the output's shape calculated is not greater than 0.
    Examples:
        .. code-block:: python
          import paddle
          import paddle.nn.functional as F
          import numpy as np
          paddle.disable_static()

          # max pool2d
          input = paddle.to_variable(np.random.uniform(-1, 1, [1, 3, 32, 32, 32]).astype(np.float32))
          output = F.max_pool2d(input,
                                kernel_size=2,
                                stride=2, padding=0)
          output.shape [1, 3, 16, 16, 16]

          # return_indices=True
          import paddle.fluid as fluid
          data = fluid.data(name='data', shape=[None, 3, 32, 32, 32], dtype='float32')

          # max pool3d
          output, max_indices = paddle.nn.functional.max_pool3d(data,
                                        kernel_size = 2,
                                        stride = 2,
                                        padding=0,
                                        return_indices=True)
          # output.shape [None, 3, 16, 16, 16], max_indices.shape [None, 3, 16, 16, 16],

    """
    check_variable_and_dtype(x, 'input', ['float16', 'float32', 'float64'],
                             'conv2d')
    kernel_size = utils.convert_to_list(kernel_size, 3, 'pool_size')
    if kernel_size is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 3, 'pool_stride')

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(pool_padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0, 0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(pool_padding) is \"VALID\", ceil_mode must be False. "
                    "Received ceil_mode: True.")
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0, 0]

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): %s" % str(data_format))
    padding = update_padding3d(padding, data_format)

    if in_dygraph_mode():
        return core.ops.max_pool3d_with_index(
            x, 'pooling_type', 'max', 'ksize', kernel_size, 'strides', stride,
            'paddings', padding, 'global_pooling', False, 'padding_algorithm',
            padding_algorithm, 'use_cudnn', True, 'ceil_mode', ceil_mode,
            'use_mkldnn', False, 'exclusive', True, 'data_format', data_format)

    op_type = "max_pool3d_with_index"
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": False,
            "data_format": data_format,
        })

    return (pool_out, mask) if return_indices else pool_out


def avg_pool3d(x,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               count_include_pad=False,
               name=None,
               data_format="NCDHW"):
    """
    This operation applies 3D max pooling over input features based on the input,
    and kernel_size, stride, padding parameters. Input(X) and Output(Out) are
    in NCDHW format, where N is batch size, C is the number of channels,
    H is the height of the feature,  D is the depth of the feature, and W is the width of the feature.

    Args:
        input (Variable): The input tensor of pooling operator, which is a 5-D tensor with
                          shape [N, C, D, H, W], where `N` is batch size, `C` is
                          the number of channels, `D` is the depth of the feature,
                          `H` is the height of the feature, and `W` is the width
                          of the feature.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size
            is a tuple or list, it must contain three integers,
            (pool_size_Depth, pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        stride (string|int|list|tuple)): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool stride size is a tuple or list,
            it must contain three integers, `[stride_Depth, stride_Height, stride_Width]`.
            Otherwise, the pool stride size will be a cube of an int.
        padding (int|list|tuple): The pool padding size. If pool padding size is a tuple or list,
            it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCDHW"`, `pool_padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NDHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
        ceil_mode (bool): ${ceil_mode_comment}
        count_include_pad (bool): Whether to exclude padding points in average pooling
                          mode, default is True.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
        data_format (string): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
                        The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_depth, input_height, input_width]`.


    Returns:
        Variable: The output tensor of pooling result. The data type is same as input tensor.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple, but the elements in the batch or channel dimensions are non-zero.
        ShapeError: If the input is not a 4-D or 5-D Tensor.
        ShapeError: If the dimension of input minus the size of `pool_stride` is not 2.
        ShapeError: If the size of `kernel_size` and `stride` is not equal.
        ShapeError: If the output's shape calculated is not greater than 0.
    Examples:
        .. code-block:: python
          import paddle.fluid as fluid
          import paddle
          data = fluid.data(name='data', shape=[None, 3, 32, 32, 32], dtype='float32')
          # avg pool3d
          pool3d = paddle.nn.functional.avg_pool3d(
                                            input = data,
                                            kernel_size = 2,
                                            stride = 2,
                                            padding=0)
          # pool3d.shape: [None, 3, 16, 16, 16]
    """
    check_variable_and_dtype(x, 'input', ['float16', 'float32', 'float64'],
                             'conv2d')
    kernel_size = utils.convert_to_list(kernel_size, 3, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 3, 'pool_stride')

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(pool_padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0, 0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(pool_padding) is \"VALID\", ceil_mode must be False. "
                    "Received ceil_mode: True.")
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0, 0]

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): %s" % str(data_format))
    padding = update_padding3d(padding, data_format)

    if in_dygraph_mode():
        return core.ops.pool3d(
            x, 'pooling_type', 'avg', 'ksize', kernel_size, 'strides', stride,
            'paddings', padding, 'global_pooling', False, 'padding_algorithm',
            padding_algorithm, 'use_cudnn', True, 'ceil_mode', ceil_mode,
            'use_mkldnn', False, 'exclusive', not count_include_pad,
            'data_format', data_format)

    op_type = "pool3d"
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out}

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'avg',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": not count_include_pad,
            "data_format": data_format,
        })

    return pool_out
