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
from ...fluid.data_feeder import check_type, check_variable_and_dtype, check_type, check_dtype, convert_dtype
from ...fluid.layers import unsqueeze, squeeze

__all__ = [
    'pool2d',
    'pool3d',
    'avg_pool1d',
    'max_pool1d',
    'adaptive_avg_pool1d',
    'adaptive_max_pool1d',
    'adaptive_avg_pool2d',
    'adaptive_avg_pool3d',
    'adaptive_pool2d',
    'adaptive_pool3d',
    'max_pool2d',
    'avg_pool2d',
    'max_pool3d',
    'avg_pool3d',
]


def check_input(x, dimension):
    if len(x.shape) != dimension:
        raise ValueError("Excepted Input X is 3-D tensor, but received {}-D {}".
                         format(len(x.shape), type(x)))


def check_instance(x, x_name, types=(int, float)):

    if not isinstance(x, types):
        raise ValueError("Excepted {} type for {} but received type: {}. ".
                         format(types, x_name, type(x)))


def update_padding1d(padding, pool_type='avg'):
    def is_list_or_tuple(ele):
        if isinstance(ele, list) or isinstance(ele, tuple):
            return True
        return False

    if is_list_or_tuple(padding):
        if padding.__len__() == 1 and not is_list_or_tuple(padding[0]):
            return [0, padding[0]]
        else:
            raise ValueError(
                "{}_pool1d() argument 'padding' should contain one int (got {})".
                format(pool_type, padding.__len__()))
    else:
        padding = [0, padding]

    return padding


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


def avg_pool1d(x,
               kernel_size,
               stride=None,
               padding=0,
               count_include_pad=True,
               ceil_mode=False,
               name=None):
    """

    This operation applies a 1D average pooling over an input signal composed
    of several input planes, based on the input, output_size, return_indices parameters.
    Input(X) and output(Out) are in NCL format, where N is batch
    size, C is the number of channels, L is the length of the feature.
    The output tensor shape will be [N, C, output_size].

    The output value of the layer with input size (N, C, L),
    output (N, C, L_{out}) and kernel_size k can be precisely described as
    For average pool1d:

    ..  math::

       Output(N_i, C_i, l) &= mean(Input[N_i, C_i, stride \times l:stride \times l+k])


    Args:
        x (Tensor): The input tensor of pooling operator which is a 3-D tensor with
                          shape [N, C, L]. where `N` is batch size, `C` is the number of channels,
                          `L` is the length of the feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain one integers.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain one integers.
        padding (string|int|list|tuple): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool padding size is a tuple or list,
            it could be the following forms: `[pad_left, pad_right]`. If padding is non-zero,
            then the input is implicitly zero-padded on both sides for padding number of points.
        count_include_pad (bool): Whether to exclude padding points in average pooling
                          mode, default is `true`.
        ceil_mode (bool): ${ceil_mode_comment}Whether to use the ceil function to calculate output height and width.
            If it is set to False, the floor function will be used. Default False
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple but its length greater than 1.
        ShapeError: If the input is not a 3-D.
        ShapeError: If the output's shape calculated is not greater than 0.


    Examples:

        .. code-block:: python

          import paddle
          import paddle.nn.functional as F
          paddle.disable_static()

          data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
          pool_out = F.avg_pool1d(data, kernel_size=2, stride=2, padding=0)
          # pool_out shape: [1, 3, 16]

    """
    """NCL to NCHW"""
    data_format = "NCHW"
    check_variable_and_dtype(x, 'input', ['float32', 'float64'], 'avg_pool1d')
    check_input(x, 3)
    x = unsqueeze(x, [2])
    kernel_size = utils.convert_to_list(kernel_size, 1, 'pool_size')
    kernel_size = [1] + kernel_size
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 1, 'pool_stride')
        stride = [1] + stride

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(padding) is \"VALID\", Attr(ceil_mode) must be False. "
                    "Received ceil_mode: True.")
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0]

    padding = update_padding1d(padding, "avg")

    if in_dygraph_mode():
        output = core.ops.pool2d(
            x, 'pooling_type', 'avg', 'ksize', kernel_size, 'global_pooling',
            False, 'strides', stride, 'paddings', padding, 'padding_algorithm',
            padding_algorithm, 'use_cudnn', not count_include_pad, 'ceil_mode',
            ceil_mode, 'use_mkldnn', False, 'exclusive', True, 'data_format',
            data_format)
        return squeeze(output, [2])

    op_type = 'pool2d'
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs={"Out": pool_out},
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

    return squeeze(pool_out, [2])


def max_pool1d(x,
               kernel_size,
               stride=None,
               padding=0,
               return_indices=False,
               ceil_mode=False,
               name=None):
    """

    Applies a 1D max pooling over an input signal composed of several input planes based
    on the input, output_size, return_indices parameters.
    Input(X) and output(Out) are in NCL format, where N is batch
    size, C is the number of channels, L is the length of the feature.

    The output value of the layer with input size (N, C, L),
    output (N, C, L_{out}) and kernel_size k can be precisely described as
    For average pool1d:

    ..  math::

       Output(N_i, C_i, l) &=  max(Input[N_i, C_i, stride \times l:stride \times l+k])}

    Args:
        x (Tensor): The input tensor of pooling operator which is a 3-D tensor with
                          shape [N, C, L], where `N` is batch size, `C` is the number of channels,
                          `L` is the length of the feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain one integers.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain one integers.
        padding (string|int|list|tuple): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool padding size is a tuple or list,
            it could be the following forms: `[pad_left, pad_right]`.
        return_indices (bool): Whether return the max indices along with the outputs. default is `False`.
        ceil_mode (bool): Whether to use the ceil function to calculate output height and width. False is the default.
            If it is set to False, the floor function will be used. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple but its length greater than 1.
        ShapeError: If the input is not a 3-D.
        ShapeError: If the output's shape calculated is not greater than 0.


    Examples:

        .. code-block:: python

          import paddle
          import paddle.nn.functional as F
          paddle.disable_static()

          data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
          pool_out = F.max_pool1d(data, kernel_size=2, stride=2, padding=0)
          # pool_out shape: [1, 3, 16]

          pool_out, indices = F.max_pool1d(data, kernel_size=2, stride=2, padding=0, return_indices=True)
          # pool_out shape: [1, 3, 16],  indices shape: [1, 3, 16]

    """
    """NCL to NCHW"""
    data_format = "NCHW"
    check_variable_and_dtype(x, 'input', ['float32', 'float64'], 'max_pool1d')
    check_input(x, 3)
    x = unsqueeze(x, [2])
    kernel_size = [1] + utils.convert_to_list(kernel_size, 1, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = [1] + utils.convert_to_list(stride, 1, 'pool_stride')

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(padding) is \"VALID\", Attr(ceil_mode) must be False. "
                    "Received ceil_mode: True.")
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0]

    padding = update_padding1d(padding, 'max')

    if in_dygraph_mode():
        pool_out = core.ops.max_pool2d_with_index(
            x, 'ksize', kernel_size, 'global_pooling', False, 'strides', stride,
            'paddings', padding, 'padding_algorithm', padding_algorithm,
            'use_cudnn', True, 'ceil_mode', ceil_mode, 'use_mkldnn', False,
            'exclusive', True, 'data_format', data_format)
        return (squeeze(pool_out[0], [2]), squeeze(
            pool_out[1], [2])) if return_indices else squeeze(pool_out[0], [2])

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

    return (squeeze(pool_out, [2]),
            squeeze(mask, [2])) if return_indices else squeeze(pool_out, [2])


def adaptive_avg_pool1d(x, output_size, name=None):
    """

    This operation applies a 1D adaptive average pooling over an input signal composed
    of several input planes, based on the input, output_size, return_indices parameters.
    Input(X) and output(Out) are in NCL format, where N is batch
    size, C is the number of channels, L is the length of the feature.
    The output tensor shape will be [N, C, output_size].

    For average adaptive pool1d:

    ..  math::

        lstart &= floor(i * L_{in} / L_{out})

        lend &= ceil((i + 1) * L_{in} / L_{out})

        Output(i) &= \\frac{sum(Input[lstart:lend])}{(lstart - lend)}

    Args:
        x (Tensor): The input tensor of pooling operator, which is a 3-D tensor
                              with shape [N, C, L].  The format of input tensor is NCL,
                              where N is batch size, C is the number of channels, L is the
                              length of the feature. The data type is float32 or float64.
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
                it must contain one int.
        name(str, optional): For detailed information, please refer
                                 to :ref:`api_guide_Name`. Usually name is no need to set and
                                 None by default.

    Returns:
            Tensor: The output tensor of adaptive average pooling result. The data type is same
                      as input tensor.

    Raises:
            ValueError: 'output_size' should be a integer or list or tuple with length as 1.

    Examples:
        .. code-block:: python

              # average adaptive pool1d
              # suppose input data in shape of [N, C, L], `output_size` is m or [m],
              # output shape is [N, C, m], adaptive pool divide L dimension
              # of input data into m grids averagely and performs poolings in each
              # grid to get output.
              # adaptive max pool performs calculations as follow:
              #
              #     for i in range(m):
              #         lstart = floor(i * L / m)
              #         lend = ceil((i + 1) * L / m)
              #         output[:, :, i] = sum(input[:, :, lstart: lend])/(lstart - lend)
              #
              import paddle
              import paddle.nn.functional as F
              paddle.disable_static()

              data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
              pool_out = F.adaptive_average_pool1d(data, output_size=16)
              # pool_out shape: [1, 3, 16])
    """
    pool_type = 'avg'
    check_variable_and_dtype(x, 'input', ['float32', 'float64'],
                             'adaptive_pool2d')
    check_input(x, 3)
    check_type(output_size, 'pool_size', (int), 'adaptive_pool1d')

    pool_size = [1] + utils.convert_to_list(output_size, 1, 'pool_size')

    l_type = "pool2d"
    x = unsqueeze(x, [2])
    if in_dygraph_mode():
        pool_out = core.ops.pool2d(x, 'pooling_type', pool_type, 'ksize',
                                   pool_size, 'adaptive', True)
        return squeeze(pool_out, [2])

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}
    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return squeeze(pool_out, [2])


def adaptive_max_pool1d(x, output_size, return_indices=False, name=None):
    """
    This operation applies a 1D adaptive max pooling over an input signal composed
    of several input planes, based on the input, output_size, return_indices parameters.
    Input(X) and output(Out) are in NCL format, where N is batch
    size, C is the number of channels, L is the length of the feature.
    The output tensor shape will be [N, C, output_size].

    For max adaptive pool1d:

    ..  math::

        lstart &= floor(i * L_{in} / L_{out})

        lend &= ceil((i + 1) * L_{in} / L_{out})

        Output(i) &= max(Input[lstart:lend])}

    Args:
        x (Tensor): The input tensor of pooling operator, which is a 3-D tensor
                              with shape [N, C, L].  The format of input tensor is NCL,
                              where N is batch size, C is the number of channels, L is the
                              length of the feature. The data type is float32 or float64.
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
                it must contain one int.
        return_indices (bool): If true, the index of max pooling point will be returned along
                with outputs. It cannot be set in average pooling type. Default False.
        name(str, optional): For detailed information, please refer
                                 to :ref:`api_guide_Name`. Usually name is no need to set and
                                 None by default.

    Returns:
            Tensor: The output tensor of adaptive pooling result. The data type is same
                      as input tensor.

    Raises:
            ValueError: 'output_size' should be a integer or list or tuple with length as 1.

    Examples:
        .. code-block:: python

              # max adaptive pool1d
              # suppose input data in shape of [N, C, L], `output_size` is m or [m],
              # output shape is [N, C, m], adaptive pool divide L dimension
              # of input data into m grids averagely and performs poolings in each
              # grid to get output.
              # adaptive max pool performs calculations as follow:
              #
              #     for i in range(m):
              #         lstart = floor(i * L / m)
              #         lend = ceil((i + 1) * L / m)
              #         output[:, :, i] = max(input[:, :, lstart: lend])
              #
              import paddle
              import paddle.nn.functional as F
              paddle.disable_static()

              data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
              pool_out = F.adaptive_max_pool1d(data, output_size=16)
              # pool_out shape: [1, 3, 16])

              pool_out, indices = F.adaptive_max_pool1d(data, output_size=16, return_indices=True)
              # pool_out shape: [1, 3, 16] indices  shape: [1, 3, 16]

    """
    pool_type = 'max'
    check_variable_and_dtype(x, 'input', ['float32', 'float64'],
                             'adaptive_max_pool1d')
    check_input(x, 3)
    check_type(output_size, 'pool_size', (int), 'adaptive_max_pool1d')
    check_type(return_indices, 'return_indices', bool, 'adaptive_max_pool1d')

    pool_size = [1] + utils.convert_to_list(output_size, 1, 'pool_size')

    l_type = 'max_pool2d_with_index'

    x = unsqueeze(x, [2])
    if in_dygraph_mode():
        pool_out = core.ops.max_pool2d_with_index(
            x, 'pooling_type', pool_type, 'ksize', pool_size, 'adaptive', True)
        return (squeeze(pool_out[0], [2]), squeeze(
            pool_out[1], [2])) if return_indices else squeeze(pool_out[0], [2])

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return (squeeze(pool_out, [2]),
            squeeze(mask, [2])) if return_indices else squeeze(pool_out, [2])


def max_pool2d(x,
               kernel_size,
               stride=None,
               padding=0,
               return_indices=False,
               ceil_mode=False,
               data_format="NCHW",
               name=None):
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
           stride: stride

      Output:
           Out shape: $(N, C, H_{out}, W_{out})$
           $$
           out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, ksize[0] -1} \max_{n=0, \ldots, ksize[1]-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
           $$

    Args:
        x (Tensor): The input tensor of pooling operator which is a 4-D tensor with
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
        return_indices (bool): Whether to return the max indices along with the outputs.
        data_format (string): The data format of the input and output data. An optional string from: `"NCHW"`, `"NDHW"`.
                        The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.
    Examples:
        .. code-block:: python
          import paddle
          import paddle.nn.functional as F
          import numpy as np
          paddle.disable_static()

          # max pool2d
          input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
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
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'max_pool2d')
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
        output = core.ops.max_pool2d_with_index(
            x, 'ksize', kernel_size, 'global_pooling', False, 'strides', stride,
            'paddings', padding, 'padding_algorithm', padding_algorithm,
            'use_cudnn', True, 'ceil_mode', ceil_mode, 'use_mkldnn', False,
            'exclusive', True, 'data_format', data_format)
        return output if return_indices else output[0]

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
               divisor_override=None,
               data_format="NCHW",
               name=None):
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
        x (Tensor): The input tensor of pooling operator which is a 4-D tensor with
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
        count_include_pad (bool): Whether to exclude padding points in average pooling
                          mode, default is `true`.
        divisor_override (float): if specified, it will be used as divisor, otherwise kernel_size will be used. Default None.
        data_format (string): The data format of the input and output data. An optional string from: `"NCHW"`, `"NDHW"`.
                        The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.
    Examples:
        .. code-block:: python
          import paddle
          import paddle.nn.functional as F
          import numpy as np
          paddle.disable_static()

          # avg pool2d
          input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
          output = F.avg_pool2d(input,
                                kernel_size=2,
                                stride=2, padding=0)
          # output.shape [1, 3, 16, 16]

    """
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'avg_pool2d')
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
        output = core.ops.pool2d(
            x, 'pooling_type', 'avg', 'ksize', kernel_size, 'global_pooling',
            False, 'padding_algorithm', padding_algorithm, 'strides', stride,
            'paddings', pool_padding, 'use_cudnn', True, 'ceil_mode', ceil_mode,
            'use_mkldnn', False, 'exclusive', not count_include_pad,
            'data_format', data_format)
        if divisor_override is None:
            return output
        else:
            check_instance(divisor_override, "divisor_override")
            return output * (kernel_size[0] * kernel_size[1]) / divisor_override

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

    if divisor_override is None:
        return pool_out
    else:
        check_instance(divisor_override, "divisor_override")
        return pool_out * (kernel_size[0] * kernel_size[1]) / divisor_override


def max_pool3d(x,
               kernel_size,
               stride=None,
               padding=0,
               return_indices=False,
               ceil_mode=False,
               data_format="NCDHW",
               name=None):
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
        x (Tensor): The input tensor of pooling operator, which is a 5-D tensor with
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
        return_indices (bool): Whether to return the max indices along with the outputs.
        data_format (string): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
                        The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.
    Examples:
        .. code-block:: python
          import paddle
          import paddle.nn.functional as F
          import numpy as np
          paddle.disable_static()

          # max pool3d
          input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32, 32]).astype(np.float32))
          output = F.max_pool2d(input,
                                kernel_size=2,
                                stride=2, padding=0)
          output.shape [1, 3, 16, 16, 16]

          # for return_indices=True
          input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32, 32]).astype(np.float32))
          output, max_indices = paddle.nn.functional.max_pool3d(input,
                                        kernel_size = 2,
                                        stride = 2,
                                        padding=0,
                                        return_indices=True)
          # output.shape [None, 3, 16, 16, 16], max_indices.shape [None, 3, 16, 16, 16],

    """
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'max_pool3d')
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
        output = core.ops.max_pool3d_with_index(
            x, 'pooling_type', 'max', 'ksize', kernel_size, 'strides', stride,
            'paddings', padding, 'global_pooling', False, 'padding_algorithm',
            padding_algorithm, 'use_cudnn', True, 'ceil_mode', ceil_mode,
            'use_mkldnn', False, 'exclusive', True, 'data_format', data_format)
        return output if return_indices else output[0]

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
               divisor_override=None,
               data_format="NCDHW",
               name=None):
    """
    This operation applies 3D max pooling over input features based on the input,
    and kernel_size, stride, padding parameters. Input(X) and Output(Out) are
    in NCDHW format, where N is batch size, C is the number of channels,
    H is the height of the feature,  D is the depth of the feature, and W is the width of the feature.

    Args:
        input (Tensor): The input tensor of pooling operator, which is a 5-D tensor with
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
        divisor_override (int|float) if specified, it will be used as divisor, otherwise kernel_size will be used. Default None.
        data_format (string): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
                        The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.


    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.
    Examples:
        .. code-block:: python
          import paddle.fluid as fluid
          import paddle
          input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32, 32]).astype(np.float32))
          # avg pool3d
          pool3d = paddle.nn.functional.avg_pool3d(
                                            input,
                                            kernel_size = 2,
                                            stride = 2,
                                            padding=0)
          # pool3d.shape: [1, 3, 16, 16, 16]
    """
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'max_pool3d')
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
        output = core.ops.pool3d(
            x, 'pooling_type', 'avg', 'ksize', kernel_size, 'strides', stride,
            'paddings', padding, 'global_pooling', False, 'padding_algorithm',
            padding_algorithm, 'use_cudnn', True, 'ceil_mode', ceil_mode,
            'use_mkldnn', False, 'exclusive', not count_include_pad,
            'data_format', data_format)
        if divisor_override is None:
            return output
        else:
            check_instance(divisor_override, "divisor_override")
            return output * (kernel_size[0] * kernel_size[1] *
                             kernel_size[2]) / divisor_override

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

    if divisor_override is None:
        return pool_out
    else:
        check_instance(divisor_override, "divisor_override")
        return pool_out * (kernel_size[0] * kernel_size[1] *
                           kernel_size[2]) / divisor_override


def adaptive_avg_pool2d(x, output_size, data_format='NCHW', name=None):
    """

    This operation applies 2D adaptive avg pooling on input tensor. The h and w dimensions
    of the output tensor are determined by the parameter output_size.
    See more detail in :ref:`api_nn_pooling_AdaptiveAvgPool2d` .

    For avg adaptive pool2d:

    ..  math::

       hstart &= floor(i * H_{in} / H_{out})

       hend &= ceil((i + 1) * H_{in} / H_{out})

       wstart &= floor(j * W_{in} / W_{out})

       wend &= ceil((j + 1) * W_{in} / W_{out})

       Output(i ,j) &= \\frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}

    Args:
        x (Tensor): The input tensor of adaptive avg pool2d operator, which is a 4-D tensor.
                          The data type can be float32 or float64.
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two element, (H, W). H and W can be either a int, or None which means
            the size will be the same as that of the input.
        data_format (str): The data format of the input and output data. An optional string
            from: "NCHW", "NHWC". The default is "NCHW". When it is "NCHW", the data is stored in
            the order of: [batch_size, input_channels, input_height, input_width].
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of avg adaptive pool2d result. The data type is same as input tensor.

    Raises:
        ValueError: If `data_format` is not "NCHW" or "NHWC".

    Examples:
        .. code-block:: python

            # adaptive avg pool2d
            # suppose input data in shape of [N, C, H, W], `output_size` is [m, n],
            # output shape is [N, C, m, n], adaptive pool divide H and W dimensions
            # of input data into m * n grids averagely and performs poolings in each
            # grid to get output.
            # adaptive avg pool performs calculations as follow:
            #
            #     for i in range(m):
            #         for j in range(n):
            #             hstart = floor(i * H / m)
            #             hend = ceil((i + 1) * H / m)
            #             wstart = floor(i * W / n)
            #             wend = ceil((i + 1) * W / n)
            #             output[:, :, i, j] = avg(input[:, :, hstart: hend, wstart: wend])
            #
            import paddle
            import numpy as np
            paddle.disable_static()
            input_data = np.random.rand(2, 3, 32, 32)
            x = paddle.to_tensor(input_data)
            # x.shape is [2, 3, 32, 32]
            pool_out = paddle.nn.functional.adaptive_avg_pool2d(
                            x = x,
                            output_size=[3, 3])
            # pool_out.shape is [2, 3, 3, 3]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                                 'adaptive_avg_pool2d')
    check_type(data_format, 'data_format', str, 'adaptive_avg_pool2d')

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    if data_format == "NCHW":
        in_h, in_w = x.shape[2:4]
    else:
        in_h, in_w = x.shape[1:3]

    if isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 2, 'output_size')
    else:
        if output_size[0] == None:
            output_size[0] = in_h
        if output_size[1] == None:
            output_size[1] = in_w

    if in_dygraph_mode():
        output = core.ops.pool2d(x, 'pooling_type', 'avg', 'ksize', output_size,
                                 'global_pooling', False, 'adaptive', True,
                                 'data_format', data_format)
        return output

    l_type = 'pool2d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": "avg",
            "ksize": output_size,
            "adaptive": True,
            "data_format": data_format,
        })

    return pool_out


def adaptive_avg_pool3d(x, output_size, data_format='NCDHW', name=None):
    """

    This operation applies 3D adaptive avg pooling on input tensor. The h and w dimensions
    of the output tensor are determined by the parameter output_size.
    See more detail in :ref:`api_nn_pooling_AdaptiveAvgPool3d` .

    For avg adaptive pool3d:

    ..  math::

      dstart &= floor(i * D_{in} / D_{out})

      dend &= ceil((i + 1) * D_{in} / D_{out})

      hstart &= floor(j * H_{in} / H_{out})

      hend &= ceil((j + 1) * H_{in} / H_{out})

      wstart &= floor(k * W_{in} / W_{out})

      wend &= ceil((k + 1) * W_{in} / W_{out})

      Output(i ,j, k) &= \\frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{(dend - dstart) * (hend - hstart) * (wend - wstart)}

    Args:
        x (Tensor): The input tensor of adaptive avg pool3d operator, which is a 5-D tensor.
                          The data type can be float32, float64.
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain three elements, (D, H, W). D, H and W can be either a int, or None which means
            the size will be the same as that of the input.
        data_format (str): The data format of the input and output data. An optional string
            from: "NCDHW", "NDHWC". The default is "NCDHW". When it is "NCDHW", the data is stored in
            the order of: [batch_size, input_channels, input_depth, input_height, input_width].
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of avg adaptive pool3d result. The data type is same as input tensor.

    Raises:
        ValueError: If `data_format` is not "NCDHW" or "NDHWC".

    Examples:
        .. code-block:: python

            # adaptive avg pool3d
            # suppose input data in shape of [N, C, D, H, W], `output_size` is [l, m, n],
            # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
            # of input data into l * m * n grids averagely and performs poolings in each
            # grid to get output.
            # adaptive avg pool performs calculations as follow:
            #
            #     for i in range(l):
            #         for j in range(m):
            #             for k in range(n):
            #                 dstart = floor(i * D / l)
            #                 dend = ceil((i + 1) * D / l)
            #                 hstart = floor(j * H / m)
            #                 hend = ceil((j + 1) * H / m)
            #                 wstart = floor(k * W / n)
            #                 wend = ceil((k + 1) * W / n)
            #                 output[:, :, i, j, k] =
            #                     avg(input[:, :, dstart:dend, hstart: hend, wstart: wend])
            import paddle
            import numpy as np
            paddle.disable_static()
            input_data = np.random.rand(2, 3, 8, 32, 32)
            x = paddle.to_tensor(input_data)
            # x.shape is [2, 3, 8, 32, 32]
            pool_out = paddle.nn.functional.adaptive_avg_pool3d(
                            x = x,
                            output_size=[3, 3, 3])
            # pool_out.shape is [2, 3, 3, 3, 3]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                                 'adaptive_avg_pool3d')
    check_type(data_format, 'data_format', str, 'adaptive_avg_pool3d')

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    if data_format == "NCDHW":
        in_l, in_h, in_w = x.shape[2:5]
    else:
        in_l, in_h, in_w = x.shape[1:4]

    if isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 3, 'output_size')
    else:
        if output_size[0] == None:
            output_size[0] = in_l
        if output_size[1] == None:
            output_size[1] = in_h
        if output_size[2] == None:
            output_size[2] = in_w

    if in_dygraph_mode():
        output = core.ops.pool3d(x, 'pooling_type', 'avg', 'ksize', output_size,
                                 'global_pooling', False, 'adaptive', True,
                                 'data_format', data_format)
        return output

    l_type = 'pool3d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": "avg",
            "ksize": output_size,
            "adaptive": True,
            "data_format": data_format,
        })

    return pool_out
