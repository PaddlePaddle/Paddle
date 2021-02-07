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

from ...fluid.dygraph import layers
from ...fluid.layer_helper import LayerHelper
from .. import functional as F

__all__ = [
    'AvgPool1D',
    'AvgPool2D',
    'AvgPool3D',
    'MaxPool1D',
    'MaxPool2D',
    'MaxPool3D',
    'AdaptiveAvgPool1D',
    'AdaptiveAvgPool2D',
    'AdaptiveAvgPool3D',
    'AdaptiveMaxPool1D',
    'AdaptiveMaxPool2D',
    'AdaptiveMaxPool3D',
]


class AvgPool1D(layers.Layer):
    """
    This operation applies a 1D average pooling over an input signal composed
    of several input planes, based on the input, output_size, return_mask parameters.
    Input(X) and output(Out) are in NCL format, where N is batch
    size, C is the number of channels, L is the length of the feature.
    The output tensor shape will be [N, C, output_size].

    The output value of the layer with input size (N, C, L),
    output (N, C, L_{out}) and kernel_size k can be precisely described as
    For average pool1d:

    ..  math::

       Output(N_i, C_i, l) &= mean(Input[N_i, C_i, stride \times l:stride \times l+k])


    Args:
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain an integer.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain an integer.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 1, which means the feature map is zero padded by the size of `padding[0]` on every sides.
            4. A list[int] or tuple(int) whose length is 2. It has the form [pad_before, pad_after].
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        exclusive (bool): Whether to exclude padding points in average pooling
                          mode, default is `True`.
        ceil_mode (bool): ${ceil_mode_comment}Whether to use the ceil function to calculate output height and width.
            If it is set to False, the floor function will be used. The default value is False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        None.

    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple but its length greater than 1.
        ShapeError: If the input is not a 3-D tensor.
        ShapeError: If the output's shape calculated is not greater than 0.


    Shape:
        - inpuut: 3-D tensor.
        - output: 3-D tensor

    Examples:

        .. code-block:: python

          import paddle
          import paddle.nn as nn
          import numpy as np

          data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
          AvgPool1D = nn.AvgPool1D(kernel_size=2, stride=2, padding=0)
          pool_out = AvgPool1D(data)
          # pool_out shape: [1, 3, 16]

    """

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 exclusive=True,
                 ceil_mode=False,
                 name=None):
        super(AvgPool1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.exclusive = exclusive
        self.name = name

    def forward(self, x):
        out = F.avg_pool1d(x, self.kernel_size, self.stride, self.padding,
                           self.exclusive, self.ceil_mode, self.name)
        return out

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}'.format(
            **self.__dict__)


class AvgPool2D(layers.Layer):
    r"""
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
       kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain two integers, (pool_stride_Height, pool_stride_Width).
            Otherwise, the pool stride size will be a square of an int.

        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 2, [pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 4. [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode (bool): when True, will use `ceil` instead of `floor` to compute the output shape
        exclusive (bool): Whether to exclude padding points in average pooling
                          mode, default is `true`.
        divisor_override (float): if specified, it will be used as divisor, otherwise kernel_size will be used. Default None.
        data_format (string): The data format of the input and output data. An optional string from: `"NCHW"`, `"NDHW"`.
                        The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Shape:
        - x: 4-D tensor.
        - out: 2-D tensor

    Returns: None.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.
    Examples:
        .. code-block:: python

          import paddle
          import paddle.nn as nn
          import numpy as np

          # max pool2d
          input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
          AvgPool2D = nn.AvgPool2D(kernel_size=2,
                                stride=2, padding=0)
          output = AvgPool2D(input)
          # output.shape [1, 3, 16, 16]

    """

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 ceil_mode=False,
                 exclusive=True,
                 divisor_override=None,
                 data_format="NCHW",
                 name=None):
        super(AvgPool2D, self).__init__()
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.exclusive = exclusive
        self.divisor = divisor_override
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        return F.avg_pool2d(
            x,
            kernel_size=self.ksize,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            exclusive=self.exclusive,
            divisor_override=self.divisor,
            data_format=self.data_format,
            name=self.name)

    def extra_repr(self):
        return 'kernel_size={ksize}, stride={stride}, padding={padding}'.format(
            **self.__dict__)


class AvgPool3D(layers.Layer):
    """
    This operation applies 3D max pooling over input features based on the input,
    and kernel_size, stride, padding parameters. Input(X) and Output(Out) are
    in NCDHW format, where N is batch size, C is the number of channels,
    H is the height of the feature,  D is the depth of the feature, and W is the width of the feature.

    Args:
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size
            is a tuple or list, it must contain three integers,
            (kernel_size_Depth, kernel_size_Height, kernel_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain three integers, [stride_Depth, stride_Height, stride_Width).
            Otherwise, the pool stride size will be a cube of an int.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 3, [pad_depth, pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 6. [pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode (bool): ${ceil_mode_comment}
        exclusive (bool): Whether to exclude padding points in average pooling
                          mode, default is True.
        divisor_override (int|float) if specified, it will be used as divisor, otherwise kernel_size will be used. Default None.
        data_format (string): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
                        The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns: None.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.

    Shape:
        - x: 5-D tensor.
        - out: 5-D tensor.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.nn as nn
          import numpy as np

          # avg pool3d
          input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 2, 3, 32, 32]).astype(np.float32))
          AvgPool3D = nn.AvgPool3D(kernel_size=2,
                                   stride=2, padding=0)
          output = AvgPool3D(input)
          # output.shape [1, 2, 3, 16, 16]

    """

    def __init__(self,
                 kernel_size,
                 stride,
                 padding=0,
                 ceil_mode=False,
                 exclusive=True,
                 divisor_override=None,
                 data_format="NCDHW",
                 name=None):
        super(AvgPool3D, self).__init__()
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.exclusive = exclusive
        self.divisor = divisor_override
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        return F.avg_pool3d(
            x,
            kernel_size=self.ksize,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            exclusive=self.exclusive,
            divisor_override=self.divisor,
            data_format=self.data_format,
            name=self.name)

    def extra_repr(self):
        return 'kernel_size={ksize}, stride={stride}, padding={padding}'.format(
            **self.__dict__)


class MaxPool1D(layers.Layer):
    """
    Applies a 1D max pooling over an input signal composed of several input planes based
    on the input, output_size, return_mask parameters.
    Input(X) and output(Out) are in NCL format, where N is batch
    size, C is the number of channels, L is the length of the feature.

    The output value of the layer with input size (N, C, L),
    output (N, C, L_{out}) and kernel_size k can be precisely described as
    For average pool1d:

    ..  math::

       Output(N_i, C_i, l) &=  max(Input[N_i, C_i, stride \times l:stride \times l+k])}

    Args:
       kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain an integer.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain an integer.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An integer, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 1, which means the feature map is zero padded by the size of `padding[0]` on every sides.
            4. A list[int] or tuple(int) whose length is 2. It has the form [pad_before, pad_after].
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        return_mask (bool): Whether return the max indices along with the outputs. default is `False`.
        ceil_mode (bool): Whether to use the ceil function to calculate output height and width. False is the default.
            If it is set to False, the floor function will be used. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        None.

    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple but its length greater than 1.
        ShapeError: If the input is not a 3-D.
        ShapeError: If the output's shape calculated is not greater than 0.


    Shape:
        - x: 3-D tensor.
        - out: 3-D tensor.

    Examples:

        .. code-block:: python

          import paddle
          import paddle.nn as nn
          import numpy as np

          data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
          MaxPool1D = nn.MaxPool1D(kernel_size=2, stride=2, padding=0)
          pool_out = MaxPool1D(data)
          # pool_out shape: [1, 3, 16]

          MaxPool1D = nn.MaxPool1D(kernel_size=2, stride=2, padding=0, return_mask=True)
          pool_out, indices = MaxPool1D(data)
          # pool_out shape: [1, 3, 16], indices shape: [1, 3, 16]

    """

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 return_mask=False,
                 ceil_mode=False,
                 name=None):
        super(MaxPool1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.return_mask = return_mask
        self.name = name

    def forward(self, input):
        out = F.max_pool1d(input, self.kernel_size, self.stride, self.padding,
                           self.return_mask, self.ceil_mode, self.name)
        return out

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}'.format(
            **self.__dict__)


class MaxPool2D(layers.Layer):
    r"""
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
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain two integers, (pool_stride_Height, pool_stride_Width).
            Otherwise, the pool stride size will be a square of an int.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 2, [pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 4. [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode (bool): when True, will use `ceil` instead of `floor` to compute the output shape
        return_mask (bool): Whether to return the max indices along with the outputs.
        data_format (string): The data format of the input and output data. An optional string from: `"NCHW"`, `"NDHW"`.
                        The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns: None
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.

    Shape:
        - x: 4-D tensor.
        - out: 4-D tensor.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.nn as nn
          import numpy as np

          # max pool2d
          input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
          MaxPool2D = nn.MaxPool2D(kernel_size=2,
                                   stride=2, padding=0)
          output = MaxPool2D(input)
          # output.shape [1, 3, 16, 16]

          # for return_mask=True
          MaxPool2D = nn.MaxPool2D(kernel_size=2, stride=2, padding=0, return_mask=True)
          output, max_indices = MaxPool2D(input)
          # output.shape [1, 3, 16, 16], max_indices.shape [1, 3, 16, 16],
    """

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 return_mask=False,
                 ceil_mode=False,
                 data_format="NCHW",
                 name=None):
        super(MaxPool2D, self).__init__()
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        self.return_mask = return_mask
        self.ceil_mode = ceil_mode
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        return F.max_pool2d(
            x,
            kernel_size=self.ksize,
            stride=self.stride,
            padding=self.padding,
            return_mask=self.return_mask,
            ceil_mode=self.ceil_mode,
            data_format=self.data_format,
            name=self.name)

    def extra_repr(self):
        return 'kernel_size={ksize}, stride={stride}, padding={padding}'.format(
            **self.__dict__)


class MaxPool3D(layers.Layer):
    """
    This operation applies 3D max pooling over input features based on the input,
    and kernel_size, stride, padding parameters. Input(X) and Output(Out) are
    in NCDHW format, where N is batch size, C is the number of channels,
    H is the height of the feature,  D is the depth of the feature, and W is the width of the feature.

    Args:
        kernel_size (int|list|tuple): The pool kernel size. If the kernel size
            is a tuple or list, it must contain three integers,
            (kernel_size_Depth, kernel_size_Height, kernel_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain three integers, [stride_Depth, stride_Height, stride_Width).
            Otherwise, the pool stride size will be a cube of an int.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 3, [pad_depth, pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 6. [pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode (bool): ${ceil_mode_comment}
        return_mask (bool): Whether to return the max indices along with the outputs.
        data_format (string): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
                        The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.


    Returns:None.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.

    Shape:
        - x: 5-D tensor.
        - out: 5-D tensor.

    Examples:
        .. code-block:: python

          import paddle
          import paddle.nn as nn
          import numpy as np

          # max pool3d
          input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 2, 3, 32, 32]).astype(np.float32))
          MaxPool3D = nn.MaxPool3D(kernel_size=2,
                                   stride=2, padding=0)
          output = MaxPool3D(input)
          # output.shape [1, 2, 3, 16, 16]

          # for return_mask=True
          MaxPool3D = nn.MaxPool3D(kernel_size=2, stride=2, padding=0, return_mask=True)
          output, max_indices = MaxPool3D(input)
          # output.shape [1, 2, 3, 16, 16], max_indices.shape [1, 2, 3, 16, 16],
    """

    def __init__(self,
                 kernel_size,
                 stride,
                 padding,
                 return_mask=False,
                 ceil_mode=False,
                 data_format="NCDHW",
                 name=None):
        super(MaxPool3D, self).__init__()
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        self.return_mask = return_mask
        self.ceil_mode = ceil_mode
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        return F.max_pool3d(
            x,
            kernel_size=self.ksize,
            stride=self.stride,
            padding=self.padding,
            return_mask=self.return_mask,
            ceil_mode=self.ceil_mode,
            data_format=self.data_format,
            name=self.name)

    def extra_repr(self):
        return 'kernel_size={ksize}, stride={stride}, padding={padding}'.format(
            **self.__dict__)


class AdaptiveAvgPool1D(layers.Layer):
    r"""

    This operation applies a 1D adaptive average pooling over an input signal composed
    of several input planes, based on the input, output_size, return_mask parameters.
    Input(X) and output(Out) are in NCL format, where N is batch
    size, C is the number of channels, L is the length of the feature.
    The output tensor shape will be [N, C, output_size].

    For average adaptive pool1d:

    ..  math::

       lstart &= floor(i * L_{in} / L_{out})

       lend &= ceil((i + 1) * L_{in} / L_{out})

       Output(i) &= \\frac{sum(Input[lstart:lend])}{(lstart - lend)}

    Args:
        output_size (int): The target output size. It must be an integer.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        None.

    Raises:
        ValueError: 'output_size' should be an integer.

    Shape:
        - x: 3-D tensor.
        - out: 3-D tensor.

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
          import paddle.nn as nn
          import numpy as np

          data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
          AdaptiveAvgPool1D = nn.AdaptiveAvgPool1D(output_size=16)
          pool_out = AdaptiveAvgPool1D(data)
          # pool_out shape: [1, 3, 16]
    """

    def __init__(self, output_size, name=None):
        super(AdaptiveAvgPool1D, self).__init__()
        self.output_size = output_size
        self.name = name

    def forward(self, input):
        return F.adaptive_avg_pool1d(input, self.output_size, self.name)

    def extra_repr(self):
        return 'output_size={}'.format(self.output_size)


class AdaptiveAvgPool2D(layers.Layer):
    r"""

    This operation applies 2D adaptive avg pooling on input tensor. The h and w dimensions
    of the output tensor are determined by the parameter output_size.

    For avg adaptive pool2d:

    ..  math::

       hstart &= floor(i * H_{in} / H_{out})

       hend &= ceil((i + 1) * H_{in} / H_{out})

       wstart &= floor(j * W_{in} / W_{out})

       wend &= ceil((j + 1) * W_{in} / W_{out})

       Output(i ,j) &= \\frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}


    Parameters:
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two element, (H, W). H and W can be either a int, or None which means
            the size will be the same as that of the input.
        data_format (str): The data format of the input and output data. An optional string
            from: "NCHW", "NHWC". The default is "NCHW". When it is "NCHW", the data is stored in
            the order of: [batch_size, input_channels, input_height, input_width].
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Shape:
        x (Tensor): The input tensor of adaptive avg pool2d operator, which is a 4-D tensor. The data type can be float32, float64.
        output (Tensor): The output tensor of adaptive avg pool2d operator, which is a 4-D tensor. The data type is same as input x.

    Returns:
        A callable object of AdaptiveAvgPool2D.

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

            input_data = np.random.rand(2, 3, 32, 32)
            x = paddle.to_tensor(input_data)
            # x.shape is [2, 3, 32, 32]
            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool2D(output_size=3)
            pool_out = adaptive_avg_pool(x = x)
            # pool_out.shape is [2, 3, 3, 3]
    """

    def __init__(self, output_size, data_format="NCHW", name=None):
        super(AdaptiveAvgPool2D, self).__init__()
        self._output_size = output_size
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x,
            output_size=self._output_size,
            data_format=self._data_format,
            name=self._name)

    def extra_repr(self):
        return 'output_size={}'.format(self._output_size)


class AdaptiveAvgPool3D(layers.Layer):
    r"""

    This operation applies 3D adaptive avg pooling on input tensor. The h and w dimensions
    of the output tensor are determined by the parameter output_size.

    For avg adaptive pool3d:

    ..  math::

      dstart &= floor(i * D_{in} / D_{out})

      dend &= ceil((i + 1) * D_{in} / D_{out})

      hstart &= floor(j * H_{in} / H_{out})

      hend &= ceil((j + 1) * H_{in} / H_{out})

      wstart &= floor(k * W_{in} / W_{out})

      wend &= ceil((k + 1) * W_{in} / W_{out})

      Output(i ,j, k) &= \\frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{(dend - dstart) * (hend - hstart) * (wend - wstart)}


    Parameters:
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain three elements, (D, H, W). D, H and W can be either a int, or None which means
            the size will be the same as that of the input.
        data_format (str): The data format of the input and output data. An optional string
            from: "NCDHW", "NDHWC". The default is "NCDHW". When it is "NCDHW", the data is stored in
            the order of: [batch_size, input_channels, input_depth, input_height, input_width].
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Shape:
        x (Tensor): The input tensor of adaptive avg pool3d operator, which is a 5-D tensor. The data type can be float32, float64.
        output (Tensor): The output tensor of adaptive avg pool3d operator, which is a 5-D tensor. The data type is same as input x.

    Returns:
        A callable object of AdaptiveAvgPool3D.

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

            input_data = np.random.rand(2, 3, 8, 32, 32)
            x = paddle.to_tensor(input_data)
            # x.shape is [2, 3, 8, 32, 32]
            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool3D(output_size=3)
            pool_out = adaptive_avg_pool(x = x)
            # pool_out = [2, 3, 3, 3, 3]
    """

    def __init__(self, output_size, data_format="NCDHW", name=None):
        super(AdaptiveAvgPool3D, self).__init__()
        self._output_size = output_size
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return F.adaptive_avg_pool3d(
            x,
            output_size=self._output_size,
            data_format=self._data_format,
            name=self._name)

    def extra_repr(self):
        return 'output_size={}'.format(self._output_size)


class AdaptiveMaxPool1D(layers.Layer):
    """

    This operation applies a 1D adaptive max pooling over an input signal composed
    of several input planes, based on the input, output_size, return_mask parameters.
    Input(X) and output(Out) are in NCL format, where N is batch
    size, C is the number of channels, L is the length of the feature.
    The output tensor shape will be [N, C, output_size].

    For max adaptive pool1d:

    ..  math::

       lstart &= floor(i * L_{in} / L_{out})

       lend &= ceil((i + 1) * L_{in} / L_{out})

       Output(i) &= max(Input[lstart:lend])

    Args:
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
             it must contain one int.
        return_mask (bool): If true, the index of max pooling point will be returned along
            with outputs. It cannot be set in average pooling type. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        None.

    Raises:
        ValueError: 'pool_size' should be a integer or list or tuple with length as 1.

    Shape:
        x (Tensor): The input tensor of adaptive max pool1d operator, which is a 3-D tensor. The data type can be float32, float64.
        output (Tensor): The output tensor of adaptive max pool1d operator, which is a 3-D tensor. The data type is same as input x.

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
          import paddle.nn as nn
          import numpy as np

          data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
          AdaptiveMaxPool1D = nn.AdaptiveMaxPool1D(output_size=16)
          pool_out = AdaptiveMaxPool1D(data)
          # pool_out shape: [1, 3, 16]

          # for return_mask = true
          AdaptiveMaxPool1D = nn.AdaptiveMaxPool1D(output_size=16, return_mask=True)
          pool_out, indices = AdaptiveMaxPool1D(data)
          # pool_out shape: [1, 3, 16], indices shape: [1, 3, 16]

    """

    def __init__(self, output_size, return_mask=False, name=None):
        super(AdaptiveMaxPool1D, self).__init__()
        self.output_size = output_size
        self.return_mask = return_mask
        self.name = name

    def forward(self, input):
        return F.adaptive_max_pool1d(input, self.output_size, self.return_mask,
                                     self.name)

    def extra_repr(self):
        return 'output_size={}, return_mask={}'.format(self.output_size,
                                                       self.return_mask)


class AdaptiveMaxPool2D(layers.Layer):
    """
    This operation applies 2D adaptive max pooling on input tensor. The h and w dimensions
    of the output tensor are determined by the parameter output_size. The difference between adaptive pooling and pooling is adaptive one focus on the output size.

    For adaptive max pool2d:

    ..  math::

       hstart &= floor(i * H_{in} / H_{out})

       hend &= ceil((i + 1) * H_{in} / H_{out})

       wstart &= floor(j * W_{in} / W_{out})

       wend &= ceil((j + 1) * W_{in} / W_{out})

       Output(i ,j) &= max(Input[hstart:hend, wstart:wend])

    Parameters:
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list, it must contain two element, (H, W). H and W can be either a int, or None which means the size will be the same as that of the input.
        return_mask (bool): If true, the index of max pooling point will be returned along with outputs. It cannot be set in average pooling type. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Shape:
        x (Tensor): The input tensor of adaptive max pool2d operator, which is a 4-D tensor. The data type can be float32, float64.
        output (Tensor): The output tensor of adaptive max pool2d operator, which is a 4-D tensor. The data type is same as input x.

    Returns:
        A callable object of AdaptiveMaxPool2D.
    Examples:
        .. code-block:: python

            # adaptive max pool2d
            # suppose input data in shape of [N, C, H, W], `output_size` is [m, n],
            # output shape is [N, C, m, n], adaptive pool divide H and W dimensions
            # of input data into m * n grids averagely and performs poolings in each
            # grid to get output.
            # adaptive max pool performs calculations as follow:
            #
            #     for i in range(m):
            #         for j in range(n):
            #             hstart = floor(i * H / m)
            #             hend = ceil((i + 1) * H / m)
            #             wstart = floor(i * W / n)
            #             wend = ceil((i + 1) * W / n)
            #             output[:, :, i, j] = max(input[:, :, hstart: hend, wstart: wend])
            #
            import paddle
            import numpy as np

            input_data = np.random.rand(2, 3, 32, 32)
            x = paddle.to_tensor(input_data)
            adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(output_size=3, return_mask=True)
            pool_out, indices = adaptive_max_pool(x = x)
    """

    def __init__(self, output_size, return_mask=False, name=None):
        super(AdaptiveMaxPool2D, self).__init__()
        self._output_size = output_size
        self._return_mask = return_mask
        self._name = name

    def forward(self, x):
        return F.adaptive_max_pool2d(
            x,
            output_size=self._output_size,
            return_mask=self._return_mask,
            name=self._name)

    def extra_repr(self):
        return 'output_size={}, return_mask={}'.format(self._output_size,
                                                       self._return_mask)


class AdaptiveMaxPool3D(layers.Layer):
    """
    This operation applies 3D adaptive max pooling on input tensor. The h and w dimensions
    of the output tensor are determined by the parameter output_size. The difference between adaptive pooling and pooling is adaptive one focus on the output size.

    For adaptive max pool3d:

    ..  math::

      dstart &= floor(i * D_{in} / D_{out})

      dend &= ceil((i + 1) * D_{in} / D_{out})

      hstart &= floor(j * H_{in} / H_{out})

      hend &= ceil((j + 1) * H_{in} / H_{out})

      wstart &= floor(k * W_{in} / W_{out})

      wend &= ceil((k + 1) * W_{in} / W_{out})

      Output(i ,j, k) &= max(Input[dstart:dend, hstart:hend, wstart:wend])

    Parameters:
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list, it must contain three elements, (D, H, W). D, H and W can be either a int, or None which means the size will be the same as that of the input.
        return_mask (bool): If true, the index of max pooling point will be returned along with outputs. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Shape:
        x (Tensor): The input tensor of adaptive max pool3d operator, which is a 5-D tensor. The data type can be float32, float64.
        output (Tensor): The output tensor of adaptive max pool3d operator, which is a 5-D tensor. The data type is same as input x.
    Returns:
        A callable object of AdaptiveMaxPool3D.
    Examples:
        .. code-block:: python

            # adaptive max pool3d
            # suppose input data in shape of [N, C, D, H, W], `output_size` is [l, m, n],
            # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
            # of input data into l * m * n grids averagely and performs poolings in each
            # grid to get output.
            # adaptive max pool performs calculations as follow:
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
            #                     max(input[:, :, dstart:dend, hstart: hend, wstart: wend])
            import paddle
            import numpy as np

            input_data = np.random.rand(2, 3, 8, 32, 32)
            x = paddle.to_tensor(input_data)
            pool = paddle.nn.AdaptiveMaxPool3D(output_size=4)
            out = pool(x)
            # out shape: [2, 3, 4, 4, 4]
            pool = paddle.nn.AdaptiveMaxPool3D(output_size=3, return_mask=True)
            out, indices = pool(x)
            # out shape: [2, 3, 4, 4, 4], indices shape: [2, 3, 4, 4, 4]

    """

    def __init__(self, output_size, return_mask=False, name=None):
        super(AdaptiveMaxPool3D, self).__init__()
        self._output_size = output_size
        self._return_mask = return_mask
        self._name = name

    def forward(self, x):
        return F.adaptive_max_pool3d(
            x,
            output_size=self._output_size,
            return_mask=self._return_mask,
            name=self._name)

    def extra_repr(self):
        return 'output_size={}, return_mask={}'.format(self._output_size,
                                                       self._return_mask)
