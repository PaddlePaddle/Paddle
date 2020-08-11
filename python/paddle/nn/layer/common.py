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

# TODO: define the common classes to build a neural network  
from ...fluid.dygraph import BilinearTensorProduct  #DEFINE_ALIAS
from ...fluid.dygraph import Pool2D  #DEFINE_ALIAS
from ...fluid.dygraph import Embedding  #DEFINE_ALIAS
from ...fluid.dygraph import Linear  #DEFINE_ALIAS
from ...fluid.dygraph import Flatten  #DEFINE_ALIAS
from ...fluid.dygraph import layers
from .. import functional as F

__all__ = [
    'BilinearTensorProduct', 'Pool2D', 'Embedding', 'Linear', 'UpSample',
    'Pad2D'
    'AvgPool1D', 'MaxPool1D', 'AdaptiveAvgPool1D', 'AdaptiveMaxPool1D'
]


class UpSample(layers.Layer):
    """
    This op resizes a batch of images.
    The input must be a 3-D Tensor of the shape (num_batches, channels, in_w)
    or 4-D (num_batches, channels, in_h, in_w), or a 5-D Tensor of the shape
    (num_batches, channels, in_d, in_h, in_w) or (num_batches, in_d, in_h, in_w, channels),
    and the resizing only applies on the three dimensions(depth, height and width).
    **Warning:** the parameter :attr:`actual_shape` will be deprecated in the
    future and only use :attr:`out_shape` instead.
    Supporting resample methods:
        'linear' : Linear interpolation
        'bilinear' : Bilinear interpolation
        'trilinear' : Trilinear interpolation
        'nearest' : Nearest neighbor interpolation
        'bicubic' : Bicubic interpolation

    Linear interpolation is the method of using a line connecting two known quantities 
    to determine the value of an unknown quantity between the two known quantities. 
    
    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimension(in height direction) and the 4th dimension(in width
    direction) on input tensor.

    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.
    
    Bicubic interpolation is an extension of cubic interpolation for interpolating
    data points on a two-dimensional regular grid. The interpolated surface is
    smoother than corresponding surfaces obtained by bilinear interpolation or
    nearest-neighbor interpolation.

    Trilinear interpolation is an extension of linear interpolation for
    interpolating functions of three variables (e.g. D-direction,
    H-direction and W-direction in this op) on a rectilinear 3D grid.
    The linear interpolation is performed on three directions.
    Align_corners and align_mode are optional parameters,the calculation method
    of interpolation can be selected by them.

    Example:

    .. code-block:: text

        For scale_factor:
            if align_corners = True && out_size > 1 :
              scale_factor = (in_size-1.0)/(out_size-1.0)
            else:
              scale_factor = float(in_size/out_size)

        Linear interpolation:
            if:
                align_corners = False , align_mode = 0
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = (W_{in}+0.5) * scale_{factor} - 0.5
            else:
                input : (N,C,W_in)
                output: (N,C,W_out) where:
                W_out = W_{in} * scale_{factor}

        Nearest neighbor interpolation:
          if:
              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = floor (H_{in} * scale_{factor})
              W_out = floor (W_{in} * scale_{factor})
          else:
              align_corners = True
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})
        
        Bilinear interpolation:
          if:
              align_corners = False , align_mode = 0

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        Bicubic interpolation:
          if:
              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        Trilinear interpolation:
          if:
              align_corners = False , align_mode = 0
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

    https://en.wikipedia.org/wiki/Linear_interpolation.
    For details of linear interpolation, please refer to Wikipedia:
    
    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.
    
    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation.
    
    For details of bicubic interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bicubic_interpolation
    
    For details of trilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Trilinear_interpolation.
    
    Parameters:
        input (Variable): 3-D, 4-D or 5-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        size (list|tuple|Variable|None): Output shape of image resize
             layer, the shape is (out_w, ) when input is a 3-D Tensor, the shape is (out_h, out_w) 
             when input is a 4-D Tensor and is (out_d, out_h, out_w) when input is a 5-D Tensor. 
             Default: None. If a list, each element can be an integer or a Tensor Variable of shape: [1].
             If a Tensor Variable, its dimensions size should be a 1.
        scale_factor (float|Variable|None): The multiplier for the input height or width. At
             least one of :attr:`out_shape` or :attr:`scale_factor` must be set.
             And :attr:`out_shape` has a higher priority than :attr:`scale_factor`.
             Default: None.
        mode (str): The resample method. It supports 'linear', 'nearst', 'bilinear',
                       'bicubic' and 'trilinear' currently. Default: 'nearest'
        align_corners(bool) :  An optional bool, If True, the centers of the 4 corner pixels of the
                               input and output tensors are aligned, preserving the values at the
                               corner pixels.
                               Default: False
        align_mode(int)  :  An optional for linear/bilinear/trilinear interpolation. Refer to the formula in the example above,
                            it can be \'0\' for src_idx = scale_factor*(dst_indx+0.5)-0.5 , can be \'1\' for
                            src_idx = scale_factor*dst_index.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from:`NCW`, `NWC`, `"NCHW"`, `"NHWC"`, `"NCDHW"`,
            `"NDHWC"`. The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. When it is `"NCHW"`, the data is stored
            in the order of: `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`
    Returns:
        A 3-D Tensor of the shape (num_batches, channels, out_w) or (num_batches, out_w, channels),
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),
        or 5-D Tensor of the shape (num_batches, channels, out_d, out_h, out_w) or (num_batches, out_d, out_h, out_w, channels).
    Raises:
        TypeError: size should be a list or tuple or Variable.
        ValueError: The 'mode' of image_resize can only be 'linear', 'bilinear',
                    'trilinear', 'bicubic', or 'nearest' currently.
        ValueError: 'linear' only support 3-D tensor.
        ValueError: 'bilinear', 'bicubic' and 'nearest' only support 4-D tensor.
        ValueError: 'trilinear' only support 5-D tensor.
        ValueError: One of size and scale_factor must not be None.
        ValueError: size length should be 1 for input 3-D tensor.
        ValueError: size length should be 2 for input 4-D tensor.
        ValueError: size length should be 3 for input 5-D tensor.
        ValueError: scale_factor should be greater than zero.
        TypeError: align_corners should be a bool value
        ValueError: align_mode can only be '0' or '1'
        ValueError: data_format can only be 'NCW', 'NWC', 'NCHW', 'NHWC', 'NCDHW' or 'NDHWC'.

    Examples:
        .. code-block:: python
            import paddle
            import numpy as np
            import paddle.fluid.dygraph as dg
            upsample_op = paddle.nn.UpSample(size=[12,12])
            input_data = np.random.rand(2,3,6,10).astype("float32")
            place = paddle.fluid.CPUPlace()
            with dg.guard(place) as g:
                input = dg.to_variable(input_data)
                output = upsample_op(input=input)
                print(output.shape)
                # [2L, 3L, 12L, 12L]
    """

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=False,
                 align_mode=1,
                 data_format='NCHW'):
        super(UpSample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode.lower()
        self.align_corners = align_corners
        self.align_mode = align_mode
        self.data_format = data_format

    def forward(self, input):
        out = F.interpolate(
            input,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format)

        return out


class Pad2D(layers.Layer):
    """
        :alias_main: paddle.nn.Pad2D
        :alias: paddle.nn.Pad2D,paddle.nn.layer.Pad2D,paddle.nn.layer.common.Pad2D

    This interface is used to construct a callable object of the ``Pad2D``  class.
    The Pad2D layer pads the input tensor boundaries according to 'paddings' and 'mode'.
    If mode is 'reflect', paddings[0] and paddings[1] must be no greater
    than height-1. And the width dimension has the same condition.

    Parameters:
        paddings (int | List[int32]): The padding size. If padding is a int, uses the same 
            padding in all boundaries, if padding is a List, it must contain four integers, 
            (padding_top, padding_bottom, padding_left, padding_right).
            Default is [0, 0, 0, 0].
        mode (str): Three modes: 'constant' (default), 'reflect', 'edge' .
        	When in 'constant' mode, this op uses a constant value to pad the input tensor.
        	When in 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
        	When in 'edge' mode, uses input boundaries to pad the input tensor.
        	Default is 'constant'
        pad_value (float32): The value to fill the padded areas in 'constant' mode . Default is 0.0
        data_format (str): An string from: "NHWC", "NCHW". Specify the data format of
                           the input data.
                           Default is  "NCHW"

    Returns: 
        None

    Examples:
        .. code-block:: text

            Input = [[[[1., 2., 3.],
                       [4., 5., 6.]]]]

            Case 0:
                paddings = [0, 1, 2, 3],
                mode = 'constant'
                pad_value = 0
                Out = [[[[0., 0., 1., 2., 3., 0., 0., 0.],
                         [0., 0., 4., 5., 6., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0.]]]]

            Case 1:
                paddings = [0, 1, 2, 1],
                mode = 'reflect'
                Out = [[[[3., 2., 1., 2., 3., 2.],
                         [6., 5., 4., 5., 6., 5.],
                         [3., 2., 1., 2., 3., 2.]]]]

            Case 2:
                paddings = [0, 1, 2, 1],
                mode = 'edge'
                Out = [[[[1., 1., 1., 2., 3., 3.],
                         [4., 4., 4., 5., 6., 6.],
                         [4., 4., 4., 5., 6., 6.]]]]

    Code Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.nn as nn
            import numpy as np
            data = np.ones((2, 2, 2, 2)).astype('float32')
            my_pad = nn.Pad2D(paddings=[1, 1, 1, 1])
            with fluid.dygraph.guard():
                data = fluid.dygraph.to_variable(data)
                result = my_pad(data)
    """

    def __init__(self,
                 paddings=0,
                 mode='constant',
                 pad_value=0.0,
                 data_format="NCHW"):
        super(Pad2D, self).__init__()
        self._mode = mode
        self._pad_value = pad_value
        self._data_format = data_format
        self._paddings = [paddings] * 4 if isinstance(paddings,
                                                      int) else paddings

    def forward(self, input):
        return F.pad2d(
            input,
            paddings=self._paddings,
            mode=self._mode,
            pad_value=self._pad_value,
            data_format=self._data_format)


class AvgPool1D(layers.Layer):
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
        input (Variable): The input tensor of pooling operator which is a 3-D tensor with
                          shape [N, C, L]. The format of input tensor is `"NCL"` or
                          `"NHL"`, where `N` is batch size, `C` is the number of channels,
                          `L` is the length of the feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain one integers.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain one integers.
        padding (string|int|list|tuple): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool padding size is a tuple or list,
            it could be the following forms: `[pad_left, pad_right]`. If padding is non-zero,
            then the input is implicitly zero-padded on both sides for padding number of points.
        use_cudnn (bool): Only used in cudnn kernel, need install cudnn. Default False
        ceil_mode (bool): ${ceil_mode_comment}Whether to use the ceil function to calculate output height and width.
            If it is set to False, the floor function will be used. Default False
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
        count_include_pad (bool): Whether to exclude padding points in average pooling
                          mode, default is `true`.

    Returns:
        Variable: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        TypeError: If `use_cudnn` is not a bool value.
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple but its length greater than 1.
        ShapeError: If the input is not a 3-D.
        ShapeError: If the output's shape calculated is not greater than 0.


    Examples:

        .. code-block:: python
          import paddle
          import numpy as np
          import paddle.fluid.dygraph as dg
          input_data = np.random.rand(2,3,32).astype("float32")
          avg_pool1d_op = paddle.nn.AvgPool1D(kernel_size=2, stride=2, padding=0)
          place = paddle.fluid.CPUPlace()
          with dg.guard(place) as g:
              input = dg.to_variable(input_data)
              output = avg_pool1d_op(input)
              print(output.shape)
              # [2L, 3L, 16L]
    """

    def __init__(self, kernel_size, stride, padding, ceil_mode,
                 count_include_pad):
        super(AvgPool1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        out = F.avg_pool1d(input, self.kernel_size, self.stride, self.padding,
                           self.ceil_mode, self.count_include_pad)
        return out


class MaxPool1D(layers.Layer):
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
        input (Variable): The input tensor of pooling operator which is a 3-D tensor with
                          shape [N, C, L]. The format of input tensor is `"NCL"` or
                          `"NHL"`, where `N` is batch size, `C` is the number of channels,
                          `L` is the length of the feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain one integers.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain one integers.
        padding (string|int|list|tuple): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool padding size is a tuple or list,
            it could be the following forms: `[pad_left, pad_right]`.
        use_cudnn (bool): Only used in cudnn kernel, need install cudnn. Default False
        ceil_mode (bool): Whether to use the ceil function to calculate output height and width. False is the default.
            If it is set to False, the floor function will be used. Default False
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
        return_indices (bool): Whether return the max indices along with the outputs. default is `False`.

    Returns:
        Variable: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        TypeError: If `use_cudnn` is not a bool value.
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple but its length greater than 1.
        ShapeError: If the input is not a 3-D.
        ShapeError: If the output's shape calculated is not greater than 0.


    Examples:

        .. code-block:: python

          import paddle
          import numpy as np
          import paddle.fluid.dygraph as dg
          input_data = np.random.rand(2,3,32).astype("float32")
          max_pool1d_op = paddle.nn.MaxPool1D(kernel_size=2, stride=2, padding=0)
          place = paddle.fluid.CPUPlace()
          with dg.guard(place) as g:
              input = dg.to_variable(input_data)
              output = max_pool1d_op(input)
              print(output.shape)
              # [2L, 3L, 16L]

          # for return_indices=True
          max_pool1d_op = paddle.nn.MaxPool1D(kernel_size=2, stride=2, padding=0, return_indices=True)
          place = paddle.fluid.CPUPlace()
          with dg.guard(place) as g:
              input = dg.to_variable(input_data)
              output, indices = max_pool1d_op(input)
              print(output.numpy().shape, indices.numpy().shape)
              # [2L, 3L, 16L], [2L, 3L, 16L]

    """

    def __init__(self,
                 kernel_size,
                 stride,
                 padding,
                 ceil_mode,
                 return_indices=False):
        super(MaxPool1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input):
        out = F.max_pool1d(input, self.kernel_size, self.stride, self.padding,
                           self.ceil_mode, self.return_indices)
        return out


class AdaptiveAvgPool1D(layers.Layer):
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
        input (Variable): The input tensor of pooling operator, which is a 3-D tensor
                          with shape [N, C, L].  The format of input tensor is NCL,
                          where N is batch size, C is the number of channels, L is the
                          length of the feature. The data type is float32 or float64.
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain one int.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Variable: The output tensor of adaptive average pooling result. The data type is same
                  as input tensor.

    Raises:
        ValueError: 'pool_size' should be a integer or list or tuple with length as 1.

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
          import numpy as np
          import paddle.fluid.dygraph as dg
          input_data = np.random.rand(2,3,32).astype("float32")
          adaptive_avg_pool1d_op = paddle.nn.AdaptiveAvgPool1D(output_size=12)
          place = paddle.fluid.CPUPlace()
          with dg.guard(place) as g:
              input = dg.to_variable(input_data)
              output = adaptive_avg_pool1d_op(input)
              print(output.shape)
              # [2L, 3L, 12L]
    """

    def __init__(self, output_size):
        super(AdaptiveAvgPool1D, self).__init__()
        self.output_size = output_size

    def forward(self, input):
        return F.adaptive_avg_pool1d(input, self.output_size)


class AdaptiveMaxPool1D(layers.Layer):
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
        input (Variable): The input tensor of pooling operator, which is a 3-D tensor
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
        Variable: The output tensor of adaptive pooling result. The data type is same
                  as input tensor.

    Raises:
        ValueError: 'pool_size' should be a integer or list or tuple with length as 1.

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
          import numpy as np
          import paddle.fluid.dygraph as dg
          input_data = np.random.rand(2,3,32).astype("float32")
          adaptive_max_pool1d_op1 = paddle.nn.AdaptiveMaxPool1D(output_size=12)
          place = paddle.fluid.CPUPlace()
          with dg.guard(place) as g:
              input = dg.to_variable(input_data)
              output = adaptive_max_pool1d_op1(input)
              print(output.shape)
              # [2L, 3L, 12L]

          adaptive_max_pool1d_op2 = paddle.nn.AdaptiveMaxPool1D(output_size=12, return_indices=True)
          place = paddle.fluid.CPUPlace()
          with dg.guard(place) as g:
              input = dg.to_variable(input_data)
              output, indices = adaptive_max_pool1d_op2(input)
              print(output.shape, indices.shape)
              # [2L, 3L, 12L], [2L, 3L, 12L]

    """

    def __init__(self, output_size, return_indices=False):
        super(AdaptiveMaxPool1D, self).__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, input):
        return F.adaptive_max_pool1d(input, self.output_size,
                                     self.return_indices)
