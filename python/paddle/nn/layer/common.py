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
from ...fluid.framework import _dygraph_tracer

__all__ = [
    'BilinearTensorProduct', 'Pool2D', 'Embedding', 'Linear', 'UpSample',
    'Pad2D', 'ReflectionPad1d', 'ReplicationPad1d', 'ConstantPad1d',
    'ReflectionPad2d', 'ReplicationPad2d', 'ConstantPad2d', 'ZeroPad2d',
    'ConstantPad3d', 'ReplicationPad3d', 'CosineSimilarity', 'Dropout',
    'Dropout2D', 'Dropout3D', 'AlphaDropout'
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


class Dropout(layers.Layer):
    """
    Dropout is a regularization technique for reducing overfitting by preventing
    neuron co-adaption during training as described in the paper:
    `Improving neural networks by preventing co-adaptation of feature detectors <https://arxiv.org/abs/1207.0580>`_ 
    The dropout operator randomly sets the outputs of some units to zero, while upscale others
    according to the given dropout probability.

    See ``paddle.nn.functional.dropout`` for more details.

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float | int): Probability of setting units to zero. Default: 0.5
        axis (int | list): The axis along which the dropout is performed. Default None.
        mode(str, optional): ['upscale_in_train'(default) | 'downscale_in_infer']

                               1. upscale_in_train(default), upscale the output at training time

                                  - train: out = input * mask / ( 1.0 - p )
                                  - inference: out = input

                               2. downscale_in_infer, downscale the output at inference

                                  - train: out = input * mask
                                  - inference: out = input * (1.0 - p)
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: N-D tensor.
        - output: N-D tensor, the same shape as input.

    Examples:
        .. code-block:: python
            import paddle
            import numpy as np

            paddle.disable_static()
            x = np.array([[1,2,3], [4,5,6]]).astype('float32')
            x = paddle.to_tensor(x)
            m = paddle.nn.Dropout(p=0.5)
            y_train = m(x)
            m.eval()  # switch the model to test phase
            y_test = m(x)
            print(x.numpy())
            print(y_train.numpy())
            print(y_test.numpy())
   """

    def __init__(self, p=0.5, axis=None, mode="upscale_in_train", name=None):
        super(Dropout, self).__init__()

        self.p = p
        self.axis = axis
        self.mode = mode
        self.name = name

    def forward(self, input):
        out = F.dropout(
            input,
            p=self.p,
            axis=self.axis,
            training=self.training,
            mode=self.mode,
            name=self.name)
        return out


class Dropout2D(layers.Layer):
    """
    Randomly zero out entire channels (in the batched input 4d tensor with the shape `NCHW` ,
    a channel is a 2D feature map with the shape `HW`). Each channel will be zeroed out independently
    on every forward call with probability `p` using samples from a Bernoulli distribution.
    Dropout2d will help promote independence between feature maps as described in the paper: 
    `Efficient Object Localization Using Convolutional Networks <https://arxiv.org/abs/1411.4280>`_ 

    See ``paddle.nn.functional.dropout2d`` for more details.

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float, optional): Probability of setting units to zero. Default: 0.5
        data_format (str, optional): Specify the data format of the input, and the data format of the output
                                     will be consistent with that of the input. An optional string from:
                                    `NCHW`, `NHWC`. The default is `NCHW`. When it is `NCHW`, the data is
                                     stored in the order of: [batch_size, input_channels, input_height, input_width].
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: 4-D tensor.
        - output: 4-D tensor, the same shape as input.

    Examples:
        .. code-block:: python
            import paddle
            import numpy as np

            paddle.disable_static()
            x = np.random.random(size=(2, 3, 4, 5)).astype('float32')
            x = paddle.to_tensor(x)
            m = paddle.nn.Dropout2D(p=0.5)
            y_train = m(x)
            m.eval()  # switch the model to test phase
            y_test = m(x)
            print(x.numpy())
            print(y_train.numpy())
            print(y_test.numpy())
   """

    def __init__(self, p=0.5, data_format='NCHW', name=None):
        super(Dropout2D, self).__init__()

        self.p = p
        self.data_format = data_format
        self.name = name

    def forward(self, input):
        out = F.dropout2d(
            input,
            p=self.p,
            training=self.training,
            data_format=self.data_format,
            name=self.name)
        return out


class Dropout3D(layers.Layer):
    """
    Randomly zero out entire channels (in the batched input 5d tensor with the shape `NCDHW` ,
    a channel is a 3D feature map with the shape `DHW` ). Each channel will be zeroed out independently
    on every forward call with probability `p` using samples from a Bernoulli distribution.
    Dropout3d will help promote independence between feature maps as described in the paper: 
    `Efficient Object Localization Using Convolutional Networks <https://arxiv.org/abs/1411.4280>`_ 

    See ``paddle.nn.functional.dropout3d`` for more details.

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float | int): Probability of setting units to zero. Default: 0.5
        data_format (str, optional): Specify the data format of the input, and the data format of the output
                                     will be consistent with that of the input. An optional string from:
                                    `NCDHW`, `NDHWC`. The default is `NCDHW`. When it is `NCDHW`, the data is
                                     stored in the order of: [batch_size, input_channels, input_depth, input_height, input_width].
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: 5-D tensor.
        - output: 5-D tensor, the same shape as input.

    Examples:
        .. code-block:: python
            import paddle
            import numpy as np

            paddle.disable_static()
            x = np.random.random(size=(2, 3, 4, 5, 6)).astype('float32')
            x = paddle.to_tensor(x)
            m = paddle.nn.Dropout3D(p=0.5)
            y_train = m(x)
            m.eval()  # switch the model to test phase
            y_test = m(x)
            print(x.numpy())
            print(y_train.numpy())
            print(y_test.numpy())
   """

    def __init__(self, p=0.5, data_format='NCDHW', name=None):
        super(Dropout3D, self).__init__()

        self.p = p
        self.data_format = data_format
        self.name = name

    def forward(self, input):
        out = F.dropout3d(
            input,
            p=self.p,
            training=self.training,
            data_format=self.data_format,
            name=self.name)
        return out


class AlphaDropout(layers.Layer):
    """
    Alpha Dropout is a type of Dropout that maintains the self-normalizing property. For an input with
    zero mean and unit standard deviation, the output of Alpha Dropout maintains the original mean and
    standard deviation of the input. Alpha Dropout fits well to SELU activate function by randomly setting
    activations to the negative saturation value.

    For more information, please refer to:
    `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float | int): Probability of setting units to zero. Default: 0.5
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: N-D tensor.
        - output: N-D tensor, the same shape as input.

    Examples:
        .. code-block:: python
            import paddle
            import numpy as np

            paddle.disable_static()
            x = np.array([[-1, 1], [-1, 1]]).astype('float32')
            x = paddle.to_tensor(x)
            m = paddle.nn.AlphaDropout(p=0.5)
            y_train = m(x)
            m.eval()  # switch the model to test phase
            y_test = m(x)
            print(x.numpy())
            print(y_train.numpy())
            # [[-0.10721093, 1.6655989 ], [-0.7791938, -0.7791938]] (randomly)
            print(y_test.numpy())
   """

    def __init__(self, p=0.5, name=None):
        super(AlphaDropout, self).__init__()
        self.p = p
        self.name = name

    def forward(self, input):
        out = F.alpha_dropout(
            input, p=self.p, training=self.training, name=self.name)
        return out


class ReflectionPad1d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ReflectionPad1d`` class.
    Uses reflection of the input boundaries to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right).
        data_format (str): An string from: "NCL", "NLC". Specify the data format of the input data.
           Default is  "NCL"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
    Returns: 
        None

    Examples:
        .. code-block:: text

            x = [[[1., 2., 3.],
                  [4., 5., 6.]]]
            padding = [1, 2],
            Out = [[[2. 1. 2. 3. 2. 1.]
                    [5. 4. 5. 6. 5. 4.]]]

    Code Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 2, 3)
            pad = [1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ReflectionPad1d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[2. 1. 2. 3. 2. 1.]
            #   [5. 4. 5. 6. 5. 4.]]]
    """

    def __init__(self, padding, data_format="NCL", name=None):
        super(ReflectionPad1d, self).__init__()
        self._mode = "reflect"
        self._data_format = data_format
        self._pad = padding
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     data_format=self._data_format,
                     name=self._name)


class ReplicationPad1d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ReplicationPad1d`` class.
    Uses input boundaries to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right).
        data_format (str): An string from: "NCL", "NLC". Specify the data format of the input data.
           Default is  "NCL"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
    Returns: 
        None

    Examples:
        .. code-block:: text

            x = [[[1., 2., 3.],
                  [4., 5., 6.]]]
            padding = [1, 2],
            Out = [[[2. 1. 2. 3. 2. 1.]
                    [5. 4. 5. 6. 5. 4.]]]

    Code Examples:
        .. code-block:: python
        
            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 2, 3)
            pad = [1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ReplicationPad1d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[1. 1. 2. 3. 3. 3.]
            #   [1. 4. 5. 6. 6. 6.]]]
    """

    def __init__(self, padding, data_format="NCL", name=None):
        super(ReplicationPad1d, self).__init__()
        self._mode = "replicate"
        self._data_format = data_format
        self._pad = padding
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     data_format=self._data_format,
                     name=self._name)


class ConstantPad1d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ConstantPad1d`` class.
    Uses a constant value to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right).
        value (float32): The value to fill the padded areas. Default is 0.0
        data_format (str): An string from: "NCL", "NLC". Specify the data format of the input data.
           Default is  "NCL"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
    Returns: 
        None

    Examples:
        .. code-block:: text

            x = [[[1., 2., 3.],
                  [4., 5., 6.]]]
            padding = [1, 2],
            value = 0.0
            Out = [[[0. 1. 2. 3. 0. 0.]
                    [0. 4. 5. 6. 0. 0.]]]

    Code Examples:
        .. code-block:: python
        
            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 2, 3)
            pad = [1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ConstantPad1d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[0. 1. 2. 3. 0. 0.]
            #   [0. 4. 5. 6. 0. 0.]]]
    """

    def __init__(self, padding, value=0.0, data_format="NCL", name=None):
        super(ConstantPad1d, self).__init__()
        self._mode = "constant"
        self._data_format = data_format
        self._pad = padding
        self._value = value
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     value=self._value,
                     data_format=self._data_format,
                     name=self._name)


class ConstantPad2d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ConstantPad2d`` class.
    Uses a constant value to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom).
        value (float32): The value to fill the padded areas. Default is 0.0
        data_format (str): An string from: "NCHW", "NHWC". Specify the data format of the input data.
           Default is  "NCHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
    Returns: 
        None

    Examples:
        .. code-block:: text

            x = [[[[1., 2., 3.],
                   [4., 5., 6.]]]]
            padding = [1, 1, 0, 0]
            value = 0.0
            Out = [[[[0. 1. 2. 3. 0.]
                     [0. 4. 5. 6. 0.]]]]

    Code Examples:
        .. code-block:: python
        
            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 1, 2, 3)
            pad = [1, 0, 1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ConstantPad2d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[[0. 0. 0. 0.]
            #    [0. 1. 2. 3.]
            #    [0. 4. 5. 6.]
            #    [0. 0. 0. 0.]
            #    [0. 0. 0. 0.]]]]
    """

    def __init__(self, padding, value=0.0, data_format="NCHW", name=None):
        super(ConstantPad2d, self).__init__()
        self._mode = "constant"
        self._data_format = data_format
        self._pad = padding
        self._value = value
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     value=self._value,
                     data_format=self._data_format,
                     name=self._name)


class ZeroPad2d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ZeroPad2d`` class.
    Uses 0 to pad the input tensor.

    Parameters:
        padding (Variable | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom).
        data_format (str): An string from: "NCHW", "NHWC". Specify the data format of the input data.
           Default is  "NCHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
    Returns: 
        None

    Examples:
        .. code-block:: text

            x = [[[[1., 2., 3.],
                   [4., 5., 6.]]]]
            padding = [1, 1, 0, 0]
            Out = [[[[0. 1. 2. 3. 0.]
                     [0. 4. 5. 6. 0.]]]]

    Code Examples:
        .. code-block:: python
        
            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 1, 2, 3)
            pad = [1, 0, 1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ZeroPad2d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[[0. 0. 0. 0.]
            #    [0. 1. 2. 3.]
            #    [0. 4. 5. 6.]
            #    [0. 0. 0. 0.]
            #    [0. 0. 0. 0.]]]]
    """

    def __init__(self, padding, data_format="NCHW", name=None):
        super(ZeroPad2d, self).__init__()
        self._mode = "constant"
        self._data_format = data_format
        self._pad = padding
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     data_format=self._data_format,
                     name=self._name)


class ReplicationPad2d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ReplicationPad2d`` class.
    Uses input boundaries to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom).
        data_format (str): An string from: "NCHW", "NHWC". Specify the data format of the input data.
           Default is  "NCHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
    Returns: 
        None

    Examples:
        .. code-block:: text

            x = [[[[1., 2., 3.],
                   [4., 5., 6.]]]]
            padding = [1, 1, 0, 0]
            Out = [[[[1. 1. 2. 3. 3.]
                     [4. 4. 5. 6. 6.]]]]

    Code Examples:
        .. code-block:: python
        
            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 1, 2, 3)
            pad = [1, 0, 1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ReplicationPad2d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[[1. 1. 2. 3.]
            #    [1. 1. 2. 3.]
            #    [4. 4. 5. 6.]
            #    [4. 4. 5. 6.]
            #    [4. 4. 5. 6.]]]]
    """

    def __init__(self, padding, data_format="NCHW", name=None):
        super(ReplicationPad2d, self).__init__()
        self._mode = "replicate"
        self._data_format = data_format
        self._pad = padding
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     data_format=self._data_format,
                     name=self._name)


class ReflectionPad2d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ReflectionPad2d`` class.
    Uses reflection of the input boundaries to pad the input tensor.

    Parameters:
        padding (Variable | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom).
        data_format (str): An string from: "NCHW", "NHWC". Specify the data format of the input data.
           Default is  "NCHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
    Returns: 
        None

    Examples:
        .. code-block:: text

            x = [[[[1., 2., 3.],
                   [4., 5., 6.]]]]
            padding = [1, 1, 0, 0]
            Out = [[[[2. 1. 2. 3. 2.]
                     [5. 4. 5. 6. 5.]]]]

    Code Examples:
        .. code-block:: python
        
            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 1, 4, 3)
            pad = [1, 0, 1, 2]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ReflectionPad2d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[[ 5.  4.  5.  6.]
            #    [ 2.  1.  2.  3.]
            #    [ 5.  4.  5.  6.]
            #    [ 8.  7.  8.  9.]
            #    [11. 10. 11. 12.]
            #    [ 8.  7.  8.  9.]
            #    [ 5.  4.  5.  6.]]]]
    """

    def __init__(self, padding, data_format="NCHW", name=None):
        super(ReflectionPad2d, self).__init__()
        self._mode = "reflect"
        self._data_format = data_format
        self._pad = padding
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     data_format=self._data_format,
                     name=self._name)


class ConstantPad3d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ConstantPad3d`` class.
    Uses a constant value to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
        value (float32): The value to fill the padded areas. Default is 0.0
        data_format (str): An string from: "NCDHW", "NDHWC". Specify the data format of the input data.
           Default is  "NCDHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
    Returns: 
        None

    Examples:
        .. code-block:: text

            x = [[[[[1., 2., 3.],
                    [4., 5., 6.]]]]]
            padding = [1, 2, 0, 0, 0, 0]
            value = 0.0
            Out = [[[[[0. 1. 2. 3. 0. 0.]
                      [0. 4. 5. 6. 0. 0.]]]]]

    Code Examples:
        .. code-block:: python
        
            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 1, 1, 2, 3)
            pad = [1, 0, 1, 2, 0, 0]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ConstantPad3d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[[[0. 0. 0. 0.]
            #     [0. 1. 2. 3.]
            #     [0. 4. 5. 6.]
            #     [0. 0. 0. 0.]
            #     [0. 0. 0. 0.]]]]]
    """

    def __init__(self, padding, value=0.0, data_format="NCDHW", name=None):
        super(ConstantPad3d, self).__init__()
        self._mode = "constant"
        self._data_format = data_format
        self._pad = padding
        self._value = value
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     value=self._value,
                     data_format=self._data_format,
                     name=self._name)


class ReplicationPad3d(layers.Layer):
    """
    This interface is used to construct a callable object of the ``ReplicationPad3d`` class.
    Uses input boundaries to pad the input tensor.

    Parameters:
        padding (Tensor | List[int32]): The padding size with data type int32. [len(padding)/2] dimensions
            of input will be padded. The pad has the form (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back).
        data_format (str): An string from: "NCDHW", "NDHWC". Specify the data format of the input data.
           Default is  "NCDHW"
        name (str, optional) : The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
    Returns: 
        None

    Examples:
        .. code-block:: text

            x = [[[[[1., 2., 3.],
                    [4., 5., 6.]]]]]
            padding = [1, 2, 0, 0, 0, 0]
            Out = [[[[[1. 1. 2. 3. 3. 3.]
                      [4. 4. 5. 6. 6. 6.]]]]]

    Code Examples:
        .. code-block:: python
        
            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            input_shape = (1, 1, 1, 2, 3)
            pad = [1, 0, 1, 2, 0, 0]
            data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) + 1
            my_pad = nn.ReplicationPad3d(padding=pad)
            data = paddle.to_tensor(data)
            result = my_pad(data)
            print(result.numpy())
            # [[[[[1. 1. 2. 3.]
            #     [1. 1. 2. 3.]
            #     [4. 4. 5. 6.]
            #     [4. 4. 5. 6.]
            #     [4. 4. 5. 6.]]]]]
    """

    def __init__(self, padding, data_format="NCDHW", name=None):
        super(ReplicationPad3d, self).__init__()
        self._mode = "replicate"
        self._data_format = data_format
        self._pad = padding
        self._name = name

    def forward(self, x):
        return F.pad(x,
                     pad=self._pad,
                     mode=self._mode,
                     data_format=self._data_format,
                     name=self._name)


class CosineSimilarity(layers.Layer):
    """
    This interface is used to compute cosine similarity between x1 and x2 along axis.

    Parameters:
        axis (int): Dimension of vectors to compute cosine similarity. Default is 1.
        eps(float): Small value to avoid division by zero. Default is 1e-8.
    Returns: 
        None

    Examples:
        .. code-block:: text

            Case 0:
                x1 = [[0.8024077  0.9927354  0.27238318 0.8344984 ]
                     [0.48949873 0.5797396  0.65444374 0.66510963]
                     [0.1031398  0.9614342  0.08365563 0.6796464 ]
                     [0.10760343 0.7461209  0.7726148  0.5801006 ]]
                x2 = [[0.62913156 0.1536727  0.9847992  0.04591406]
                     [0.9098952  0.15715368 0.8671125  0.3156102 ]
                     [0.4427798  0.54136837 0.5276275  0.32394758]
                     [0.3769419  0.8535014  0.48041078 0.9256797 ]]
                axis = 1
                eps = 1e-8
                Out: [0.5275037  0.8368967  0.75037485 0.9245899]

    Code Examples:
        .. code-block:: python
        
            import paddle
            import paddle.nn as nn
            import numpy as np
            paddle.disable_static()

            np.random.seed(0)
            x1 = np.random.rand(2,3)
            x2 = np.random.rand(2,3)
            x1 = paddle.to_tensor(x1)
            x2 = paddle.to_tensor(x2)

            cos_sim_func = nn.CosineSimilarity(axis=0)
            result = cos_sim_func(x1, x2)
            print(result.numpy())
            # [0.99806249 0.9817672  0.94987036]
    """

    def __init__(self, axis=1, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self._axis = axis
        self._eps = eps

    def forward(self, x1, x2):
        return F.cosine_similarity(x1, x2, axis=self._axis, eps=self._eps)
