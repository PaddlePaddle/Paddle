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

# TODO: define specitial functions used in computer vision task

from .. import Layer
from .. import functional

__all__ = []


class PixelShuffle(Layer):
    """
    
    PixelShuffle Layer    

    Rearranges elements in a tensor of shape :math:`[N, C, H, W]`
    to a tensor of shape :math:`[N, C/upscale_factor^2, H*upscale_factor, W \times upscale_factor]`,
    or from shape :math:`[N, H, W, C]` to :math:`[N, H \times upscale_factor, W \times upscale_factor, C/upscale_factor^2]`.
    This is useful for implementing efficient sub-pixel convolution
    with a stride of 1/upscale_factor.
    Please refer to the paper: `Real-Time Single Image and Video Super-Resolution
    Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ .
    by Shi et. al (2016) for more details.

    Parameters:

        upscale_factor(int): factor to increase spatial resolution.
        data_format (str, optional): The data format of the input and output data. An optional string from: "NCHW", "NHWC". The default is "NCHW". When it is "NCHW", the data is stored in the order of: [batch_size, input_channels, input_height, input_width].
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - x: 4-D tensor with shape of :math:`(N, C, H, W)` or :math:`(N, H, W, C)`.
        - out: 4-D tensor with shape of :math:`(N, C/upscale_factor^2, H \times upscale_factor, W \times upscale_factor)` or :math:`(N, H \times upscale_factor, W \times upscale_factor, C/upscale_factor^2)`.


    Examples:
        .. code-block:: python
            
            import paddle
            import paddle.nn as nn

            x = paddle.randn(shape=[2,9,4,4])
            pixel_shuffle = nn.PixelShuffle(3)
            out_var = pixel_shuffle(x)
            out = out_var.numpy()
            print(out.shape)
            # (2, 1, 12, 12)

    """

    def __init__(self, upscale_factor, data_format="NCHW", name=None):
        super(PixelShuffle, self).__init__()

        if not isinstance(upscale_factor, int):
            raise TypeError("upscale factor must be int type")

        if data_format not in ["NCHW", "NHWC"]:
            raise ValueError("Data format should be 'NCHW' or 'NHWC'."
                             "But recevie data format: {}".format(data_format))

        self._upscale_factor = upscale_factor
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return functional.pixel_shuffle(x, self._upscale_factor,
                                        self._data_format, self._name)

    def extra_repr(self):
        main_str = 'upscale_factor={}'.format(self._upscale_factor)
        if self._data_format != 'NCHW':
            main_str += ', data_format={}'.format(self._data_format)
        if self._name is not None:
            main_str += ', name={}'.format(self._name)
        return main_str


class PixelUnshuffle(Layer):
    """
    Rearranges elements in a tensor of shape :math:`[N, C, H, W]`
    to a tensor of shape :math:`[N, r^2C, H/r, W/r]`, or from shape 
    :math:`[N, H, W, C]` to :math:`[N, H/r, W/r, r^2C]`, where :math:`r` is the 
    downscale factor. This operation is the reversion of PixelShuffle operation.
    Please refer to the paper: `Real-Time Single Image and Video Super-Resolution
    Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ .
    by Shi et. al (2016) for more details.

    Parameters:
        downscale_factor (int): Factor to decrease spatial resolution.
        data_format (str, optional): The data format of the input and output data. An optional string of NCHW or NHWC. The default is NCHW. When it is NCHW, the data is stored in the order of [batch_size, input_channels, input_height, input_width].
        name (str, optional): Name for the operation (optional, default is None). Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - **x**: 4-D tensor with shape of :math:`[N, C, H, W]` or :math:`[N, C, H, W]`.
        - **out**: 4-D tensor with shape of :math:`[N, r^2C, H/r, W/r]` or :math:`[N, H/r, W/r, r^2C]`, where :math:`r` is :attr:`downscale_factor`.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            x = paddle.randn([2, 1, 12, 12])
            pixel_unshuffle = nn.PixelUnshuffle(3)
            out = pixel_unshuffle(x)
            print(out.shape)
            # [2, 9, 4, 4]

    """

    def __init__(self, downscale_factor, data_format="NCHW", name=None):
        super(PixelUnshuffle, self).__init__()

        if not isinstance(downscale_factor, int):
            raise TypeError("Downscale factor must be int type")

        if downscale_factor <= 0:
            raise ValueError("Downscale factor must be positive")

        if data_format not in ["NCHW", "NHWC"]:
            raise ValueError("Data format should be 'NCHW' or 'NHWC'."
                             "But recevie data format: {}".format(data_format))

        self._downscale_factor = downscale_factor
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return functional.pixel_unshuffle(x, self._downscale_factor,
                                          self._data_format, self._name)

    def extra_repr(self):
        main_str = 'downscale_factor={}'.format(self._downscale_factor)
        if self._data_format != 'NCHW':
            main_str += ', data_format={}'.format(self._data_format)
        if self._name is not None:
            main_str += ', name={}'.format(self._name)
        return main_str


class ChannelShuffle(Layer):
    """
    This operator divides channels in a tensor of shape [N, C, H, W] or [N, H, W, C] into g groups,
    getting a tensor with the shape of [N, g, C/g, H, W] or [N, H, W, g, C/g], and transposes them
    as [N, C/g, g, H, W] or [N, H, W, g, C/g], then rearranges them to original tensor shape. This
    operation can improve the interaction between channels, using features efficiently. Please 
    refer to the paper: `ShuffleNet: An Extremely Efficient 
    Convolutional Neural Network for Mobile Devices <https://arxiv.org/abs/1707.01083>`_ .
    by Zhang et. al (2017) for more details. 

    Parameters:
        groups (int): Number of groups to divide channels in.
        data_format (str): The data format of the input and output data. An optional string of NCHW or NHWC. The default is NCHW. When it is NCHW, the data is stored in the order of [batch_size, input_channels, input_height, input_width].
        name (str, optional): Name for the operation (optional, default is None). Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - **x**: 4-D tensor with shape of [N, C, H, W] or [N, H, W, C].
        - **out**: 4-D tensor with shape and dtype same as x.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            x = paddle.arange(0, 0.6, 0.1, 'float32')
            x = paddle.reshape(x, [1, 6, 1, 1])
            # [[[[0.        ]],
            #   [[0.10000000]],
            #   [[0.20000000]],
            #   [[0.30000001]],
            #   [[0.40000001]],
            #   [[0.50000000]]]]
            channel_shuffle = nn.ChannelShuffle(3)
            y = channel_shuffle(x)
            # [[[[0.        ]],
            #   [[0.20000000]],
            #   [[0.40000001]],
            #   [[0.10000000]],
            #   [[0.30000001]],
            #   [[0.50000000]]]]
    """

    def __init__(self, groups, data_format="NCHW", name=None):
        super(ChannelShuffle, self).__init__()

        if not isinstance(groups, int):
            raise TypeError("groups must be int type")

        if groups <= 0:
            raise ValueError("groups must be positive")

        if data_format not in ["NCHW", "NHWC"]:
            raise ValueError("Data format should be 'NCHW' or 'NHWC'."
                             "But recevie data format: {}".format(data_format))

        self._groups = groups
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return functional.channel_shuffle(x, self._groups, self._data_format,
                                          self._name)

    def extra_repr(self):
        main_str = 'groups={}'.format(self._groups)
        if self._data_format != 'NCHW':
            main_str += ', data_format={}'.format(self._data_format)
        if self._name is not None:
            main_str += ', name={}'.format(self._name)
        return main_str
