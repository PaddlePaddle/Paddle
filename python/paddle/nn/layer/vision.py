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

from ...fluid.dygraph import layers
from .. import functional

__all__ = ['PixelShuffle']


class PixelShuffle(layers.Layer):
    """
        :alias_main: paddle.nn.PixelShuffle
        :alias: paddle.nn.PixelShuffle,paddle.nn.layer.PixelShuffle,paddle.nn.layer.vision.PixelShuffle
    
    PixelShuffle Layer    

    This operator rearranges elements in a tensor of shape [N, C, H, W]
    to a tensor of shape [N, C/upscale_factor**2, H*upscale_factor, W*upscale_factor].
    This is useful for implementing efficient sub-pixel convolution
    with a stride of 1/upscale_factor.
    Please refer to the paper: `Real-Time Single Image and Video Super-Resolution
    Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ .
    by Shi et. al (2016) for more details.

    Parameters:

        upscale_factor(int): factor to increase spatial resolution.

    Shape:
        - x: 4-D tensor with shape: (N, C, H, W).
        - out: 4-D tensor with shape: (N, C/upscale_factor**2, H*upscale_factor, W*upscale_factor).


    Examples:
        .. code-block:: python
            
            import paddle
            import paddle.nn as nn
            import numpy as np

            paddle.enable_imperative()
            x = np.random.randn(2, 9, 4, 4).astype(np.float32)
            place = fluid.CPUPlace()
            x_var = paddle.imperative.to_variable(x)
            pixel_shuffle = nn.PixelShuffle(3)
            y_var = pixel_shuffle(x_var)
            y_np = y_var.numpy()
            print(y_np.shape) # (2, 1, 12, 12)

    """

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self._upscale_factor = upscale_factor

    def forward(self, x):
        return functional.pixel_shuffle(x, self._upscale_factor)
