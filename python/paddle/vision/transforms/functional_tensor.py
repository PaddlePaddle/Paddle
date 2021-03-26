# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division

import paddle


def normalize(img, mean, std, data_format='CHW'):
    """Normalizes a tensor image with mean and standard deviation.

    Args:
        img (paddle.Tensor): input data to be normalized.
        mean (list|tuple): Sequence of means for each channel.
        std (list|tuple): Sequence of standard deviations for each channel.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        Tensor: Normalized mage.

    """
    assert data_format.lower() in ('chw', 'hwc'
                                   ), "data_format should in ('chw', 'hwc')"

    mean = paddle.to_tensor(mean, place=img.place)
    std = paddle.to_tensor(std, place=img.place)

    if data_format.lower == 'chw':
        mean = mean.reshape([-1, 1, 1])
        std = std.reshape([-1, 1, 1])

    return (img - mean) / std


def to_grayscale(img, num_output_channels=1, data_format='CHW'):
    """Converts image to grayscale version of image.

    Args:
        img (paddel.Tensor): Image to be converted to grayscale.
        num_output_channels (int, optionl[1, 3]):
            if num_output_channels = 1 : returned image is single channel
            if num_output_channels = 3 : returned image is 3 channel 
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        PIL.Image: Grayscale version of the image.
    """
    assert data_format.lower() in ('chw', 'hwc'
                                   ), "data_format should in ('chw', 'hwc')"
    assert num_output_channels in (1, 3), ""

    rgb_weights = paddle.to_tensor(
        [0.2989, 0.5870, 0.1140], place=img.place).astype(img.dtype)

    if data_format.lower() == 'chw':
        _c_index = -3
        rgb_weights = rgb_weights.reshape((-1, 1, 1))
    else:
        _c_index = -1

    img = (img * rgb_weights).sum(axis=_c_index, keepdim=True)
    _shape = img.shape
    _shape[_c_index] = num_output_channels

    return img.expand(_shape)
