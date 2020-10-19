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
    if data_format == 'CHW':
        mean = paddle.to_tensor(mean).reshape([-1, 1, 1])
        std = paddle.to_tensor(std).reshape([-1, 1, 1])
    else:
        mean = paddle.to_tensor(mean)
        std = paddle.to_tensor(std)
    return (img - mean) / std
