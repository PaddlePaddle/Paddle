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

import sys
import collections
import random

import cv2
import numpy as np

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = ['flip', 'resize']


def flip(image, code):
    """
    Accordding to the code (the type of flip), flip the input image

    Args:
        image: Input image, with (H, W, C) shape
        code: Code that indicates the type of flip.
            -1 : Flip horizontally and vertically
            0 : Flip vertically
            1 : Flip horizontally

    Examples:
        .. code-block:: python

            import numpy as np
            from paddle.incubate.hapi.vision.transforms import functional as F

            fake_img = np.random.rand(224, 224, 3)

            # flip horizontally and vertically
            F.flip(fake_img, -1)

            # flip vertically
            F.flip(fake_img, 0)

            # flip horizontally
            F.flip(fake_img, 1)
    """
    return cv2.flip(image, flipCode=code)


def resize(img, size, interpolation=cv2.INTER_LINEAR):
    """
    resize the input data to given size

    Args:
        input: Input data, could be image or masks, with (H, W, C) shape
        size: Target size of input data, with (height, width) shape.
        interpolation: Interpolation method.

    Examples:
        .. code-block:: python

            import numpy as np
            from paddle.incubate.hapi.vision.transforms import functional as F

            fake_img = np.random.rand(256, 256, 3)

            F.resize(fake_img, 224)

            F.resize(fake_img, (200, 150))
    """

    if isinstance(interpolation, Sequence):
        interpolation = random.choice(interpolation)

    if isinstance(size, int):
        h, w = img.shape[:2]
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, (ow, oh), interpolation=interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, (ow, oh), interpolation=interpolation)
    else:
        return cv2.resize(img, size[::-1], interpolation=interpolation)
