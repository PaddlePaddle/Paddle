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

import sys
import cv2
import random

import numpy as np
import collections

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = [
    "BrightnessTransform",
    "SaturationTransform",
    "ContrastTransform",
    "HueTransform",
    "ColorJitter",
]


class BrightnessTransform(object):
    """Adjust brightness of the image.
    Args:
        value (float): How much to adjust the brightness. Can be any
            non negative number. 0 gives the original image
    Examples:
    
        .. code-block:: python
            import numpy as np
            from paddle.incubate.hapi.vision.transforms import BrightnessTransform
            transform = BrightnessTransform(0.4)
            fake_img = np.random.rand(500, 500, 3).astype('float32')
            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, value):
        if value < 0:
            raise ValueError("brightness value should be non-negative")
        self.value = value

    def __call__(self, img):
        if self.value == 0:
            return img

        dtype = img.dtype
        img = img.astype(np.float32)
        alpha = np.random.uniform(max(0, 1 - self.value), 1 + self.value)
        img = img * alpha
        return img.clip(0, 255).astype(dtype)


class ContrastTransform(object):
    """Adjust contrast of the image.
    Args:
        value (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives the original image
    Examples:
    
        .. code-block:: python
            import numpy as np
            from paddle.incubate.hapi.vision.transforms import ContrastTransform
            transform = ContrastTransform(0.4)
            fake_img = np.random.rand(500, 500, 3).astype('float32')
            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, value):
        if value < 0:
            raise ValueError("contrast value should be non-negative")
        self.value = value

    def __call__(self, img):
        if self.value == 0:
            return img

        dtype = img.dtype
        img = img.astype(np.float32)
        alpha = np.random.uniform(max(0, 1 - self.value), 1 + self.value)
        img = img * alpha + cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() * (
            1 - alpha)
        return img.clip(0, 255).astype(dtype)


class SaturationTransform(object):
    """Adjust saturation of the image.
    Args:
        value (float): How much to adjust the saturation. Can be any
            non negative number. 0 gives the original image
    Examples:
    
        .. code-block:: python
            import numpy as np
            from paddle.incubate.hapi.vision.transforms import SaturationTransform
            transform = SaturationTransform(0.4)
            fake_img = np.random.rand(500, 500, 3).astype('float32')
        
            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, value):
        if value < 0:
            raise ValueError("saturation value should be non-negative")
        self.value = value

    def __call__(self, img):
        if self.value == 0:
            return img

        dtype = img.dtype
        img = img.astype(np.float32)
        alpha = np.random.uniform(max(0, 1 - self.value), 1 + self.value)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img[..., np.newaxis]
        img = img * alpha + gray_img * (1 - alpha)
        return img.clip(0, 255).astype(dtype)


class HueTransform(object):
    """Adjust hue of the image.
    Args:
        value (float): How much to adjust the hue. Can be any number
            between 0 and 0.5, 0 gives the original image
    Examples:
    
        .. code-block:: python
            import numpy as np
            from paddle.incubate.hapi.vision.transforms import HueTransform
            transform = HueTransform(0.4)
            fake_img = np.random.rand(500, 500, 3).astype('float32')
            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, value):
        if value < 0 or value > 0.5:
            raise ValueError("hue value should be in [0.0, 0.5]")
        self.value = value

    def __call__(self, img):
        if self.value == 0:
            return img

        dtype = img.dtype
        img = img.astype(np.uint8)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        h, s, v = cv2.split(hsv_img)

        alpha = np.random.uniform(-self.value, self.value)
        h = h.astype(np.uint8)
        # uint8 addition take cares of rotation across boundaries
        with np.errstate(over="ignore"):
            h += np.uint8(alpha * 255)
        hsv_img = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR_FULL).astype(dtype)


class ColorJitter(object):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    Args:
        brightness: How much to jitter brightness.
            Chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast: How much to jitter contrast.
            Chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation: How much to jitter saturation.
            Chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue: How much to jitter hue.
            Chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    Examples:
    
        .. code-block:: python
            import numpy as np
            from paddle.incubate.hapi.vision.transforms import ColorJitter
            transform = ColorJitter(0.4)
            fake_img = np.random.rand(500, 500, 3).astype('float32')
            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        transforms = []
        if brightness != 0:
            transforms.append(BrightnessTransform(brightness))
        if contrast != 0:
            transforms.append(ContrastTransform(contrast))
        if saturation != 0:
            transforms.append(SaturationTransform(saturation))
        if hue != 0:
            transforms.append(HueTransform(hue))

        random.shuffle(transforms)
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
