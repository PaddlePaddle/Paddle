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
import math
import functools

import cv2
import numbers
import numpy as np

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = ['flip', 'resize', 'pad', 'rotate', 'to_grayscale']


def keepdims(func):
    """Keep the dimension of input images unchanged"""

    @functools.wraps(func)
    def wrapper(image, *args, **kwargs):
        if len(image.shape) != 3:
            raise ValueError("Expect image have 3 dims, but got {} dims".format(
                len(image.shape)))
        ret = func(image, *args, **kwargs)
        if len(ret.shape) == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    return wrapper


@keepdims
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
            from paddle.vision.transforms import functional as F

            fake_img = np.random.rand(224, 224, 3)

            # flip horizontally and vertically
            F.flip(fake_img, -1)

            # flip vertically
            F.flip(fake_img, 0)

            # flip horizontally
            F.flip(fake_img, 1)
    """
    return cv2.flip(image, flipCode=code)


@keepdims
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
            from paddle.vision.transforms import functional as F

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


@keepdims
def pad(img, padding, fill=(0, 0, 0), padding_mode='constant'):
    """Pads the given CV Image on all sides with speficified padding mode and fill value.

    Args:
        img (np.ndarray): Image to be padded.
        padding (int|tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int|tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            ``constant`` means padding with a constant value, this value is specified with fill. 
            ``edge`` means padding with the last value at the edge of the image. 
            ``reflect`` means padding with reflection of image (without repeating the last value on the edge) 
            padding ``[1, 2, 3, 4]`` with 2 elements on both sides in reflect mode 
            will result in ``[3, 2, 1, 2, 3, 4, 3, 2]``.
            ``symmetric`` menas pads with reflection of image (repeating the last value on the edge)
            padding ``[1, 2, 3, 4]`` with 2 elements on both sides in symmetric mode 
            will result in ``[2, 1, 1, 2, 3, 4, 4, 3]``.

    Returns:
        numpy ndarray: Padded image.

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.vision.transforms.functional import pad

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = pad(fake_img, 2)
            print(fake_img.shape)

    """

    if not isinstance(padding, (numbers.Number, list, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, list, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError(
            "Padding must be an int or a 2, or 4 element tuple, not a " +
            "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Expected padding mode be either constant, edge, reflect or symmetric, but got {}'.format(padding_mode)

    PAD_MOD = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_DEFAULT,
        'symmetric': cv2.BORDER_REFLECT
    }

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left, pad_top, pad_right, pad_bottom = padding

    if isinstance(fill, numbers.Number):
        fill = (fill, ) * (2 * len(img.shape) - 3)

    if padding_mode == 'constant':
        assert (len(fill) == 3 and len(img.shape) == 3) or (len(fill) == 1 and len(img.shape) == 2), \
            'channel of image is {} but length of fill is {}'.format(img.shape[-1], len(fill))

    img = cv2.copyMakeBorder(
        src=img,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=PAD_MOD[padding_mode],
        value=fill)

    return img


@keepdims
def rotate(img,
           angle,
           interpolation=cv2.INTER_LINEAR,
           expand=False,
           center=None):
    """Rotates the image by angle.

    Args:
        img (numpy.ndarray): Image to be rotated.
        angle (float|int): In degrees clockwise order.
        interpolation (int, optional):
            interpolation: Interpolation method.
        expand (bool|optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple|optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.

    Returns:
        numpy ndarray: Rotated image.

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.vision.transforms.functional import rotate

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = rotate(fake_img, 10)
            print(fake_img.shape)
    """
    dtype = img.dtype

    h, w, _ = img.shape
    point = center or (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(point, angle=-angle, scale=1)

    if expand:
        if center is None:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            M[0, 2] += (nW / 2) - point[0]
            M[1, 2] += (nH / 2) - point[1]

            dst = cv2.warpAffine(img, M, (nW, nH))
        else:
            xx = []
            yy = []
            for point in (np.array([0, 0, 1]), np.array([w - 1, 0, 1]),
                          np.array([w - 1, h - 1, 1]), np.array([0, h - 1, 1])):
                target = np.dot(M, point)
                xx.append(target[0])
                yy.append(target[1])
            nh = int(math.ceil(max(yy)) - math.floor(min(yy)))
            nw = int(math.ceil(max(xx)) - math.floor(min(xx)))

            M[0, 2] += (nw - w) / 2
            M[1, 2] += (nh - h) / 2
            dst = cv2.warpAffine(img, M, (nw, nh), flags=interpolation)
    else:
        dst = cv2.warpAffine(img, M, (w, h), flags=interpolation)
    return dst.astype(dtype)


@keepdims
def to_grayscale(img, num_output_channels=1):
    """Converts image to grayscale version of image.

    Args:
        img (numpy.ndarray): Image to be converted to grayscale.

    Returns:
        numpy.ndarray:  Grayscale version of the image.
                        if num_output_channels == 1, returned image is single channel
                        if num_output_channels == 3, returned image is 3 channel with r == g == b
    
    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.vision.transforms.functional import to_grayscale

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = to_grayscale(fake_img)
            print(fake_img.shape)
    """

    if num_output_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif num_output_channels == 3:
        img = cv2.cvtColor(
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img
