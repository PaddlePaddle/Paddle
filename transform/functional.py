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


def flip(image, code):
    """
    Accordding to the code (the type of flip), flip the input image

    Args:
        image: Input image, with (H, W, C) shape
        code: code that indicates the type of flip.
            -1 : Flip horizontally and vertically
            0 : Flip vertically
            1 : Flip horizontally
    """
    return cv2.flip(image, flipCode=code)


def resize(img, size, interpolation=cv2.INTER_LINEAR):
    """
    resize the input data to given size

    Args:
        input: Input data, could be image or masks, with (H, W, C) shape
        size: Target size of input data, with (height, width) shape.
        interpolation: Interpolation method.
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
