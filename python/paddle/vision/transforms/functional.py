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
import math
from PIL import Image, ImageOps, ImageEnhance

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
from numpy import sin, cos, tan
import numbers
import collections
import warnings
import paddle

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = [
    'to_tensor', 'hflip', 'vflip', 'resize', 'pad', 'rotate', 'to_grayscale',
    'crop', 'center_crop', 'adjust_brightness', 'adjust_contrast', 'adjust_hue',
    'to_grayscale', 'normalize'
]

_pil_interp_from_str = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'box': Image.BOX,
    'lanczos': Image.LANCZOS,
    'hamming': Image.HAMMING
}

if cv2 is not None:
    _cv2_interp_from_str = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }

    _cv2_pad_from_str = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return isinstance(img, paddle.Tensor)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic, data_format='CHW'):
    """Converts a ``PIL.Image`` or ``numpy.ndarray`` to paddle.Tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL.Image|np.ndarray): Image to be converted to tensor.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        Tensor: Converted image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            tensor = F.to_tensor(fake_img)
            print(tensor.shape)

    """
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(
            type(pic)))

    if not data_format in ['CHW', 'HWC']:
        raise ValueError('data_format should be CHW or HWC. Got {}'.format(
            data_format))

    if isinstance(pic, np.ndarray):
        # numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        if data_format == 'CHW':
            img = paddle.to_tensor(pic.transpose((2, 0, 1)))
        else:
            img = paddle.to_tensor(pic)

        if paddle.fluid.data_feeder.convert_dtype(img.dtype) == 'uint8':
            return paddle.cast(img, np.float32) / 255.
        else:
            return img

    # PIL Image
    if pic.mode == 'I':
        img = paddle.to_tensor(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        # cast and reshape not support int16
        img = paddle.to_tensor(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'F':
        img = paddle.to_tensor(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * paddle.to_tensor(np.array(pic, np.uint8, copy=False))
    else:
        img = paddle.to_tensor(np.array(pic, copy=False))

    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)

    dtype = paddle.fluid.data_feeder.convert_dtype(img.dtype)
    if dtype == 'uint8':
        img = paddle.cast(img, np.float32) / 255.

    img = img.reshape([pic.size[1], pic.size[0], nchannel])

    if data_format == 'CHW':
        img = img.transpose([2, 0, 1])

    return img


def resize(img, size, interpolation='bilinear', backend='pil'):
    """
    Resizes the image to given size

    Args:
        input (PIL.Image|np.ndarray): Image to be resized.
        size (int|list|tuple): Target size of input data, with (height, width) shape.
        interpolation (int|str, optional): Interpolation method. when use pil backend, 
            support method are as following: 
            - 'nearest': Image.NEAREST, 
            - 'bilinear': Image.BILINEAR, 
            - 'bicubic': Image.BICUBIC, 
            - 'box': Image.BOX, 
            - 'lanczos': Image.LANCZOS, 
            - 'hamming': Image.HAMMING
            when use cv2 backend, support method are as following: 
            - 'nearest': cv2.INTER_NEAREST, 
            - 'bilinear': cv2.INTER_LINEAR, 
            - 'area': cv2.INTER_AREA, 
            - 'bicubic': cv2.INTER_CUBIC, 
            - 'lanczos': cv2.INTER_LANCZOS4
        backend (str, optional): The image resize backend type. Options are `pil`, 
            `cv2`. Default: 'pil'. 

    Returns:
        PIL.Image or np.array: Resized image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            converted_img = F.resize(fake_img, 224)
            print(converted_img.size)

            converted_img = F.resize(fake_img, (200, 150))
            print(converted_img.size)
    """

    if backend not in ['cv2', 'pil']:
        raise ValueError("Expected backend is 'cv2' or 'pil', \
                          but got {}".format(backend))

    if not (isinstance(size, int) or
            (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if backend == 'pil':
        if not _is_pil_image(img):
            raise TypeError(
                'img should be PIL Image when backend is pil. Got {}'.format(
                    type(img)))

        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), _pil_interp_from_str[interpolation])
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), _pil_interp_from_str[interpolation])
        else:
            return img.resize(size[::-1], _pil_interp_from_str[interpolation])
    else:
        if not _is_numpy_image(img):
            raise TypeError(
                'img should be numpy image when backend is cv2. Got {}'.format(
                    type(img)))

        h, w = img.shape[:2]

        if isinstance(size, int):
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                output = cv2.resize(
                    img,
                    dsize=(ow, oh),
                    interpolation=_cv2_interp_from_str[interpolation])
            else:
                oh = size
                ow = int(size * w / h)
                output = cv2.resize(
                    img,
                    dsize=(ow, oh),
                    interpolation=_cv2_interp_from_str[interpolation])
        else:
            output = cv2.resize(
                img,
                dsize=(size[1], size[0]),
                interpolation=_cv2_interp_from_str[interpolation])
        if len(img.shape) == 3 and img.shape[2] == 1:
            return output[:, :, np.newaxis]
        else:
            return output


def pad(img, padding, fill=0, padding_mode='constant', backend='pil'):
    """
    Pads the given PIL.Image or numpy.array on all sides with specified padding mode and fill value.

    Args:
        img (PIL.Image|np.array): Image to be padded.
        padding (int|list|tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (float, optional): Pixel fill value for constant fill. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant. Default: 0. 
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default: 'constant'.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value on the edge of the image

            - reflect: pads with reflection of image (without repeating the last value on the edge)

                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image (repeating the last value on the edge)

                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
        backend (str, optional): The image resize backend type. Options are `pil`, 
            `cv2`. Default: 'pil'. 
    Returns:
        PIL.Image or np.array: Padded image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            padded_img = F.pad(fake_img, padding=1)
            print(padded_img.size)

            padded_img = F.pad(fake_img, padding=(2, 1))
            print(padded_img.size)
    """
    if backend not in ['cv2', 'pil']:
        raise ValueError("Expected backend is 'cv2' or 'pil', \
                          but got {}".format(backend))

    if not isinstance(padding, (numbers.Number, list, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, list, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        raise ValueError(
            "Padding must be an int or a 2, or 4 element tuple, not a " +
            "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, list):
        padding = tuple(padding)
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    if backend == 'pil':
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if padding_mode == 'constant':
            if img.mode == 'P':
                palette = img.getpalette()
                image = ImageOps.expand(img, border=padding, fill=fill)
                image.putpalette(palette)
                return image

            return ImageOps.expand(img, border=padding, fill=fill)
        else:
            if img.mode == 'P':
                palette = img.getpalette()
                img = np.asarray(img)
                img = np.pad(img, ((pad_top, pad_bottom),
                                   (pad_left, pad_right)), padding_mode)
                img = Image.fromarray(img)
                img.putpalette(palette)
                return img

            img = np.asarray(img)
            # RGB image
            if len(img.shape) == 3:
                img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right),
                                   (0, 0)), padding_mode)
            # Grayscale image
            if len(img.shape) == 2:
                img = np.pad(img, ((pad_top, pad_bottom),
                                   (pad_left, pad_right)), padding_mode)

            return Image.fromarray(img)
    elif backend == 'cv2':
        if len(img.shape) == 3 and img.shape[2] == 1:
            return cv2.copyMakeBorder(
                img,
                top=pad_top,
                bottom=pad_bottom,
                left=pad_left,
                right=pad_right,
                borderType=_cv2_pad_from_str[padding_mode],
                value=fill)[:, :, np.newaxis]
        else:
            return cv2.copyMakeBorder(
                img,
                top=pad_top,
                bottom=pad_bottom,
                left=pad_left,
                right=pad_right,
                borderType=_cv2_pad_from_str[padding_mode],
                value=fill)


def crop(img, top, left, height, width, backend='pil'):
    """Crops the given PIL Image.

    Args:
        img (PIL.Image|np.array): Image to be cropped. (0,0) denotes the top left 
            corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        backend (str, optional): The image resize backend type. Options are `pil`, 
            `cv2`. Default: 'pil'. 

    Returns:
        PIL.Image or np.array: Cropped image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            cropped_img = F.crop(fake_img, 56, 150, 200, 100)
            print(cropped_img.size)

    """
    if backend not in ['cv2', 'pil']:
        raise ValueError("Expected backend is 'cv2' or 'pil', \
                          but got {}".format(backend))

    if backend == 'pil':
        if not _is_pil_image(img):
            raise TypeError(
                'img should be PIL Image when use backend pil. Got {}'.format(
                    type(img)))

        return img.crop((left, top, left + width, top + height))
    elif backend == 'cv2':
        if not _is_numpy_image(img):
            raise TypeError(
                'img should be numpy image when use backend cv2. Got {}'.format(
                    type(img)))

        return img[top:top + height, left:left + width, :]


def center_crop(img, output_size, backend='pil'):
    """Crops the given PIL Image and resize it to desired size.

        Args:
            img (PIL.Image|np.array): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
            backend (str, optional): The image resize backend type. Options are `pil`, `cv2`. Default: 'pil'. 
        
        Returns:
            PIL.Image or np.array: Cropped image.

        Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            cropped_img = F.center_crop(fake_img, (150, 100))
            print(cropped_img.size)
        """
    if backend not in ['cv2', 'pil']:
        raise ValueError("Expected backend is 'cv2' or 'pil', \
                          but got {}".format(backend))

    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))

    if backend == 'pil':
        image_width, image_height = img.size
        crop_height, crop_width = output_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(
            img, crop_top, crop_left, crop_height, crop_width, backend='pil')
    elif backend == 'cv2':
        h, w = img.shape[0:2]
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return crop(img, i, j, th, tw, backend='cv2')


def hflip(img, backend='pil'):
    """Horizontally flips the given PIL Image or np.array.

    Args:
        img (PIL.Image|np.array): Image to be flipped.
        backend (str, optional): The image resize backend type. Options are `pil`, 
            `cv2`. Default: 'pil'. 

    Returns:
        PIL.Image or np.array:  Horizontall flipped image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            flpped_img = F.hflip(fake_img)
            print(flpped_img.size)

    """
    if backend not in ['cv2', 'pil']:
        raise ValueError("Expected backend is 'cv2' or 'pil', \
                          but got {}".format(backend))

    if backend == 'pil':
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif backend == 'cv2':
        if not _is_numpy_image(img):
            raise TypeError('img should be numpy image. Got {}'.format(
                type(img)))

        return cv2.flip(img, 1)


def vflip(img, backend='pil'):
    """Vertically flips the given PIL Image or np.array.

    Args:
        img (PIL.Image|np.array): Image to be flipped.
        backend (str, optional): The image resize backend type. Options are `pil`, 
            `cv2`. Default: 'pil'. 

    Returns:
        PIL.Image or np.array:  Vertically flipped image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            flpped_img = F.vflip(fake_img)
            print(flpped_img.size)

    """
    if backend not in ['cv2', 'pil']:
        raise ValueError("Expected backend is 'cv2' or 'pil', \
                          but got {}".format(backend))

    if backend == 'pil':
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif backend == 'cv2':
        if not _is_numpy_image(img):
            raise TypeError('img should be numpy Image. Got {}'.format(
                type(img)))
        if len(img.shape) == 3 and img.shape[2] == 1:
            return cv2.flip(img, 0)[:, :, np.newaxis]
        else:
            return cv2.flip(img, 0)


def adjust_brightness(img, brightness_factor, backend='pil'):
    """Adjusts brightness of an Image.

    Args:
        img (PIL.Image|np.array): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
        backend (str, optional): The image resize backend type. Options are `pil`, 
            `cv2`. Default: 'pil'. 
    Returns:
        PIL.Image or np.array: Brightness adjusted image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            converted_img = F.adjust_brightness(fake_img, 0.4)
            print(converted_img.size)
    """
    if backend not in ['cv2', 'pil']:
        raise ValueError("Expected backend is 'cv2' or 'pil', \
                          but got {}".format(backend))

    if backend == 'pil':
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        return img
    elif backend == 'cv2':
        if not _is_numpy_image(img):
            raise TypeError('img should be numpy Image. Got {}'.format(
                type(img)))
        table = np.array([i * brightness_factor
                          for i in range(0, 256)]).clip(0, 255).astype('uint8')

        if len(img.shape) == 3 and img.shape[2] == 1:
            return cv2.LUT(img, table)[:, :, np.newaxis]
        else:
            return cv2.LUT(img, table)


def adjust_contrast(img, contrast_factor, backend='pil'):
    """Adjusts contrast of an Image.

    Args:
        img (PIL.Image|np.array): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
        backend (str, optional): The image resize backend type. Options are `pil`, 
            `cv2`. Default: 'pil'. 
    Returns:
        PIL.Image or np.array: Contrast adjusted image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            converted_img = F.adjust_contrast(fake_img, 0.4)
            print(converted_img.size)
    """
    if backend not in ['cv2', 'pil']:
        raise ValueError("Expected backend is 'cv2' or 'pil', \
                          but got {}".format(backend))

    if backend == 'pil':
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        return img
    elif backend == 'cv2':
        if not _is_numpy_image(img):
            raise TypeError('img should be numpy Image. Got {}'.format(
                type(img)))
        table = np.array([(i - 74) * contrast_factor + 74
                          for i in range(0, 256)]).clip(0, 255).astype('uint8')
        if len(img.shape) == 3 and img.shape[2] == 1:
            return cv2.LUT(img, table)[:, :, np.newaxis]
        else:
            return cv2.LUT(img, table)


def adjust_saturation(img, saturation_factor, backend='pil'):
    """Adjusts color saturation of an image.

    Args:
        img (PIL.Image|np.array): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
        backend (str, optional): The image resize backend type. Options are `pil`, 
            `cv2`. Default: 'pil'. 

    Returns:
        PIL.Image or np.array: Saturation adjusted image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            converted_img = F.adjust_saturation(fake_img, 0.4)
            print(converted_img.size)

    """
    if backend not in ['cv2', 'pil']:
        raise ValueError("Expected backend is 'cv2' or 'pil', \
                          but got {}".format(backend))

    if backend == 'pil':
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        return img
    elif backend == 'cv2':
        if not _is_numpy_image(img):
            raise TypeError('img should be numpy Image. Got {}'.format(
                type(img)))

        dtype = img.dtype
        img = img.astype(np.float32)
        alpha = np.random.uniform(
            max(0, 1 - saturation_factor), 1 + saturation_factor)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img[..., np.newaxis]
        img = img * alpha + gray_img * (1 - alpha)
        return img.clip(0, 255).astype(dtype)


def adjust_hue(img, hue_factor, backend='pil'):
    """Adjusts hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    Args:
        img (PIL.Image|np.array): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
        backend (str, optional): The image resize backend type. Options are `pil`, 
            `cv2`. Default: 'pil'. 

    Returns:
        PIL.Image or np.array: Hue adjusted image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            converted_img = F.adjust_hue(fake_img, 0.4)
            print(converted_img.size)

    """
    if backend not in ['cv2', 'pil']:
        raise ValueError("Expected backend is 'cv2' or 'pil', \
                          but got {}".format(backend))

    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if backend == 'pil':
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        input_mode = img.mode
        if input_mode in {'L', '1', 'I', 'F'}:
            return img

        h, s, v = img.convert('HSV').split()

        np_h = np.array(h, dtype=np.uint8)
        # uint8 addition take cares of rotation across boundaries
        with np.errstate(over='ignore'):
            np_h += np.uint8(hue_factor * 255)
        h = Image.fromarray(np_h, 'L')

        img = Image.merge('HSV', (h, s, v)).convert(input_mode)
        return img
    elif backend == 'cv2':
        if not _is_numpy_image(img):
            raise TypeError('img should be numpy Image. Got {}'.format(
                type(img)))

        dtype = img.dtype
        img = img.astype(np.uint8)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        h, s, v = cv2.split(hsv_img)

        alpha = np.random.uniform(hue_factor, hue_factor)
        h = h.astype(np.uint8)
        # uint8 addition take cares of rotation across boundaries
        with np.errstate(over="ignore"):
            h += np.uint8(alpha * 255)
        hsv_img = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR_FULL).astype(dtype)


def rotate(img,
           angle,
           resample=False,
           expand=False,
           center=None,
           fill=0,
           backend='pil'):
    """Rotates the image by angle.


    Args:
        img (PIL.Image|np.array): Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        resample (int|str, optional): An optional resampling filter. If omitted, or if the 
            image has only one channel, it is set to PIL.Image.NEAREST or cv2.INTER_NEAREST 
            according the backend. when use pil backend, support method are as following: 
            - 'nearest': Image.NEAREST, 
            - 'bilinear': Image.BILINEAR, 
            - 'bicubic': Image.BICUBIC
            when use cv2 backend, support method are as following: 
            - 'nearest': cv2.INTER_NEAREST, 
            - 'bilinear': cv2.INTER_LINEAR, 
            - 'bicubic': cv2.INTER_CUBIC
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.
        backend (str, optional): The image resize backend type. Options are `pil`, 
                    `cv2`. Default: 'pil'. 

    Returns:
        PIL.Image or np.array: Rotated image.

    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            rotated_img = F.rotate(fake_img, 90)
            print(rotated_img.size)

    """
    if backend not in ['cv2', 'pil']:
        raise ValueError("Expected backend is 'cv2' or 'pil', \
                          but got {}".format(backend))

    if backend == 'pil':
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if isinstance(fill, int):
            fill = tuple([fill] * 3)

        return img.rotate(angle, resample, expand, center, fillcolor=fill)
    elif backend == 'cv2':
        if not _is_numpy_image(img):
            raise TypeError('img should be numpy Image. Got {}'.format(
                type(img)))

        rows, cols = img.shape[0:2]
        if center is None:
            center = (cols / 2, rows / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        if len(img.shape) == 3 and img.shape[2] == 1:
            return cv2.warpAffine(img, M, (cols, rows))[:, :, np.newaxis]
        else:
            return cv2.warpAffine(img, M, (cols, rows))


def to_grayscale(img, num_output_channels=1, backend='pil'):
    """Converts image to grayscale version of image.

    Args:
        img (PIL.Image|np.array): Image to be converted to grayscale.
        backend (str, optional): The image resize backend type. Options are `pil`, 
                    `cv2`. Default: 'pil'. 

    Returns:
        PIL.Image or np.array: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel

            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    
    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            gray_img = F.to_grayscale(fake_img)
            print(gray_img.size)

    """
    if backend not in ['cv2', 'pil']:
        raise ValueError("Expected backend is 'cv2' or 'pil', \
                          but got {}".format(backend))

    if backend == 'pil':
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if num_output_channels == 1:
            img = img.convert('L')
        elif num_output_channels == 3:
            img = img.convert('L')
            np_img = np.array(img, dtype=np.uint8)
            np_img = np.dstack([np_img, np_img, np_img])
            img = Image.fromarray(np_img, 'RGB')
        else:
            raise ValueError('num_output_channels should be either 1 or 3')

        return img
    elif backend == 'cv2':
        if not _is_numpy_image(img):
            raise TypeError('img should be numpy ndarray. Got {}'.format(
                type(img)))

        if num_output_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        elif num_output_channels == 3:
            # much faster than doing cvtColor to go back to gray
            img = np.broadcast_to(
                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis],
                img.shape)
        else:
            raise ValueError('num_output_channels should be either 1 or 3')

        return img


def normalize(img, mean, std, data_format='CHW', to_rgb=False):
    """Normalizes a tensor or image with mean and standard deviation.

    Args:
        img (PIL.Image|np.array|paddle.Tensor): input data to be normalized.
        mean (list|tuple): Sequence of means for each channel.
        std (list|tuple): Sequence of standard deviations for each channel.
        data_format (str, optional): Data format of img, should be 'HWC' or 
            'CHW'. Default: 'CHW'.
        to_rgb (bool, optional): Whether to convert to rgb. Default: False.
        backend (str, optional): The image resize backend type. Options are `pil`, 
                    `cv2`. Default: 'pil'. 

    Returns:
        Tensor: Normalized mage.
    
    Examples:
        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import functional as F

            fake_img = (np.random.rand(256, 300, 3) * 255.).astype('uint8')

            fake_img = Image.fromarray(fake_img)

            mean = [127.5, 127.5, 127.5]
            std = [127.5, 127.5, 127.5]

            normalized_img = F.normalize(fake_img, mean, std, data_format='HWC')
            print(normalized_img.max(), normalized_img.min())

    """

    if _is_tensor_image(img):
        if data_format == 'CHW':
            mean = paddle.to_tensor(mean).reshape([-1, 1, 1])
            std = paddle.to_tensor(std).reshape([-1, 1, 1])
        else:
            mean = paddle.to_tensor(mean)
            std = paddle.to_tensor(std)
        return (img - mean) / std
    else:
        if _is_pil_image(img):
            img = np.array(img).astype(np.float32)

        if data_format == 'CHW':
            mean = np.float32(np.array(mean).reshape(-1, 1, 1))
            std = np.float32(np.array(std).reshape(-1, 1, 1))
        else:
            mean = np.float32(np.array(mean).reshape(1, 1, -1))
            std = np.float32(np.array(std).reshape(1, 1, -1))
        if to_rgb:
            # inplace
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

        img = (img - mean) / std
    return img
