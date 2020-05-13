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

import math
import sys
import random
import cv2

import numpy as np
import numbers
import types
import collections
import warnings
import traceback

from . import functional as F

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

__all__ = [
    "Compose",
    "BatchCompose",
    "Resize",
    "RandomResizedCrop",
    "CenterCropResize",
    "CenterCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "Permute",
    "Normalize",
    "GaussianNoise",
    "BrightnessTransform",
    "SaturationTransform",
    "ContrastTransform",
    "HueTransform",
    "ColorJitter",
]


class Compose(object):
    """
    Composes several transforms together use for composing list of transforms
    together for a dataset transform.

    Args:
        transforms (list): List of transforms to compose.

    Returns:
        A compose object which is callable, __call__ for this Compose
        object will call each given :attr:`transforms` sequencely.

    Examples:
    
        .. code-block:: python

            from paddle.incubate.hapi.datasets import Flowers
            from paddle.incubate.hapi.vision.transforms import Compose, ColorJitter, Resize

            transform = Compose([ColorJitter(), Resize(size=608)])
            flowers = Flowers(mode='test', transform=transform)

            for i in range(10):
                sample = flowers[i]
                print(sample[0].shape, sample[1])

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *data):
        for f in self.transforms:
            try:
                # multi-fileds in a sample
                if isinstance(data, Sequence):
                    data = f(*data)
                # single field in a sample, call transform directly
                else:
                    data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                print("fail to perform transform [{}] with error: "
                      "{} and stack:\n{}".format(f, e, str(stack_info)))
                raise e
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class BatchCompose(object):
    """Composes several batch transforms together

    Args:
        transforms (list): List of transforms to compose.
                           these transforms perform on batch data.

    Examples:
    
        .. code-block:: python

            import numpy as np
            from paddle.io import DataLoader

            from paddle.incubate.hapi.model import set_device
            from paddle.incubate.hapi.datasets import Flowers
            from paddle.incubate.hapi.vision.transforms import Compose, BatchCompose, Resize

            class NormalizeBatch(object):
                def __init__(self,
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            scale=True,
                            channel_first=True):

                    self.mean = mean
                    self.std = std
                    self.scale = scale
                    self.channel_first = channel_first
                    if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                            isinstance(self.scale, bool)):
                        raise TypeError("{}: input type is invalid.".format(self))
                    from functools import reduce
                    if reduce(lambda x, y: x * y, self.std) == 0:
                        raise ValueError('{}: std is invalid!'.format(self))

                def __call__(self, samples):
                    for i in range(len(samples)):
                        samples[i] = list(samples[i])
                        im = samples[i][0]
                        im = im.astype(np.float32, copy=False)
                        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
                        std = np.array(self.std)[np.newaxis, np.newaxis, :]
                        if self.scale:
                            im = im / 255.0
                        im -= mean
                        im /= std
                        if self.channel_first:
                            im = im.transpose((2, 0, 1))
                        samples[i][0] = im
                    return samples

            transform = Compose([Resize((500, 500))])
            flowers_dataset = Flowers(mode='test', transform=transform)

            device = set_device('cpu')

            collate_fn = BatchCompose([NormalizeBatch()])
            loader = DataLoader(
                        flowers_dataset,
                        batch_size=4,
                        places=device,
                        return_list=True,
                        collate_fn=collate_fn)

            for data in loader:
                # do something
                break
    """

    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, data):
        for f in self.transforms:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                print("fail to perform batch transform [{}] with error: "
                      "{} and stack:\n{}".format(f, e, str(stack_info)))
                raise e

        # sample list to batch data
        batch = list(zip(*data))

        return batch


class Resize(object):
    """Resize the input Image to the given size.

    Args:
        size (int|list|tuple): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int): Interpolation mode of resize. Default: cv2.INTER_LINEAR.

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.incubate.hapi.vision.transforms import Resize

            transform = Resize(size=224)

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and
                                         len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return F.resize(img, self.size, self.interpolation)


class RandomResizedCrop(object):
    """Crop the input data to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 1.33) of the original aspect ratio is made.
    After applying crop transfrom, the input data will be resized to given size.

    Args:
        output_size (int|list|tuple): Target size of output image, with (height, width) shape.
        scale (list|tuple): Range of size of the origin size cropped. Default: (0.08, 1.0)
        ratio (list|tuple): Range of aspect ratio of the origin aspect ratio cropped. Default: (0.75, 1.33)

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.incubate.hapi.vision.transforms import RandomResizedCrop

            transform = RandomResizedCrop(224)

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self,
                 output_size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4, 4. / 3),
                 interpolation=cv2.INTER_LINEAR):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        assert (scale[0] <= scale[1]), "scale should be of kind (min, max)"
        assert (ratio[0] <= ratio[1]), "ratio should be of kind (min, max)"
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def _get_params(self, image, attempts=10):
        height, width, _ = image.shape
        area = height * width

        for _ in range(attempts):
            target_area = np.random.uniform(*self.scale) * area
            log_ratio = tuple(math.log(x) for x in self.ratio)
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                x = np.random.randint(0, width - w + 1)
                y = np.random.randint(0, height - h + 1)
                return x, y, w, h

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        x = (width - w) // 2
        y = (height - h) // 2
        return x, y, w, h

    def __call__(self, img):
        x, y, w, h = self._get_params(img)
        cropped_img = img[y:y + h, x:x + w]
        return F.resize(cropped_img, self.output_size, self.interpolation)


class CenterCropResize(object):
    """Crops to center of image with padding then scales size.

    Args:
        size (int|list|tuple): Target size of output image, with (height, width) shape.
        crop_padding (int): Center crop with the padding. Default: 32.
        interpolation (int): Interpolation mode of resize. Default: cv2.INTER_LINEAR.

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.incubate.hapi.vision.transforms import CenterCropResize

            transform = CenterCropResize(224)

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, size, crop_padding=32, interpolation=cv2.INTER_LINEAR):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.crop_padding = crop_padding
        self.interpolation = interpolation

    def _get_params(self, img):
        h, w = img.shape[:2]
        size = min(self.size)
        c = int(size / (size + self.crop_padding) * min((h, w)))
        x = (h + 1 - c) // 2
        y = (w + 1 - c) // 2
        return c, x, y

    def __call__(self, img):
        c, x, y = self._get_params(img)
        cropped_img = img[x:x + c, y:y + c, :]
        return F.resize(cropped_img, self.size, self.interpolation)


class CenterCrop(object):
    """Crops the given the input data at the center.

    Args:
        output_size: Target size of output image, with (height, width) shape.
    
    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.incubate.hapi.vision.transforms import CenterCrop

            transform = CenterCrop(224)

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def _get_params(self, img):
        th, tw = self.output_size
        h, w, _ = img.shape
        assert th <= h and tw <= w, "output size is bigger than image size"
        x = int(round((w - tw) / 2.0))
        y = int(round((h - th) / 2.0))
        return x, y

    def __call__(self, img):
        x, y = self._get_params(img)
        th, tw = self.output_size
        return img[y:y + th, x:x + tw]


class RandomHorizontalFlip(object):
    """Horizontally flip the input data randomly with a given probability.

    Args:
        prob (float): Probability of the input data being flipped. Default: 0.5

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.incubate.hapi.vision.transforms import RandomHorizontalFlip

            transform = RandomHorizontalFlip(224)

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if np.random.random() < self.prob:
            return F.flip(img, code=1)
        return img


class RandomVerticalFlip(object):
    """Vertically flip the input data randomly with a given probability.

    Args:
        prob (float): Probability of the input data being flipped. Default: 0.5

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.incubate.hapi.vision.transforms import RandomVerticalFlip

            transform = RandomVerticalFlip(224)

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if np.random.random() < self.prob:
            return F.flip(img, code=0)
        return img


class Normalize(object):
    """Normalize the input data with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input data.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (int|float|list): Sequence of means for each channel.
        std (int|float|list): Sequence of standard deviations for each channel.

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.incubate.hapi.vision.transforms import Normalize

            normalize = Normalize(mean=[0.5, 0.5, 0.5], 
                                std=[0.5, 0.5, 0.5])

            fake_img = np.random.rand(3, 500, 500).astype('float32')

            fake_img = normalize(fake_img)
            print(fake_img.shape)
    
    """

    def __init__(self, mean=0.0, std=1.0):
        if isinstance(mean, numbers.Number):
            mean = [mean, mean, mean]

        if isinstance(std, numbers.Number):
            mean = [std, std, std]

        self.mean = np.array(mean, dtype=np.float32).reshape(len(mean), 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(len(std), 1, 1)

    def __call__(self, img):
        return (img - self.mean) / self.std


class Permute(object):
    """Change input data to a target mode.
    For example, most transforms use HWC mode image,
    while the Neural Network might use CHW mode input tensor.
    Input image should be HWC mode and an instance of numpy.ndarray. 

    Args:
        mode (str): Output mode of input. Default: "CHW".
        to_rgb (bool): Convert 'bgr' image to 'rgb'. Default: True.

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.incubate.hapi.vision.transforms import Permute

            transform = Permute()

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, mode="CHW", to_rgb=True):
        assert mode in [
            "CHW"
        ], "Only support 'CHW' mode, but received mode: {}".format(mode)
        self.mode = mode
        self.to_rgb = to_rgb

    def __call__(self, img):
        if self.to_rgb:
            img = img[..., ::-1]
        if self.mode == "CHW":
            return img.transpose((2, 0, 1))
        return img


class GaussianNoise(object):
    """Add random gaussian noise to the input data.
    Gaussian noise is generated with given mean and std.

    Args:
        mean (float): Gaussian mean used to generate noise.
        std (float): Gaussian standard deviation used to generate noise.
    
    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.incubate.hapi.vision.transforms import GaussianNoise

            transform = GaussianNoise()

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, mean=0.0, std=1.0):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img):
        dtype = img.dtype
        noise = np.random.normal(self.mean, self.std, img.shape) * 255
        img = img + noise.astype(np.float32)
        return np.clip(img, 0, 255).astype(dtype)


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
        self.transforms = Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
