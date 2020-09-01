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
    "RandomCrop",
    "RandomErasing",
    "Pad",
    "RandomRotate",
    "Grayscale",
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

            from paddle.vision.datasets import Flowers
            from paddle.vision.transforms import Compose, ColorJitter, Resize

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

            from paddle import set_device
            from paddle.vision.datasets import Flowers
            from paddle.vision.transforms import Compose, BatchCompose, Resize

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

            from paddle.vision.transforms import Resize

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

            from paddle.vision.transforms import RandomResizedCrop

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

            from paddle.vision.transforms import CenterCropResize

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

            from paddle.vision.transforms import CenterCrop

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

            from paddle.vision.transforms import RandomHorizontalFlip

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

            from paddle.vision.transforms import RandomVerticalFlip

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

            from paddle.vision.transforms import Normalize

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
            std = [std, std, std]

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

            from paddle.vision.transforms import Permute

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

            from paddle.vision.transforms import GaussianNoise

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

            from paddle.vision.transforms import BrightnessTransform

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

            from paddle.vision.transforms import ContrastTransform

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

            from paddle.vision.transforms import SaturationTransform

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

            from paddle.vision.transforms import HueTransform

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
            Chosen uniformly from [max(0, 1 - brightness), 1 + brightness]. Should be non negative numbers.
        contrast: How much to jitter contrast.
            Chosen uniformly from [max(0, 1 - contrast), 1 + contrast]. Should be non negative numbers.
        saturation: How much to jitter saturation.
            Chosen uniformly from [max(0, 1 - saturation), 1 + saturation]. Should be non negative numbers.
        hue: How much to jitter hue.
            Chosen uniformly from [-hue, hue]. Should have 0<= hue <= 0.5.

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.vision.transforms import ColorJitter

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


class RandomCrop(object):
    """Crops the given CV Image at a random location.

    Args:
        size (sequence|int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int|sequence|optional): Optional padding on each border
            of the image. If a sequence of length 4 is provided, it is used to pad left, 
            top, right, bottom borders respectively. Default: 0.
        pad_if_needed (boolean|optional): It will pad the image if smaller than the
            desired size to avoid raising an exception. Default: False.
    
    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.vision.transforms import RandomCrop

            transform = RandomCrop(224)

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    def _get_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (numpy.ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.

        """
        h, w, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        try:
            i = random.randint(0, h - th)
        except ValueError:
            i = random.randint(h - th, 0)
        try:
            j = random.randint(0, w - tw)
        except ValueError:
            j = random.randint(w - tw, 0)
        return i, j, th, tw

    def __call__(self, img):
        """

        Args:
            img (numpy.ndarray): Image to be cropped.
        Returns:
            numpy.ndarray: Cropped image.

        """
        if self.padding > 0:
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.shape[1]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.shape[0]) / 2)))

        i, j, h, w = self._get_params(img, self.size)

        return img[i:i + h, j:j + w]


class RandomErasing(object):
    """Randomly selects a rectangle region in an image and erases its pixels.
    ``Random Erasing Data Augmentation`` by Zhong et al.
    See https://arxiv.org/pdf/1708.04896.pdf

    Args:
         prob (float): probability that the random erasing operation will be performed.
         scale (tuple): range of proportion of erased area against input image. Should be (min, max).
         ratio (float): range of aspect ratio of erased area.
         value (float|list|tuple): erasing value. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively. Default: 0. 

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.vision.transforms import RandomCrop

            transform = RandomCrop(224)

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self,
                 prob=0.5,
                 scale=(0.02, 0.4),
                 ratio=0.3,
                 value=[0., 0., 0.]):
        assert isinstance(value, (
            float, Sequence
        )), "Expected type of value in [float, list, tupue], but got {}".format(
            type(value))
        assert scale[0] <= scale[1], "scale range should be of kind (min, max)!"

        if isinstance(value, float):
            self.value = [value, value, value]
        else:
            self.value = value

        self.p = prob
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img

        for _ in range(100):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio, 1 / self.ratio)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)

                if len(img.shape) == 3 and img.shape[2] == 3:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.value[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = self.value[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = self.value[2]
                else:
                    img[x1:x1 + h, y1:y1 + w] = self.value[1]
                return img

        return img


class Pad(object):
    """Pads the given CV Image on all sides with the given "pad" value.

    Args:
        padding (int|list|tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int|list|tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            ``constant`` means pads with a constant value, this value is specified with fill. 
            ``edge`` means pads with the last value at the edge of the image. 
            ``reflect`` means pads with reflection of image (without repeating the last value on the edge) 
            padding ``[1, 2, 3, 4]`` with 2 elements on both sides in reflect mode 
            will result in ``[3, 2, 1, 2, 3, 4, 3, 2]``.
            ``symmetric`` menas pads with reflection of image (repeating the last value on the edge)
            padding ``[1, 2, 3, 4]`` with 2 elements on both sides in symmetric mode 
            will result in ``[2, 1, 1, 2, 3, 4, 4, 3]``.

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.vision.transforms import Pad

            transform = Pad(2)

            fake_img = np.random.rand(500, 500, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, list, tuple))
        assert isinstance(fill, (numbers.Number, str, list, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding,
                      collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError(
                "Padding must be an int or a 2, or 4 element tuple, not a " +
                "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be padded.
        Returns:
            numpy.ndarray: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode)


class RandomRotate(object):
    """Rotates the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees) clockwise order.
        interpolation (int|optional): Interpolation mode of resize. Default: cv2.INTER_LINEAR.
        expand (bool|optional): Optional expansion flag. Default: False.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple|optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    
    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.vision.transforms import RandomRotate

            transform = RandomRotate(90)

            fake_img = np.random.rand(500, 400, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self,
                 degrees,
                 interpolation=cv2.INTER_LINEAR,
                 expand=False,
                 center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.interpolation = interpolation
        self.expand = expand
        self.center = center

    def _get_params(self, degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
            img (np.ndarray): Image to be rotated.
        Returns:
            np.ndarray: Rotated image.
        """

        angle = self._get_params(self.degrees)

        return F.rotate(img, angle, self.interpolation, self.expand,
                        self.center)


class Grayscale(object):
    """Converts image to grayscale.

    Args:
        output_channels (int): (1 or 3) number of channels desired for output image
    Returns:
        CV Image: Grayscale version of the input.
        - If output_channels == 1 : returned image is single channel
        - If output_channels == 3 : returned image is 3 channel with r == g == b

    Examples:
    
        .. code-block:: python

            import numpy as np

            from paddle.vision.transforms import Grayscale

            transform = Grayscale()

            fake_img = np.random.rand(500, 400, 3).astype('float32')

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(self, output_channels=1):
        self.output_channels = output_channels

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be converted to grayscale.
        Returns:
            numpy.ndarray: Randomly grayscaled image.
        """
        return F.to_grayscale(img, num_output_channels=self.output_channels)
