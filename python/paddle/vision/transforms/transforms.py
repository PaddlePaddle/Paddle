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

import math
import numbers
import random
import traceback
from collections.abc import Iterable, Sequence

import numpy as np

import paddle

from . import functional as F

__all__ = []


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif F._is_numpy_image(img):
        return img.shape[:2][::-1]
    elif F._is_tensor_image(img):
        if len(img.shape) == 3:
            return img.shape[1:][::-1]  # chw -> wh
        elif len(img.shape) == 4:
            return img.shape[2:][::-1]  # nchw -> wh
        else:
            raise ValueError(
                "The dim for input Tensor should be 3-D or 4-D, but received {}".format(
                    len(img.shape)
                )
            )
    else:
        raise TypeError(f"Unexpected type {type(img)}")


def _check_input(
    value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True
):
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError(
                "If {} is a single number, it must be non negative.".format(
                    name
                )
            )
        value = [center - value, center + value]
        if clip_first_on_zero:
            value[0] = max(value[0], 0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound}")
    else:
        raise TypeError(
            "{} should be a single number or a list/tuple with lenght 2.".format(
                name
            )
        )

    if value[0] == value[1] == center:
        value = None
    return value


class Compose:
    """
    Composes several transforms together use for composing list of transforms
    together for a dataset transform.

    Args:
        transforms (list|tuple): List/Tuple of transforms to compose.

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
                print(sample[0].size, sample[1])

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for f in self.transforms:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                print(
                    "fail to perform transform [{}] with error: "
                    "{} and stack:\n{}".format(f, e, str(stack_info))
                )
                raise e
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class BaseTransform:
    """
    Base class of all transforms used in computer vision.

    calling logic:

    .. code-block:: text

        if keys is None:
            _get_params -> _apply_image()
        else:
            _get_params -> _apply_*() for * in keys

    If you want to implement a self-defined transform method for image,
    rewrite _apply_* method in subclass.

    Args:
        keys (list[str]|tuple[str], optional): Input type. Input is a tuple contains different structures,
            key is used to specify the type of input. For example, if your input
            is image type, then the key can be None or ("image"). if your input
            is (image, image) type, then the keys should be ("image", "image").
            if your input is (image, boxes), then the keys should be ("image", "boxes").

            Current available strings & data type are describe below:

                - "image": input image, with shape of (H, W, C)
                - "coords": coordinates, with shape of (N, 2)
                - "boxes": bounding boxes, with shape of (N, 4), "xyxy" format,the 1st "xy" represents
                  top left point of a box,the 2nd "xy" represents right bottom point.
                - "mask": map used for segmentation, with shape of (H, W, 1)

            You can also customize your data types only if you implement the corresponding
            _apply_*() methods, otherwise ``NotImplementedError`` will be raised.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            import paddle.vision.transforms.functional as F
            from paddle.vision.transforms import BaseTransform

            def _get_image_size(img):
                if F._is_pil_image(img):
                    return img.size
                elif F._is_numpy_image(img):
                    return img.shape[:2][::-1]
                else:
                    raise TypeError("Unexpected type {}".format(type(img)))

            class CustomRandomFlip(BaseTransform):
                def __init__(self, prob=0.5, keys=None):
                    super().__init__(keys)
                    self.prob = prob

                def _get_params(self, inputs):
                    image = inputs[self.keys.index('image')]
                    params = {}
                    params['flip'] = np.random.random() < self.prob
                    params['size'] = _get_image_size(image)
                    return params

                def _apply_image(self, image):
                    if self.params['flip']:
                        return F.hflip(image)
                    return image

                # if you only want to transform image, do not need to rewrite this function
                def _apply_coords(self, coords):
                    if self.params['flip']:
                        w = self.params['size'][0]
                        coords[:, 0] = w - coords[:, 0]
                    return coords

                # if you only want to transform image, do not need to rewrite this function
                def _apply_boxes(self, boxes):
                    idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
                    coords = np.asarray(boxes).reshape(-1, 4)[:, idxs].reshape(-1, 2)
                    coords = self._apply_coords(coords).reshape((-1, 4, 2))
                    minxy = coords.min(axis=1)
                    maxxy = coords.max(axis=1)
                    trans_boxes = np.concatenate((minxy, maxxy), axis=1)
                    return trans_boxes

                # if you only want to transform image, do not need to rewrite this function
                def _apply_mask(self, mask):
                    if self.params['flip']:
                        return F.hflip(mask)
                    return mask

            # create fake inputs
            fake_img = Image.fromarray((np.random.rand(400, 500, 3) * 255.).astype('uint8'))
            fake_boxes = np.array([[2, 3, 200, 300], [50, 60, 80, 100]])
            fake_mask = fake_img.convert('L')

            # only transform for image:
            flip_transform = CustomRandomFlip(1.0)
            converted_img = flip_transform(fake_img)

            # transform for image, boxes and mask
            flip_transform = CustomRandomFlip(1.0, keys=('image', 'boxes', 'mask'))
            (converted_img, converted_boxes, converted_mask) = flip_transform((fake_img, fake_boxes, fake_mask))
            print('converted boxes', converted_boxes)

    """

    def __init__(self, keys=None):
        if keys is None:
            keys = ("image",)
        elif not isinstance(keys, Sequence):
            raise ValueError(f"keys should be a sequence, but got keys={keys}")
        for k in keys:
            if self._get_apply(k) is None:
                raise NotImplementedError(f"{k} is unsupported data structure")
        self.keys = keys

        # storage some params get from function get_params()
        self.params = None

    def _get_params(self, inputs):
        pass

    def __call__(self, inputs):
        """Apply transform on single input data"""
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        self.params = self._get_params(inputs)

        outputs = []
        for i in range(min(len(inputs), len(self.keys))):
            apply_func = self._get_apply(self.keys[i])
            if apply_func is None:
                outputs.append(inputs[i])
            else:
                outputs.append(apply_func(inputs[i]))
        if len(inputs) > len(self.keys):
            outputs.extend(inputs[len(self.keys) :])

        if len(outputs) == 1:
            outputs = outputs[0]
        else:
            outputs = tuple(outputs)
        return outputs

    def _get_apply(self, key):
        return getattr(self, f"_apply_{key}", None)

    def _apply_image(self, image):
        raise NotImplementedError

    def _apply_boxes(self, boxes):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError


class ToTensor(BaseTransform):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to ``paddle.Tensor``.

    Converts a PIL.Image or numpy.ndarray (H x W x C) to a paddle.Tensor of shape (C x H x W).

    If input is a grayscale image (H x W), it will be converted to an image of shape (H x W x 1).
    And the shape of output tensor will be (1 x H x W).

    If you want to keep the shape of output tensor as (H x W x C), you can set data_format = ``HWC`` .

    Converts a PIL.Image or numpy.ndarray in the range [0, 255] to a paddle.Tensor in the
    range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr,
    RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8.

    In the other cases, tensors are returned without scaling.

    Args:
        data_format (str, optional): Data format of output tensor, should be 'HWC' or
            'CHW'. Default: 'CHW'.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray): The input image with shape (H x W x C).
        - output(np.ndarray): A tensor with shape (C x H x W) or (H x W x C) according option data_format.

    Returns:
        A callable object of ToTensor.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image

            import paddle.vision.transforms as T
            import paddle.vision.transforms.functional as F

            fake_img = Image.fromarray((np.random.rand(4, 5, 3) * 255.).astype(np.uint8))

            transform = T.ToTensor()

            tensor = transform(fake_img)

            print(tensor.shape)
            # [3, 4, 5]

            print(tensor.dtype)
            # paddle.float32
    """

    def __init__(self, data_format='CHW', keys=None):
        super().__init__(keys)
        self.data_format = data_format

    def _apply_image(self, img):
        """
        Args:
            img (PIL.Image|np.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(img, self.data_format)


class Resize(BaseTransform):
    """Resize the input Image to the given size.

    Args:
        size (int|list|tuple): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int|str, optional): Interpolation method. Default: 'bilinear'.
            when use pil backend, support method are as following:
            - "nearest": Image.NEAREST,
            - "bilinear": Image.BILINEAR,
            - "bicubic": Image.BICUBIC,
            - "box": Image.BOX,
            - "lanczos": Image.LANCZOS,
            - "hamming": Image.HAMMING
            when use cv2 backend, support method are as following:
            - "nearest": cv2.INTER_NEAREST,
            - "bilinear": cv2.INTER_LINEAR,
            - "area": cv2.INTER_AREA,
            - "bicubic": cv2.INTER_CUBIC,
            - "lanczos": cv2.INTER_LANCZOS4
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A resized image.

    Returns:
        A callable object of Resize.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import Resize

            fake_img = Image.fromarray((np.random.rand(256, 300, 3) * 255.).astype(np.uint8))

            transform = Resize(size=224)
            converted_img = transform(fake_img)
            print(converted_img.size)
            # (262, 224)

            transform = Resize(size=(200,150))
            converted_img = transform(fake_img)
            print(converted_img.size)
            # (150, 200)
    """

    def __init__(self, size, interpolation='bilinear', keys=None):
        super().__init__(keys)
        assert isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2
        )
        self.size = size
        self.interpolation = interpolation

    def _apply_image(self, img):
        return F.resize(img, self.size, self.interpolation)


class RandomResizedCrop(BaseTransform):
    """Crop the input data to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 1.33) of the original aspect ratio is made.
    After applying crop transfrom, the input data will be resized to given size.

    Args:
        size (int|list|tuple): Target size of output image, with (height, width) shape.
        scale (list|tuple, optional): Scale range of the cropped image before resizing, relatively to the origin
            image. Default: (0.08, 1.0).
        ratio (list|tuple, optional): Range of aspect ratio of the origin aspect ratio cropped. Default: (0.75, 1.33)
        interpolation (int|str, optional): Interpolation method. Default: 'bilinear'. when use pil backend,
            support method are as following:
            - "nearest": Image.NEAREST,
            - "bilinear": Image.BILINEAR,
            - "bicubic": Image.BICUBIC,
            - "box": Image.BOX,
            - "lanczos": Image.LANCZOS,
            - "hamming": Image.HAMMING
            when use cv2 backend, support method are as following:
            - "nearest": cv2.INTER_NEAREST,
            - "bilinear": cv2.INTER_LINEAR,
            - "area": cv2.INTER_AREA,
            - "bicubic": cv2.INTER_CUBIC,
            - "lanczos": cv2.INTER_LANCZOS4
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A cropped image.

    Returns:
        A callable object of RandomResizedCrop.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import RandomResizedCrop

            transform = RandomResizedCrop(224)

            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)

    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4, 4.0 / 3),
        interpolation='bilinear',
        keys=None,
    ):
        super().__init__(keys)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        assert scale[0] <= scale[1], "scale should be of kind (min, max)"
        assert ratio[0] <= ratio[1], "ratio should be of kind (min, max)"
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def _dynamic_get_param(self, image, attempts=10):
        width, height = _get_image_size(image)
        area = height * width

        for _ in range(attempts):
            target_area = np.random.uniform(*self.scale) * area
            log_ratio = tuple(math.log(x) for x in self.ratio)
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:
            # return whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def _static_get_param(self, image, attempts=10):
        width, height = _get_image_size(image)
        area = height * width
        log_ratio = tuple(math.log(x) for x in self.ratio)

        counter = paddle.full(
            shape=[1], fill_value=0, dtype='int32'
        )  # loop counter

        ten = paddle.full(
            shape=[1], fill_value=10, dtype='int32'
        )  # loop length

        i = paddle.zeros([1], dtype="int32")
        j = paddle.zeros([1], dtype="int32")
        h = paddle.ones([1], dtype="int32") * (height + 1)
        w = paddle.ones([1], dtype="int32") * (width + 1)

        def cond(counter, ten, i, j, h, w):
            return (counter < ten) and (w > width or h > height)

        def body(counter, ten, i, j, h, w):
            target_area = (
                paddle.uniform(shape=[1], min=self.scale[0], max=self.scale[1])
                * area
            )
            aspect_ratio = paddle.exp(
                paddle.uniform(shape=[1], min=log_ratio[0], max=log_ratio[1])
            )

            w = paddle.round(paddle.sqrt(target_area * aspect_ratio)).astype(
                'int32'
            )
            h = paddle.round(paddle.sqrt(target_area / aspect_ratio)).astype(
                'int32'
            )

            i = paddle.static.nn.cond(
                0 < w <= width and 0 < h <= height,
                lambda: paddle.uniform(shape=[1], min=0, max=height - h).astype(
                    "int32"
                ),
                lambda: i,
            )

            j = paddle.static.nn.cond(
                0 < w <= width and 0 < h <= height,
                lambda: paddle.uniform(shape=[1], min=0, max=width - w).astype(
                    "int32"
                ),
                lambda: j,
            )

            counter += 1

            return counter, ten, i, j, h, w

        counter, ten, i, j, h, w = paddle.static.nn.while_loop(
            cond, body, [counter, ten, i, j, h, w]
        )

        def central_crop(width, height):

            height = paddle.assign([height]).astype("float32")
            width = paddle.assign([width]).astype("float32")

            # Fallback to central crop
            in_ratio = width / height

            w, h = paddle.static.nn.cond(
                in_ratio < self.ratio[0],
                lambda: [
                    width.astype("int32"),
                    paddle.round(width / self.ratio[0]).astype("int32"),
                ],
                lambda: paddle.static.nn.cond(
                    in_ratio > self.ratio[1],
                    lambda: [
                        paddle.round(height * self.ratio[1]),
                        height.astype("int32"),
                    ],
                    lambda: [width.astype("int32"), height.astype("int32")],
                ),
            )
            i = (height.astype("int32") - h) // 2
            j = (width.astype("int32") - w) // 2

            return i, j, h, w, counter

        return paddle.static.nn.cond(
            0 < w <= width and 0 < h <= height,
            lambda: [i, j, h, w, counter],
            lambda: central_crop(width, height),
        )

    def _apply_image(self, img):
        if paddle.in_dynamic_mode():
            i, j, h, w = self._dynamic_get_param(img)
        else:
            i, j, h, w, counter = self._static_get_param(img)

        cropped_img = F.crop(img, i, j, h, w)
        return F.resize(cropped_img, self.size, self.interpolation)


class CenterCrop(BaseTransform):
    """Crops the given the input data at the center.

    Args:
        size (int|list|tuple): Target size of output image, with (height, width) shape.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A cropped image.

    Returns:
        A callable object of CenterCrop.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import CenterCrop

            transform = CenterCrop(224)

            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)
    """

    def __init__(self, size, keys=None):
        super().__init__(keys)
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def _apply_image(self, img):
        return F.center_crop(img, self.size)


class RandomHorizontalFlip(BaseTransform):
    """Horizontally flip the input data randomly with a given probability.

    Args:
        prob (float, optional): Probability of the input data being flipped. Should be in [0, 1]. Default: 0.5
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A horiziotal flipped image.

    Returns:
        A callable object of RandomHorizontalFlip.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import RandomHorizontalFlip

            transform = RandomHorizontalFlip(0.5)

            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)
    """

    def __init__(self, prob=0.5, keys=None):
        super().__init__(keys)
        assert 0 <= prob <= 1, "probability must be between 0 and 1"
        self.prob = prob

    def _apply_image(self, img):
        if paddle.in_dynamic_mode():
            return self._dynamic_apply_image(img)
        else:
            return self._static_apply_image(img)

    def _dynamic_apply_image(self, img):
        if random.random() < self.prob:
            return F.hflip(img)
        return img

    def _static_apply_image(self, img):
        return paddle.static.nn.cond(
            paddle.rand(shape=(1,)) < self.prob,
            lambda: F.hflip(img),
            lambda: img,
        )


class RandomVerticalFlip(BaseTransform):
    """Vertically flip the input data randomly with a given probability.

    Args:
        prob (float, optional): Probability of the input data being flipped. Default: 0.5
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A vertical flipped image.

    Returns:
        A callable object of RandomVerticalFlip.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import RandomVerticalFlip

            transform = RandomVerticalFlip()

            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)

    """

    def __init__(self, prob=0.5, keys=None):
        super().__init__(keys)
        assert 0 <= prob <= 1, "probability must be between 0 and 1"
        self.prob = prob

    def _apply_image(self, img):
        if paddle.in_dynamic_mode():
            return self._dynamic_apply_image(img)
        else:
            return self._static_apply_image(img)

    def _dynamic_apply_image(self, img):
        if random.random() < self.prob:
            return F.vflip(img)
        return img

    def _static_apply_image(self, img):
        return paddle.static.nn.cond(
            paddle.rand(shape=(1,)) < self.prob,
            lambda: F.vflip(img),
            lambda: img,
        )


class Normalize(BaseTransform):
    """Normalize the input data with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input data.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (int|float|list|tuple, optional): Sequence of means for each channel.
        std (int|float|list|tuple, optional): Sequence of standard deviations for each channel.
        data_format (str, optional): Data format of img, should be 'HWC' or
            'CHW'. Default: 'CHW'.
        to_rgb (bool, optional): Whether to convert to rgb. Default: False.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A normalized array or tensor.

    Returns:
        A callable object of Normalize.

    Examples:

        .. code-block:: python
          :name: code-example
            import paddle
            from paddle.vision.transforms import Normalize

            normalize = Normalize(mean=[127.5, 127.5, 127.5],
                                  std=[127.5, 127.5, 127.5],
                                  data_format='HWC')

            fake_img = paddle.rand([300,320,3]).numpy() * 255.

            fake_img = normalize(fake_img)
            print(fake_img.shape)
            # (300, 320, 3)
            print(fake_img.max(), fake_img.min())
            # 0.99999905 -0.999974

    """

    def __init__(
        self, mean=0.0, std=1.0, data_format='CHW', to_rgb=False, keys=None
    ):
        super().__init__(keys)
        if isinstance(mean, numbers.Number):
            mean = [mean, mean, mean]

        if isinstance(std, numbers.Number):
            std = [std, std, std]

        self.mean = mean
        self.std = std
        self.data_format = data_format
        self.to_rgb = to_rgb

    def _apply_image(self, img):
        return F.normalize(
            img, self.mean, self.std, self.data_format, self.to_rgb
        )


class Transpose(BaseTransform):
    """Transpose input data to a target format.
    For example, most transforms use HWC mode image,
    while the Neural Network might use CHW mode input tensor.
    output image will be an instance of numpy.ndarray.

    Args:
        order (list|tuple, optional): Target order of input data. Default: (2, 0, 1).
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(np.ndarray|Paddle.Tensor): A transposed array or tensor. If input
            is a PIL.Image, output will be converted to np.ndarray automatically.

    Returns:
        A callable object of Transpose.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import Transpose

            transform = Transpose()

            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.shape)

    """

    def __init__(self, order=(2, 0, 1), keys=None):
        super().__init__(keys)
        self.order = order

    def _apply_image(self, img):
        if F._is_tensor_image(img):
            return img.transpose(self.order)

        if F._is_pil_image(img):
            img = np.asarray(img)

        if len(img.shape) == 2:
            img = img[..., np.newaxis]
        return img.transpose(self.order)


class BrightnessTransform(BaseTransform):
    """Adjust brightness of the image.

    Args:
        value (float): How much to adjust the brightness. Can be any
            non negative number. 0 gives the original image.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An image with a transform in brghtness.

    Returns:
        A callable object of BrightnessTransform.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import BrightnessTransform

            transform = BrightnessTransform(0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)

    """

    def __init__(self, value, keys=None):
        super().__init__(keys)
        self.value = _check_input(value, 'brightness')

    def _apply_image(self, img):
        if self.value is None:
            return img

        brightness_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_brightness(img, brightness_factor)


class ContrastTransform(BaseTransform):
    """Adjust contrast of the image.

    Args:
        value (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives the original image.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An image with a transform in contrast.

    Returns:
        A callable object of ContrastTransform.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import ContrastTransform

            transform = ContrastTransform(0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)

    """

    def __init__(self, value, keys=None):
        super().__init__(keys)
        if value < 0:
            raise ValueError("contrast value should be non-negative")
        self.value = _check_input(value, 'contrast')

    def _apply_image(self, img):
        if self.value is None:
            return img

        contrast_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_contrast(img, contrast_factor)


class SaturationTransform(BaseTransform):
    """Adjust saturation of the image.

    Args:
        value (float): How much to adjust the saturation. Can be any
            non negative number. 0 gives the original image.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An image with a transform in saturation.

    Returns:
        A callable object of SaturationTransform.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import SaturationTransform

            transform = SaturationTransform(0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)

    """

    def __init__(self, value, keys=None):
        super().__init__(keys)
        self.value = _check_input(value, 'saturation')

    def _apply_image(self, img):
        if self.value is None:
            return img

        saturation_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_saturation(img, saturation_factor)


class HueTransform(BaseTransform):
    """Adjust hue of the image.

    Args:
        value (float): How much to adjust the hue. Can be any number
            between 0 and 0.5, 0 gives the original image.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An image with a transform in hue.

    Returns:
        A callable object of HueTransform.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import HueTransform

            transform = HueTransform(0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)

    """

    def __init__(self, value, keys=None):
        super().__init__(keys)
        self.value = _check_input(
            value, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )

    def _apply_image(self, img):
        if self.value is None:
            return img

        hue_factor = random.uniform(self.value[0], self.value[1])
        return F.adjust_hue(img, hue_factor)


class ColorJitter(BaseTransform):
    """Randomly change the brightness, contrast, saturation and hue of an image.

    Args:
        brightness (float, optional): How much to jitter brightness.
            Chosen uniformly from [max(0, 1 - brightness), 1 + brightness]. Should be non negative numbers. Default: 0.
        contrast (float, optional): How much to jitter contrast.
            Chosen uniformly from [max(0, 1 - contrast), 1 + contrast]. Should be non negative numbers. Default: 0.
        saturation (float, optional): How much to jitter saturation.
            Chosen uniformly from [max(0, 1 - saturation), 1 + saturation]. Should be non negative numbers. Default: 0.
        hue (float, optional): How much to jitter hue.
            Chosen uniformly from [-hue, hue]. Should have 0<= hue <= 0.5. Default: 0.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A color jittered image.

    Returns:
        A callable object of ColorJitter.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import ColorJitter

            transform = ColorJitter(0.4, 0.4, 0.4, 0.4)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)

    """

    def __init__(
        self, brightness=0, contrast=0, saturation=0, hue=0, keys=None
    ):
        super().__init__(keys)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _get_param(self, brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            transforms.append(BrightnessTransform(brightness, self.keys))

        if contrast is not None:
            transforms.append(ContrastTransform(contrast, self.keys))

        if saturation is not None:
            transforms.append(SaturationTransform(saturation, self.keys))

        if hue is not None:
            transforms.append(HueTransform(hue, self.keys))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def _apply_image(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self._get_param(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        return transform(img)


class RandomCrop(BaseTransform):
    """Crops the given CV Image at a random location.

    Args:
        size (sequence|int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int|sequence, optional): Optional padding on each border
            of the image. If a sequence of length 4 is provided, it is used to pad left,
            top, right, bottom borders respectively. Default: None, without padding.
        pad_if_needed (boolean, optional): It will pad the image if smaller than the
            desired size to avoid raising an exception. Default: False.
        fill (float|tuple, optional): Pixel fill value for constant fill. If a tuple of
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
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A random cropped image.

    Returns:
        A callable object of RandomCrop.

    Examples:

        .. code-block:: python
          :name: code-example1

            import paddle
            from paddle.vision.transforms import RandomCrop
            transform = RandomCrop(224)

            fake_img = paddle.randint(0, 255, shape=(3, 324,300), dtype = 'int32')
            print(fake_img.shape) # [3, 324, 300]

            crop_img = transform(fake_img)
            print(crop_img.shape) # [3, 224, 224]
    """

    def __init__(
        self,
        size,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode='constant',
        keys=None,
    ):
        super().__init__(keys)
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def _get_param(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        if paddle.in_dynamic_mode():
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            i = paddle.randint(low=0, high=h - th)
            j = paddle.randint(low=0, high=w - tw)
        return i, j, th, tw

    def _apply_image(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        w, h = _get_image_size(img)

        # pad the width if needed
        if self.pad_if_needed and w < self.size[1]:
            img = F.pad(
                img, (self.size[1] - w, 0), self.fill, self.padding_mode
            )
        # pad the height if needed
        if self.pad_if_needed and h < self.size[0]:
            img = F.pad(
                img, (0, self.size[0] - h), self.fill, self.padding_mode
            )

        i, j, h, w = self._get_param(img, self.size)

        return F.crop(img, i, j, h, w)


class Pad(BaseTransform):
    """Pads the given CV Image on all sides with the given "pad" value.

    Args:
        padding (int|list|tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If list/tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a list/tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int|list|tuple): Pixel fill value for constant fill. Default is 0. If a list/tuple of
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
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A paded image.

    Returns:
        A callable object of Pad.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import Pad

            transform = Pad(2)

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)
    """

    def __init__(self, padding, fill=0, padding_mode='constant', keys=None):
        assert isinstance(padding, (numbers.Number, list, tuple))
        assert isinstance(fill, (numbers.Number, str, list, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        if isinstance(padding, list):
            padding = tuple(padding)
        if isinstance(fill, list):
            fill = tuple(fill)

        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError(
                "Padding must be an int or a 2, or 4 element tuple, not a "
                + f"{len(padding)} element tuple"
            )

        super().__init__(keys)
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def _apply_image(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode)


def _check_sequence_input(x, name, req_sizes):
    msg = (
        req_sizes[0]
        if len(req_sizes) < 2
        else " or ".join([str(s) for s in req_sizes])
    )
    if not isinstance(x, Sequence):
        raise TypeError(f"{name} should be a sequence of length {msg}.")
    if len(x) not in req_sizes:
        raise ValueError(f"{name} should be sequence of length {msg}.")


def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(
                f"If {name} is a single number, it must be positive."
            )
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


class RandomAffine(BaseTransform):
    """Random affine transformation of the image.

    Args:
        degrees (int|float|tuple): The angle interval of the random rotation.
            If set as a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees) in clockwise order. If set 0, will not rotate.
        translate (tuple, optional): Maximum absolute fraction for horizontal and vertical translations.
            For example translate=(a, b), then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a
            and vertical shift is randomly sampled in the range -img_height * b < dy < img_height * b.
            Default is None, will not translate.
        scale (tuple, optional): Scaling factor interval, e.g (a, b), then scale is randomly sampled from the range a <= scale <= b.
            Default is None, will keep original scale and not scale.
        shear (sequence or number, optional): Range of degrees to shear, ranges from -180 to 180 in clockwise order.
            If set as a number, a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            Else if set as a sequence of 2 values a shear parallel to the x axis in the range (shear[0], shear[1]) will be applied.
            Else if set as a sequence of 4 values, a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Default is None, will not apply shear.
        interpolation (str, optional): Interpolation method. If omitted, or if the
            image has only one channel, it is set to PIL.Image.NEAREST or cv2.INTER_NEAREST
            according the backend.
            When use pil backend, support method are as following:
            - "nearest": Image.NEAREST,
            - "bilinear": Image.BILINEAR,
            - "bicubic": Image.BICUBIC
            When use cv2 backend, support method are as following:
            - "nearest": cv2.INTER_NEAREST,
            - "bilinear": cv2.INTER_LINEAR,
            - "bicubic": cv2.INTER_CUBIC
        fill (int|list|tuple, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        center (2-tuple, optional): Optional center of rotation, (x, y).
            Origin is the upper left corner.
            Default is the center of the image.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): An affined image.

    Returns:
        A callable object of RandomAffine.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.vision.transforms import RandomAffine

            transform = RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[-10, 10])

            fake_img = paddle.randn((3, 256, 300)).astype(paddle.float32)

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(
        self,
        degrees,
        translate=None,
        scale=None,
        shear=None,
        interpolation='nearest',
        fill=0,
        center=None,
        keys=None,
    ):
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        super().__init__(keys)
        assert interpolation in ['nearest', 'bilinear', 'bicubic']
        self.interpolation = interpolation

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError(
                        "translation values should be between 0 and 1"
                    )
        self.translate = translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")
        self.fill = fill

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))
        self.center = center

    def _get_param(
        self, img_size, degrees, translate=None, scale_ranges=None, shears=None
    ):
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])

        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(random.uniform(-max_dx, max_dx))
            ty = int(random.uniform(-max_dy, max_dy))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        shear_x, shear_y = 0.0, 0.0
        if shears is not None:
            shear_x = random.uniform(shears[0], shears[1])
            if len(shears) == 4:
                shear_y = random.uniform(shears[2], shears[3])
        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def _apply_image(self, img):
        """
        Args:
            img (PIL.Image|np.array): Image to be affine transformed.

        Returns:
            PIL.Image or np.array: Affine transformed image.
        """

        w, h = _get_image_size(img)
        img_size = [w, h]

        ret = self._get_param(
            img_size, self.degrees, self.translate, self.scale, self.shear
        )

        return F.affine(
            img,
            *ret,
            interpolation=self.interpolation,
            fill=self.fill,
            center=self.center,
        )


class RandomRotation(BaseTransform):
    """Rotates the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees) clockwise order.
        interpolation (str, optional): Interpolation method. If omitted, or if the
            image has only one channel, it is set to PIL.Image.NEAREST or cv2.INTER_NEAREST
            according the backend. when use pil backend, support method are as following:
            - "nearest": Image.NEAREST,
            - "bilinear": Image.BILINEAR,
            - "bicubic": Image.BICUBIC
            when use cv2 backend, support method are as following:
            - "nearest": cv2.INTER_NEAREST,
            - "bilinear": cv2.INTER_LINEAR,
            - "bicubic": cv2.INTER_CUBIC
        expand (bool|optional): Optional expansion flag. Default: False.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple|optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A rotated image.

    Returns:
        A callable object of RandomRotation.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import RandomRotation

            transform = RandomRotation(90)

            fake_img = Image.fromarray((np.random.rand(200, 150, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)
    """

    def __init__(
        self,
        degrees,
        interpolation='nearest',
        expand=False,
        center=None,
        fill=0,
        keys=None,
    ):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive."
                )
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2."
                )
            self.degrees = degrees

        super().__init__(keys)
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill

    def _get_param(self, degrees):
        if paddle.in_dynamic_mode():
            angle = random.uniform(degrees[0], degrees[1])
        else:
            angle = paddle.uniform(
                [1], dtype="float32", min=degrees[0], max=degrees[1]
            )

        return angle

    def _apply_image(self, img):
        """
        Args:
            img (PIL.Image|np.array): Image to be rotated.

        Returns:
            PIL.Image or np.array: Rotated image.
        """

        angle = self._get_param(self.degrees)

        return F.rotate(
            img, angle, self.interpolation, self.expand, self.center, self.fill
        )


class RandomPerspective(BaseTransform):
    """Random perspective transformation with a given probability.

    Args:
        prob (float, optional): Probability of using transformation, ranges from
            0 to 1, default is 0.5.
        distortion_scale (float, optional): Degree of distortion, ranges from
            0 to 1, default is 0.5.
        interpolation (str, optional): Interpolation method. If omitted, or if
            the image has only one channel, it is set to PIL.Image.NEAREST or
            cv2.INTER_NEAREST.
            When use pil backend, support method are as following:
            - "nearest": Image.NEAREST,
            - "bilinear": Image.BILINEAR,
            - "bicubic": Image.BICUBIC
            When use cv2 backend, support method are as following:
            - "nearest": cv2.INTER_NEAREST,
            - "bilinear": cv2.INTER_LINEAR,
            - "bicubic": cv2.INTER_CUBIC
        fill (int|list|tuple, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A perspectived image.

    Returns:
        A callable object of RandomPerspective.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.vision.transforms import RandomPerspective

            transform = RandomPerspective(prob=1.0, distortion_scale=0.9)

            fake_img = paddle.randn((3, 200, 150)).astype(paddle.float32)

            fake_img = transform(fake_img)
            print(fake_img.shape)
    """

    def __init__(
        self,
        prob=0.5,
        distortion_scale=0.5,
        interpolation='nearest',
        fill=0,
        keys=None,
    ):
        super().__init__(keys)
        assert 0 <= prob <= 1, "probability must be between 0 and 1"
        assert (
            0 <= distortion_scale <= 1
        ), "distortion_scale must be between 0 and 1"
        assert interpolation in ['nearest', 'bilinear', 'bicubic']
        assert isinstance(fill, (numbers.Number, str, list, tuple))

        self.prob = prob
        self.distortion_scale = distortion_scale
        self.interpolation = interpolation
        self.fill = fill

    def get_params(self, width, height, distortion_scale):
        """
        Returns:
            startpoints (list[list[int]]): [top-left, top-right, bottom-right, bottom-left] of the original image,
            endpoints (list[list[int]]): [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(random.uniform(0, int(distortion_scale * half_width) + 1)),
            int(random.uniform(0, int(distortion_scale * half_height) + 1)),
        ]
        topright = [
            int(
                random.uniform(
                    width - int(distortion_scale * half_width) - 1, width
                )
            ),
            int(random.uniform(0, int(distortion_scale * half_height) + 1)),
        ]
        botright = [
            int(
                random.uniform(
                    width - int(distortion_scale * half_width) - 1, width
                )
            ),
            int(
                random.uniform(
                    height - int(distortion_scale * half_height) - 1, height
                )
            ),
        ]
        botleft = [
            int(random.uniform(0, int(distortion_scale * half_width) + 1)),
            int(
                random.uniform(
                    height - int(distortion_scale * half_height) - 1, height
                )
            ),
        ]
        startpoints = [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]
        endpoints = [topleft, topright, botright, botleft]

        return startpoints, endpoints

    def _apply_image(self, img):
        """
        Args:
            img (PIL.Image|np.array|paddle.Tensor): Image to be Perspectively transformed.

        Returns:
            PIL.Image|np.array|paddle.Tensor: Perspectively transformed image.
        """

        width, height = _get_image_size(img)

        if random.random() < self.prob:
            startpoints, endpoints = self.get_params(
                width, height, self.distortion_scale
            )
            return F.perspective(
                img, startpoints, endpoints, self.interpolation, self.fill
            )
        return img


class Grayscale(BaseTransform):
    """Converts image to grayscale.

    Args:
        num_output_channels (int, optional): (1 or 3) number of channels desired for output image. Default: 1.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): Grayscale version of the input image.
            - If output_channels == 1 : returned image is single channel
            - If output_channels == 3 : returned image is 3 channel with r == g == b

    Returns:
        A callable object of Grayscale.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import Grayscale

            transform = Grayscale()

            fake_img = Image.fromarray((np.random.rand(224, 224, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(np.array(fake_img).shape)
    """

    def __init__(self, num_output_channels=1, keys=None):
        super().__init__(keys)
        self.num_output_channels = num_output_channels

    def _apply_image(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        return F.to_grayscale(img, self.num_output_channels)


class RandomErasing(BaseTransform):
    """Erase the pixels in a rectangle region selected randomly.

    Args:
        prob (float, optional): Probability of the input data being erased. Default: 0.5.
        scale (sequence, optional): The proportional range of the erased area to the input image.
                                    Default: (0.02, 0.33).
        ratio (sequence, optional): Aspect ratio range of the erased area. Default: (0.3, 3.3).
        value (int|float|sequence|str, optional): The value each pixel in erased area will be replaced with.
                               If value is a single number, all pixels will be erased with this value.
                               If value is a sequence with length 3, the R, G, B channels will be ereased
                               respectively. If value is set to "random", each pixel will be erased with
                               random values. Default: 0.
        inplace (bool, optional): Whether this transform is inplace. Default: False.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(paddle.Tensor | np.array | PIL.Image): The input image. For Tensor input, the shape should be (C, H, W).
                 For np.array input, the shape should be (H, W, C).
        - output(paddle.Tensor | np.array | PIL.Image): A random erased image.

    Returns:
        A callable object of RandomErasing.

    Examples:

        .. code-block:: python

            import paddle

            fake_img = paddle.randn((3, 10, 10)).astype(paddle.float32)
            transform = paddle.vision.transforms.RandomErasing()
            result = transform(fake_img)

            print(result)
    """

    def __init__(
        self,
        prob=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value=0,
        inplace=False,
        keys=None,
    ):
        super().__init__(keys)
        assert isinstance(
            scale, (tuple, list)
        ), "scale should be a tuple or list"
        assert (
            scale[0] >= 0 and scale[1] <= 1 and scale[0] <= scale[1]
        ), "scale should be of kind (min, max) and in range [0, 1]"
        assert isinstance(
            ratio, (tuple, list)
        ), "ratio should be a tuple or list"
        assert (
            ratio[0] >= 0 and ratio[0] <= ratio[1]
        ), "ratio should be of kind (min, max)"
        assert (
            prob >= 0 and prob <= 1
        ), "The probability should be in range [0, 1]"
        assert isinstance(
            value, (numbers.Number, str, tuple, list)
        ), "value should be a number, tuple, list or str"
        if isinstance(value, str) and value != "random":
            raise ValueError("value must be 'random' when type is str")

        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def _dynamic_get_param(self, img, scale, ratio, value):
        """Get parameters for ``erase`` for a random erasing in dynamic mode.

        Args:
            img (paddle.Tensor | np.array | PIL.Image): Image to be erased.
            scale (sequence, optional): The proportional range of the erased area to the input image.
            ratio (sequence, optional): Aspect ratio range of the erased area.
            value (sequence | None): The value each pixel in erased area will be replaced with.
                               If value is a sequence with length 3, the R, G, B channels will be ereased
                               respectively. If value is None, each pixel will be erased with random values.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erase.
        """
        if F._is_pil_image(img):
            shape = np.asarray(img).astype(np.uint8).shape
            h, w, c = shape[-3], shape[-2], shape[-1]
        elif F._is_numpy_image(img):
            h, w, c = img.shape[-3], img.shape[-2], img.shape[-1]
        elif F._is_tensor_image(img):
            c, h, w = img.shape[-3], img.shape[-2], img.shape[-1]

        img_area = h * w
        log_ratio = np.log(ratio)
        for _ in range(10):
            erase_area = np.random.uniform(*scale) * img_area
            aspect_ratio = np.exp(np.random.uniform(*log_ratio))
            erase_h = int(round(np.sqrt(erase_area * aspect_ratio)))
            erase_w = int(round(np.sqrt(erase_area / aspect_ratio)))
            if erase_h >= h or erase_w >= w:
                continue
            if F._is_tensor_image(img):
                if value is None:
                    v = paddle.normal(shape=[c, erase_h, erase_w]).astype(
                        img.dtype
                    )
                else:
                    v = paddle.to_tensor(value, dtype=img.dtype)[:, None, None]
            else:
                if value is None:
                    v = np.random.normal(size=[erase_h, erase_w, c]) * 255
                else:
                    v = np.array(value)[None, None, :]
            top = np.random.randint(0, h - erase_h + 1)
            left = np.random.randint(0, w - erase_w + 1)

            return top, left, erase_h, erase_w, v

        return 0, 0, h, w, img

    def _static_get_param(self, img, scale, ratio, value):
        """Get parameters for ``erase`` for a random erasing in static mode.

        Args:
            img (paddle.static.Variable): Image to be erased.
            scale (sequence, optional): The proportional range of the erased area to the input image.
            ratio (sequence, optional): Aspect ratio range of the erased area.
            value (sequence | None): The value each pixel in erased area will be replaced with.
                               If value is a sequence with length 3, the R, G, B channels will be ereased
                               respectively. If value is None, each pixel will be erased with random values.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erase.
        """

        c, h, w = img.shape[-3], img.shape[-2], img.shape[-1]

        img_area = h * w
        log_ratio = np.log(np.array(ratio))

        def cond(counter, ten, erase_h, erase_w):
            return counter < ten and (erase_h >= h or erase_w >= w)

        def body(counter, ten, erase_h, erase_w):

            erase_area = (
                paddle.uniform([1], min=scale[0], max=scale[1]) * img_area
            )
            aspect_ratio = paddle.exp(
                paddle.uniform([1], min=log_ratio[0], max=log_ratio[1])
            )
            erase_h = paddle.round(paddle.sqrt(erase_area * aspect_ratio)).cast(
                "int32"
            )
            erase_w = paddle.round(paddle.sqrt(erase_area / aspect_ratio)).cast(
                "int32"
            )

            counter += 1

            return [counter, ten, erase_h, erase_w]

        h = paddle.assign([h]).astype("int32")
        w = paddle.assign([w]).astype("int32")
        erase_h, erase_w = h.clone(), w.clone()
        counter = paddle.full(
            shape=[1], fill_value=0, dtype='int32'
        )  # loop counter
        ten = paddle.full(
            shape=[1], fill_value=10, dtype='int32'
        )  # loop length
        counter, ten, erase_h, erase_w = paddle.static.nn.while_loop(
            cond, body, [counter, ten, erase_h, erase_w]
        )

        if value is None:
            v = paddle.normal(shape=[c, erase_h, erase_w]).astype(img.dtype)
        else:
            v = value[:, None, None]

        zero = paddle.zeros([1]).astype("int32")
        top = paddle.static.nn.cond(
            erase_h < h and erase_w < w,
            lambda: paddle.uniform(
                shape=[1], min=0, max=h - erase_h + 1
            ).astype("int32"),
            lambda: zero,
        )

        left = paddle.static.nn.cond(
            erase_h < h and erase_w < w,
            lambda: paddle.uniform(
                shape=[1], min=0, max=w - erase_w + 1
            ).astype("int32"),
            lambda: zero,
        )

        erase_h = paddle.static.nn.cond(
            erase_h < h and erase_w < w, lambda: erase_h, lambda: h
        )

        erase_w = paddle.static.nn.cond(
            erase_h < h and erase_w < w, lambda: erase_w, lambda: w
        )

        v = paddle.static.nn.cond(
            erase_h < h and erase_w < w, lambda: v, lambda: img
        )

        return top, left, erase_h, erase_w, v, counter

    def _dynamic_apply_image(self, img):
        """
        Args:
            img (paddle.Tensor | np.array | PIL.Image): Image to be Erased.

        Returns:
            output (paddle.Tensor | np.array | PIL.Image): A random erased image.
        """

        if random.random() < self.prob:
            if isinstance(self.value, numbers.Number):
                value = [self.value]
            elif isinstance(self.value, str):
                value = None
            else:
                value = self.value
            if value is not None and not (len(value) == 1 or len(value) == 3):
                raise ValueError(
                    "Value should be a single number or a sequence with length equals to image's channel."
                )
            top, left, erase_h, erase_w, v = self._dynamic_get_param(
                img, self.scale, self.ratio, value
            )
            return F.erase(img, top, left, erase_h, erase_w, v, self.inplace)
        return img

    def _static_apply_image(self, img):
        """
        Args:
            img (paddle.static.Variable): Image to be Erased.

        Returns:
            output (paddle.static.Variable): A random erased image.
        """

        if isinstance(self.value, numbers.Number):
            value = paddle.assign([self.value]).astype(img.dtype)
        elif isinstance(self.value, str):
            value = None
        else:
            value = paddle.assign(self.value).astype(img.dtype)
        if value is not None and not (
            value.shape[0] == 1 or value.shape[0] == 3
        ):
            raise ValueError(
                "Value should be a single number or a sequence with length equals to image's channel."
            )

        top, left, erase_h, erase_w, v, counter = self._static_get_param(
            img, self.scale, self.ratio, value
        )
        return F.erase(img, top, left, erase_h, erase_w, v, self.inplace)

    def _apply_image(self, img):
        if paddle.in_dynamic_mode():
            return self._dynamic_apply_image(img)
        else:
            return paddle.static.nn.cond(
                paddle.rand([1]) < self.prob,
                lambda: self._static_apply_image(img),
                lambda: img,
            )
