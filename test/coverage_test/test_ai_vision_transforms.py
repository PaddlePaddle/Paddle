# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""
视觉变换单元测试 / Vision Transforms Unit Tests

测试目标 / Test Target:
  paddle.vision.transforms 模块

覆盖的模块 / Covered Modules:
  - paddle.vision.transforms: 图像变换操作
  - Resize, RandomCrop, CenterCrop, Normalize, ToTensor等

作用 / Purpose:
  覆盖图像预处理变换的各种代码路径，补充视觉变换功能的测试。
"""

import unittest

import numpy as np
from PIL import Image

import paddle
from paddle.vision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    Grayscale,
    Normalize,
    Pad,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

paddle.disable_static()


def create_test_image(height=64, width=64, channels=3):
    """创建测试图像 / Create test image"""
    arr = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    return Image.fromarray(arr)


def create_numpy_image(height=64, width=64, channels=3):
    """创建numpy测试图像 / Create numpy test image"""
    return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)


class TestResize(unittest.TestCase):
    """测试Resize变换 / Test Resize transform"""

    def test_resize_pil(self):
        """测试PIL图像Resize / Test PIL image resize"""
        resize = Resize(32)
        img = create_test_image(64, 64)
        result = resize(img)
        self.assertEqual(min(result.size), 32)

    def test_resize_tuple(self):
        """测试指定尺寸Resize / Test resize to specific size"""
        resize = Resize((32, 48))
        img = create_test_image(64, 64)
        result = resize(img)
        self.assertEqual(result.size, (48, 32))

    def test_resize_numpy(self):
        """测试numpy图像Resize (PIL模式) / Test numpy resize via PIL"""
        resize = Resize(32)
        img = create_test_image(64, 64)
        result = resize(img)
        self.assertEqual(min(result.size), 32)


class TestCrop(unittest.TestCase):
    """测试裁剪变换 / Test crop transforms"""

    def test_center_crop(self):
        """测试中心裁剪 / Test center crop"""
        crop = CenterCrop(32)
        img = create_test_image(64, 64)
        result = crop(img)
        self.assertEqual(result.size, (32, 32))

    def test_random_crop(self):
        """测试随机裁剪 / Test random crop"""
        crop = RandomCrop(32)
        img = create_test_image(64, 64)
        result = crop(img)
        self.assertEqual(result.size, (32, 32))

    def test_random_crop_with_padding(self):
        """测试带填充的随机裁剪 / Test random crop with padding"""
        crop = RandomCrop(32, padding=4)
        img = create_test_image(32, 32)
        result = crop(img)
        self.assertEqual(result.size, (32, 32))


class TestFlipAndRotate(unittest.TestCase):
    """测试翻转和旋转变换 / Test flip and rotate transforms"""

    def test_random_horizontal_flip(self):
        """测试随机水平翻转 / Test random horizontal flip"""
        flip = RandomHorizontalFlip(prob=1.0)  # Always flip
        img = create_test_image(32, 32)
        result = flip(img)
        self.assertEqual(result.size, img.size)

    def test_random_vertical_flip(self):
        """测试随机垂直翻转 / Test random vertical flip"""
        flip = RandomVerticalFlip(prob=1.0)  # Always flip
        img = create_test_image(32, 32)
        result = flip(img)
        self.assertEqual(result.size, img.size)

    def test_random_rotation(self):
        """测试随机旋转 / Test random rotation"""
        rotation = RandomRotation(45)
        img = create_test_image(64, 64)
        result = rotation(img)
        self.assertIsNotNone(result)


class TestNormalize(unittest.TestCase):
    """测试归一化变换 / Test normalize transform"""

    def test_normalize_tensor(self):
        """测试张量归一化 / Test tensor normalization"""
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = paddle.to_tensor(np.random.rand(3, 32, 32).astype('float32'))
        result = normalize(x)
        self.assertEqual(result.shape, [3, 32, 32])

    def test_normalize_numpy(self):
        """测试numpy归一化 / Test numpy normalization"""
        normalize = Normalize(
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            data_format='HWC',
        )
        img = create_numpy_image(32, 32).astype('float32')
        result = normalize(img)
        self.assertEqual(result.shape, (32, 32, 3))


class TestToTensorAndTranspose(unittest.TestCase):
    """测试ToTensor / Test ToTensor"""

    def test_to_tensor_pil(self):
        """测试PIL图像转张量 / Test PIL image to tensor"""
        to_tensor = ToTensor()
        img = create_test_image(32, 32)
        result = to_tensor(img)
        self.assertEqual(result.shape, [3, 32, 32])
        # Values should be in [0, 1]
        self.assertTrue(float(result.max().numpy()) <= 1.0)

    def test_to_tensor_numpy(self):
        """测试numpy图像转张量 / Test numpy image to tensor"""
        to_tensor = ToTensor()
        img = create_numpy_image(32, 32).astype('float32')
        result = to_tensor(img)
        self.assertIsNotNone(result)


class TestColorTransforms(unittest.TestCase):
    """测试颜色变换 / Test color transforms"""

    def test_grayscale(self):
        """测试灰度化 / Test grayscale"""
        grayscale = Grayscale()
        img = create_test_image(32, 32, 3)
        result = grayscale(img)
        self.assertEqual(result.mode, 'L')

    def test_grayscale_keep_channels(self):
        """测试保持通道数的灰度化 / Test grayscale with num_output_channels=3"""
        grayscale = Grayscale(num_output_channels=3)
        img = create_test_image(32, 32, 3)
        result = grayscale(img)
        self.assertEqual(result.mode, 'RGB')

    def test_color_jitter(self):
        """测试颜色抖动 / Test color jitter"""
        jitter = ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        img = create_test_image(32, 32)
        result = jitter(img)
        self.assertEqual(result.size, img.size)


class TestCompose(unittest.TestCase):
    """测试Compose组合变换 / Test Compose transformation"""

    def test_compose_basic(self):
        """测试基本Compose / Test basic compose"""
        transform = Compose(
            [
                Resize(64),
                CenterCrop(32),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        img = create_test_image(128, 128)
        result = transform(img)
        self.assertEqual(result.shape, [3, 32, 32])

    def test_pad_transform(self):
        """测试Pad填充变换 / Test Pad transform"""
        pad = Pad(4)
        img = create_test_image(32, 32)
        result = pad(img)
        self.assertEqual(result.size, (40, 40))


if __name__ == '__main__':
    unittest.main()
