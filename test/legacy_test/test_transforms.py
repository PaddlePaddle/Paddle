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

import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

import paddle
import paddle.vision.transforms.functional as F
from paddle.vision import image_load, set_image_backend
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms


class TestTransformsCV2(unittest.TestCase):
    def setUp(self):
        self.backend = self.get_backend()
        set_image_backend(self.backend)
        self.data_dir = tempfile.mkdtemp()
        for i in range(2):
            sub_dir = os.path.join(self.data_dir, 'class_' + str(i))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            for j in range(2):
                if j == 0:
                    fake_img = (np.random.random((280, 350, 3)) * 255).astype(
                        'uint8'
                    )
                else:
                    fake_img = (np.random.random((400, 300, 3)) * 255).astype(
                        'uint8'
                    )
                cv2.imwrite(os.path.join(sub_dir, str(j) + '.jpg'), fake_img)

    def get_backend(self):
        return 'cv2'

    def create_image(self, shape):
        if self.backend == 'cv2':
            return (np.random.rand(*shape) * 255).astype('uint8')
        elif self.backend == 'pil':
            return Image.fromarray(
                (np.random.rand(*shape) * 255).astype('uint8')
            )

    def get_shape(self, img):
        if isinstance(img, paddle.Tensor):
            return img.shape

        elif self.backend == 'pil':
            return np.array(img).shape

        return img.shape

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def do_transform(self, trans):
        dataset_folder = DatasetFolder(self.data_dir, transform=trans)

        for _ in dataset_folder:
            pass

    def test_trans_all(self):
        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.120, 57.375],
        )
        trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4
                ),
                transforms.RandomHorizontalFlip(),
                transforms.Transpose(),
                normalize,
            ]
        )

        self.do_transform(trans)

    def test_normalize(self):
        normalize = transforms.Normalize(mean=0.5, std=0.5)
        trans = transforms.Compose([transforms.Transpose(), normalize])
        self.do_transform(trans)

    def test_trans_resize(self):
        trans = transforms.Compose(
            [
                transforms.Resize(300),
                transforms.RandomResizedCrop((280, 280)),
                transforms.Resize(280),
                transforms.Resize((256, 200)),
                transforms.Resize((180, 160)),
                transforms.CenterCrop(128),
                transforms.CenterCrop((128, 128)),
            ]
        )
        self.do_transform(trans)

    def test_flip(self):
        trans = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(1.0),
                transforms.RandomHorizontalFlip(0.0),
                transforms.RandomVerticalFlip(0.0),
                transforms.RandomVerticalFlip(1.0),
            ]
        )
        self.do_transform(trans)

    def test_color_jitter(self):
        trans = transforms.Compose(
            [
                transforms.BrightnessTransform(0.0),
                transforms.HueTransform(0.0),
                transforms.SaturationTransform(0.0),
                transforms.ContrastTransform(0.0),
            ]
        )
        self.do_transform(trans)

    def test_affine(self):
        trans = transforms.Compose(
            [
                transforms.RandomAffine(90),
                transforms.RandomAffine([-10, 10], translate=[0.1, 0.3]),
                transforms.RandomAffine(
                    45, translate=[0.2, 0.2], scale=[0.2, 0.5]
                ),
                transforms.RandomAffine(
                    10, translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[-10, 10]
                ),
                transforms.RandomAffine(
                    10,
                    translate=[0.5, 0.3],
                    scale=[0.7, 1.3],
                    shear=[-10, 10, 20, 40],
                ),
                transforms.RandomAffine(
                    10,
                    translate=[0.5, 0.3],
                    scale=[0.7, 1.3],
                    shear=[-10, 10, 20, 40],
                    interpolation='bilinear',
                ),
                transforms.RandomAffine(
                    10,
                    translate=[0.5, 0.3],
                    scale=[0.7, 1.3],
                    shear=[-10, 10, 20, 40],
                    interpolation='bilinear',
                    fill=114,
                ),
                transforms.RandomAffine(
                    10,
                    translate=[0.5, 0.3],
                    scale=[0.7, 1.3],
                    shear=[-10, 10, 20, 40],
                    interpolation='bilinear',
                    fill=114,
                    center=(60, 80),
                ),
            ]
        )
        self.do_transform(trans)

    def test_rotate(self):
        trans = transforms.Compose(
            [
                transforms.RandomRotation(90),
                transforms.RandomRotation([-10, 10]),
                transforms.RandomRotation(45, expand=True),
                transforms.RandomRotation(10, expand=True, center=(60, 80)),
            ]
        )
        self.do_transform(trans)

    def test_perspective(self):
        trans = transforms.Compose(
            [
                transforms.RandomPerspective(prob=1.0),
                transforms.RandomPerspective(prob=1.0, distortion_scale=0.9),
            ]
        )
        self.do_transform(trans)

    def test_pad(self):
        trans = transforms.Compose([transforms.Pad(2)])
        self.do_transform(trans)

        fake_img = self.create_image((200, 150, 3))
        trans_pad = transforms.Pad(10)
        fake_img_padded = trans_pad(fake_img)
        np.testing.assert_equal(self.get_shape(fake_img_padded), (220, 170, 3))
        trans_pad1 = transforms.Pad([1, 2])
        trans_pad2 = transforms.Pad([1, 2, 3, 4])
        img = trans_pad1(fake_img)
        img = trans_pad2(img)

    def test_random_crop(self):
        trans = transforms.Compose(
            [
                transforms.RandomCrop(200),
                transforms.RandomCrop((140, 160)),
            ]
        )
        self.do_transform(trans)

        trans_random_crop1 = transforms.RandomCrop(224)
        trans_random_crop2 = transforms.RandomCrop((140, 160))

        fake_img = self.create_image((500, 400, 3))
        fake_img_crop1 = trans_random_crop1(fake_img)
        fake_img_crop2 = trans_random_crop2(fake_img_crop1)

        np.testing.assert_equal(self.get_shape(fake_img_crop1), (224, 224, 3))

        np.testing.assert_equal(self.get_shape(fake_img_crop2), (140, 160, 3))

        trans_random_crop_same = transforms.RandomCrop((140, 160))
        img = trans_random_crop_same(fake_img_crop2)

        trans_random_crop_bigger = transforms.RandomCrop(
            (180, 200), pad_if_needed=True
        )
        img = trans_random_crop_bigger(img)

        trans_random_crop_pad = transforms.RandomCrop((224, 256), 2, True)
        img = trans_random_crop_pad(img)

    def test_erase(self):
        trans = transforms.Compose(
            [
                transforms.RandomErasing(),
                transforms.RandomErasing(value="random"),
            ]
        )
        self.do_transform(trans)

    def test_grayscale(self):
        trans = transforms.Compose([transforms.Grayscale()])
        self.do_transform(trans)

        trans_gray = transforms.Grayscale()
        fake_img = self.create_image((500, 400, 3))
        fake_img_gray = trans_gray(fake_img)

        np.testing.assert_equal(self.get_shape(fake_img_gray)[0], 500)
        np.testing.assert_equal(self.get_shape(fake_img_gray)[1], 400)

        trans_gray3 = transforms.Grayscale(3)
        fake_img = self.create_image((500, 400, 3))
        fake_img_gray = trans_gray3(fake_img)

    def test_transpose(self):
        trans = transforms.Compose([transforms.Transpose()])
        self.do_transform(trans)

        fake_img = self.create_image((50, 100, 3))
        converted_img = trans(fake_img)

        np.testing.assert_equal(self.get_shape(converted_img), (3, 50, 100))

    def test_to_tensor(self):
        trans = transforms.Compose([transforms.ToTensor()])
        fake_img = self.create_image((50, 100, 3))

        tensor = trans(fake_img)

        assert isinstance(tensor, paddle.Tensor)
        np.testing.assert_equal(tensor.shape, (3, 50, 100))

    def test_keys(self):
        fake_img1 = self.create_image((200, 150, 3))
        fake_img2 = self.create_image((200, 150, 3))
        trans_pad = transforms.Pad(10, keys=("image",))
        fake_img_padded = trans_pad((fake_img1, fake_img2))

    def test_exception(self):
        trans = transforms.Compose([transforms.Resize(-1)])

        trans_batch = transforms.Compose([transforms.Resize(-1)])

        with self.assertRaises((cv2.error, AssertionError, ValueError)):
            self.do_transform(trans)

        with self.assertRaises((cv2.error, AssertionError, ValueError)):
            self.do_transform(trans_batch)

        with self.assertRaises(ValueError):
            transforms.ContrastTransform(-1.0)

        with self.assertRaises(ValueError):
            transforms.SaturationTransform(-1.0),

        with self.assertRaises(ValueError):
            transforms.HueTransform(-1.0)

        with self.assertRaises(ValueError):
            transforms.BrightnessTransform(-1.0)

        with self.assertRaises(ValueError):
            transforms.Pad([1.0, 2.0, 3.0])

        with self.assertRaises(TypeError):
            fake_img = self.create_image((100, 120, 3))
            F.pad(fake_img, '1')

        with self.assertRaises(TypeError):
            fake_img = self.create_image((100, 120, 3))
            F.pad(fake_img, 1, {})

        with self.assertRaises(TypeError):
            fake_img = self.create_image((100, 120, 3))
            F.pad(fake_img, 1, padding_mode=-1)

        with self.assertRaises(ValueError):
            fake_img = self.create_image((100, 120, 3))
            F.pad(fake_img, [1.0, 2.0, 3.0])

        with self.assertRaises(TypeError):
            tensor_img = paddle.rand((3, 100, 100))
            F.pad(tensor_img, '1')

        with self.assertRaises(TypeError):
            tensor_img = paddle.rand((3, 100, 100))
            F.pad(tensor_img, 1, {})

        with self.assertRaises(TypeError):
            tensor_img = paddle.rand((3, 100, 100))
            F.pad(tensor_img, 1, padding_mode=-1)

        with self.assertRaises(ValueError):
            tensor_img = paddle.rand((3, 100, 100))
            F.pad(tensor_img, [1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            transforms.RandomAffine(-10)

        with self.assertRaises(ValueError):
            transforms.RandomAffine([-30, 60], translate=[2, 2])

        with self.assertRaises(ValueError):
            transforms.RandomAffine(10, translate=[0.2, 0.2], scale=[1, 2, 3]),

        with self.assertRaises(ValueError):
            transforms.RandomAffine(
                10, translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[1, 2, 3]
            ),

        with self.assertRaises(ValueError):
            transforms.RandomAffine(
                10,
                translate=[0.5, 0.3],
                scale=[0.7, 1.3],
                shear=[-10, 10, 0, 20, 40],
            )

        with self.assertRaises(ValueError):
            transforms.RandomAffine(
                10,
                translate=[0.5, 0.3],
                scale=[0.7, 1.3],
                shear=[-10, 10, 20, 40],
                fill=114,
                center=(1, 2, 3),
            )

        with self.assertRaises(ValueError):
            transforms.RandomRotation(-2)

        with self.assertRaises(ValueError):
            transforms.RandomRotation([1, 2, 3])

        with self.assertRaises(ValueError):
            trans_gray = transforms.Grayscale(5)
            fake_img = self.create_image((100, 120, 3))
            trans_gray(fake_img)

        with self.assertRaises(TypeError):
            transform = transforms.RandomResizedCrop(64)
            transform(1)

        with self.assertRaises(ValueError):
            transform = transforms.BrightnessTransform([-0.1, -0.2])

        with self.assertRaises(TypeError):
            transform = transforms.BrightnessTransform('0.1')

        with self.assertRaises(ValueError):
            transform = transforms.BrightnessTransform('0.1', keys=1)

        with self.assertRaises(NotImplementedError):
            transform = transforms.BrightnessTransform('0.1', keys='a')

        with self.assertRaisesRegex(
            AssertionError, "scale should be a tuple or list"
        ):
            transform = transforms.RandomErasing(scale=0.5)

        with self.assertRaisesRegex(
            AssertionError, "ratio should be a tuple or list"
        ):
            transform = transforms.RandomErasing(ratio=0.8)

        with self.assertRaisesRegex(
            AssertionError,
            r"scale should be of kind \(min, max\) and in range \[0, 1\]",
        ):
            transform = transforms.RandomErasing(scale=(10, 0.4))

        with self.assertRaisesRegex(
            AssertionError, r"ratio should be of kind \(min, max\)"
        ):
            transform = transforms.RandomErasing(ratio=(3.3, 0.3))

        with self.assertRaisesRegex(
            AssertionError, r"The probability should be in range \[0, 1\]"
        ):
            transform = transforms.RandomErasing(prob=1.5)

        with self.assertRaisesRegex(
            ValueError, r"value must be 'random' when type is str"
        ):
            transform = transforms.RandomErasing(value="0")

    def test_info(self):
        str(transforms.Compose([transforms.Resize((224, 224))]))
        str(transforms.Compose([transforms.Resize((224, 224))]))


class TestTransformsPIL(TestTransformsCV2):
    def get_backend(self):
        return 'pil'


class TestTransformsTensor(TestTransformsCV2):
    def get_backend(self):
        return 'tensor'

    def create_image(self, shape):
        return paddle.to_tensor(np.random.rand(*shape)).transpose(
            (2, 0, 1)
        )  # hwc->chw

    def do_transform(self, trans):
        trans.transforms.insert(0, transforms.ToTensor(data_format='CHW'))
        trans.transforms.append(transforms.Transpose(order=(1, 2, 0)))
        dataset_folder = DatasetFolder(self.data_dir, transform=trans)
        for _ in dataset_folder:
            pass

    def test_trans_all(self):
        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.120, 57.375],
        )
        trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                normalize,
            ]
        )
        self.do_transform(trans)

    def test_grayscale(self):
        trans = transforms.Compose([transforms.Grayscale()])
        self.do_transform(trans)

        trans_gray = transforms.Grayscale()
        fake_img = self.create_image((500, 400, 3))
        fake_img_gray = trans_gray(fake_img)

        np.testing.assert_equal(self.get_shape(fake_img_gray)[1], 500)
        np.testing.assert_equal(self.get_shape(fake_img_gray)[2], 400)

        trans_gray3 = transforms.Grayscale(3)
        fake_img = self.create_image((500, 400, 3))
        fake_img_gray = trans_gray3(fake_img)

    def test_normalize(self):
        normalize = transforms.Normalize(mean=0.5, std=0.5)
        trans = transforms.Compose([normalize])
        self.do_transform(trans)

    def test_color_jitter(self):
        trans = transforms.Compose([transforms.ColorJitter(1.1, 2.2, 0.8, 0.1)])
        self.do_transform(trans)

        color_jitter_trans = transforms.ColorJitter(1.2, 0.2, 0.5, 0.2)
        batch_input = paddle.rand((2, 3, 4, 4), dtype=paddle.float32)
        result = color_jitter_trans(batch_input)

    def test_perspective(self):
        trans = transforms.RandomPerspective(prob=1.0, distortion_scale=0.7)
        batch_input = paddle.rand((2, 3, 4, 4), dtype=paddle.float32)
        result = trans(batch_input)

    def test_affine(self):
        trans = transforms.RandomAffine(15, translate=[0.1, 0.1])
        batch_input = paddle.rand((2, 3, 4, 4), dtype=paddle.float32)
        result = trans(batch_input)

    def test_pad(self):
        trans = transforms.Compose([transforms.Pad(2)])
        self.do_transform(trans)

        fake_img = self.create_image((200, 150, 3))
        trans_pad = transforms.Compose([transforms.Pad(10)])
        fake_img_padded = trans_pad(fake_img)
        np.testing.assert_equal(self.get_shape(fake_img_padded), (3, 220, 170))
        trans_pad1 = transforms.Pad([1, 2])
        trans_pad2 = transforms.Pad([1, 2, 3, 4])
        trans_pad4 = transforms.Pad(1, padding_mode='edge')
        img = trans_pad1(fake_img)
        img = trans_pad2(img)
        img = trans_pad4(img)

    def test_random_crop(self):
        trans = transforms.Compose(
            [
                transforms.RandomCrop(200),
                transforms.RandomCrop((140, 160)),
            ]
        )
        self.do_transform(trans)

        trans_random_crop1 = transforms.RandomCrop(224)
        trans_random_crop2 = transforms.RandomCrop((140, 160))

        fake_img = self.create_image((500, 400, 3))
        fake_img_crop1 = trans_random_crop1(fake_img)
        fake_img_crop2 = trans_random_crop2(fake_img_crop1)

        np.testing.assert_equal(self.get_shape(fake_img_crop1), (3, 224, 224))

        np.testing.assert_equal(self.get_shape(fake_img_crop2), (3, 140, 160))

        trans_random_crop_same = transforms.RandomCrop((140, 160))
        img = trans_random_crop_same(fake_img_crop2)

        trans_random_crop_bigger = transforms.RandomCrop(
            (180, 200), pad_if_needed=True
        )
        img = trans_random_crop_bigger(img)

        trans_random_crop_pad = transforms.RandomCrop((224, 256), 2, True)
        img = trans_random_crop_pad(img)

    def test_erase(self):
        trans = transforms.Compose(
            [
                transforms.RandomErasing(value=(0.5,)),
                transforms.RandomErasing(value="random"),
            ]
        )
        self.do_transform(trans)

        erase_trans = transforms.RandomErasing(value=(0.5, 0.2, 0.01))
        batch_input = paddle.rand((2, 3, 4, 4), dtype=paddle.float32)
        result = erase_trans(batch_input)

    def test_exception(self):
        trans = transforms.Compose([transforms.Resize(-1)])

        trans_batch = transforms.Compose([transforms.Resize(-1)])

        with self.assertRaises((cv2.error, AssertionError, ValueError)):
            self.do_transform(trans)

        with self.assertRaises((cv2.error, AssertionError, ValueError)):
            self.do_transform(trans_batch)

        with self.assertRaises(ValueError):
            transforms.Pad([1.0, 2.0, 3.0])

        with self.assertRaises(TypeError):
            fake_img = self.create_image((100, 120, 3))
            F.pad(fake_img, '1')

        with self.assertRaises(TypeError):
            fake_img = self.create_image((100, 120, 3))
            F.pad(fake_img, 1, {})

        with self.assertRaises(TypeError):
            fake_img = self.create_image((100, 120, 3))
            F.pad(fake_img, 1, padding_mode=-1)

        with self.assertRaises(ValueError):
            fake_img = self.create_image((100, 120, 3))
            F.pad(fake_img, [1.0, 2.0, 3.0])

        with self.assertRaises(TypeError):
            tensor_img = paddle.rand((3, 100, 100))
            F.pad(tensor_img, '1')

        with self.assertRaises(TypeError):
            tensor_img = paddle.rand((3, 100, 100))
            F.pad(tensor_img, 1, {})

        with self.assertRaises(TypeError):
            tensor_img = paddle.rand((3, 100, 100))
            F.pad(tensor_img, 1, padding_mode=-1)

        with self.assertRaises(ValueError):
            tensor_img = paddle.rand((3, 100, 100))
            F.pad(tensor_img, [1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            transforms.RandomAffine(-10)

        with self.assertRaises(ValueError):
            transforms.RandomAffine([-30, 60], translate=[2, 2])

        with self.assertRaises(ValueError):
            transforms.RandomAffine(10, translate=[0.2, 0.2], scale=[-2, -1]),

        with self.assertRaises(ValueError):
            transforms.RandomAffine(10, translate=[0.2, 0.2], scale=[1, 2, 3]),

        with self.assertRaises(ValueError):
            transforms.RandomAffine(
                10, translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[1, 2, 3]
            ),

        with self.assertRaises(ValueError):
            transforms.RandomAffine(
                10,
                translate=[0.5, 0.3],
                scale=[0.7, 1.3],
                shear=[-10, 10, 0, 20, 40],
            )

        with self.assertRaises(ValueError):
            transforms.RandomRotation(-2)

        with self.assertRaises(ValueError):
            transforms.RandomRotation([1, 2, 3])

        with self.assertRaises(ValueError):
            trans_gray = transforms.Grayscale(5)
            fake_img = self.create_image((100, 120, 3))
            trans_gray(fake_img)

        with self.assertRaises(TypeError):
            transform = transforms.RandomResizedCrop(64)
            transform(1)

    test_color_jitter = None  # noqa: F811


class TestFunctional(unittest.TestCase):
    def test_errors(self):
        with self.assertRaises(TypeError):
            F.to_tensor(1)

        with self.assertRaises(ValueError):
            fake_img = Image.fromarray(
                (np.random.rand(28, 28, 3) * 255).astype('uint8')
            )
            F.to_tensor(fake_img, data_format=1)

        with self.assertRaises(ValueError):
            fake_img = paddle.rand((3, 100, 100))
            F.pad(fake_img, 1, padding_mode='symmetric')

        with self.assertRaises(TypeError):
            fake_img = paddle.rand((3, 100, 100))
            F.resize(fake_img, {1: 1})

        with self.assertRaises(TypeError):
            fake_img = Image.fromarray(
                (np.random.rand(28, 28, 3) * 255).astype('uint8')
            )
            F.resize(fake_img, '1')

        with self.assertRaises(TypeError):
            F.resize(1, 1)

        with self.assertRaises(TypeError):
            F.pad(1, 1)

        with self.assertRaises(TypeError):
            F.crop(1, 1, 1, 1, 1)

        with self.assertRaises(TypeError):
            F.hflip(1)

        with self.assertRaises(TypeError):
            F.vflip(1)

        with self.assertRaises(TypeError):
            F.adjust_brightness(1, 0.1)

        with self.assertRaises(TypeError):
            F.adjust_contrast(1, 0.1)

        with self.assertRaises(TypeError):
            F.adjust_hue(1, 0.1)

        with self.assertRaises(TypeError):
            F.adjust_saturation(1, 0.1)

        with self.assertRaises(TypeError):
            F.affine('45')

        with self.assertRaises(TypeError):
            F.affine(45, translate=0.3)

        with self.assertRaises(TypeError):
            F.affine(45, translate=[0.2, 0.2, 0.3])

        with self.assertRaises(TypeError):
            F.affine(45, translate=[0.2, 0.2], scale=-0.5)

        with self.assertRaises(TypeError):
            F.affine(45, translate=[0.2, 0.2], scale=0.5, shear=10)

        with self.assertRaises(TypeError):
            F.affine(45, translate=[0.2, 0.2], scale=0.5, shear=[-10, 0, 10])

        with self.assertRaises(TypeError):
            F.affine(
                45,
                translate=[0.2, 0.2],
                scale=0.5,
                shear=[-10, 10],
                interpolation=2,
            )

        with self.assertRaises(TypeError):
            F.affine(
                45, translate=[0.2, 0.2], scale=0.5, shear=[-10, 10], center=0
            )

        with self.assertRaises(TypeError):
            F.rotate(1, 0.1)

        with self.assertRaises(TypeError):
            F.to_grayscale(1)

        with self.assertRaises(ValueError):
            set_image_backend(1)

        with self.assertRaises(ValueError):
            image_load('tmp.jpg', backend=1)

    def test_normalize(self):
        np_img = (np.random.rand(28, 24, 3) * 255).astype('uint8')
        pil_img = Image.fromarray(np_img)
        tensor_img = F.to_tensor(pil_img)
        tensor_img_hwc = F.to_tensor(pil_img, data_format='HWC') * 255

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        normalized_img = F.normalize(tensor_img, mean, std)
        normalized_img_tensor = F.normalize(
            tensor_img_hwc, mean, std, data_format='HWC'
        )

        normalized_img_pil = F.normalize(pil_img, mean, std, data_format='HWC')
        normalized_img_np = F.normalize(
            np_img, mean, std, data_format='HWC', to_rgb=False
        )

        np.testing.assert_almost_equal(
            np.array(normalized_img_pil), normalized_img_np
        )
        np.testing.assert_almost_equal(
            normalized_img_tensor.numpy(), normalized_img_np, decimal=4
        )

    def test_center_crop(self):
        np_img = (np.random.rand(28, 24, 3) * 255).astype('uint8')
        pil_img = Image.fromarray(np_img)
        tensor_img = F.to_tensor(pil_img, data_format='CHW') * 255

        np_cropped_img = F.center_crop(np_img, 4)
        pil_cropped_img = F.center_crop(pil_img, 4)
        tensor_cropped_img = F.center_crop(tensor_img, 4)

        np.testing.assert_almost_equal(
            np_cropped_img, np.array(pil_cropped_img)
        )
        np.testing.assert_almost_equal(
            np_cropped_img,
            tensor_cropped_img.numpy().transpose((1, 2, 0)),
            decimal=4,
        )

    def test_color_jitter_sub_function(self):
        np.random.seed(555)
        np_img = (np.random.rand(28, 28, 3) * 255).astype('uint8')
        pil_img = Image.fromarray(np_img)
        tensor_img = F.to_tensor(np_img)
        np_img = pil_img

        np_img_gray = (np.random.rand(28, 28, 1) * 255).astype('uint8')
        tensor_img_gray = F.to_tensor(np_img_gray)

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.device.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if paddle.device.is_compiled_with_cuda():
            places.append('gpu')

        def test_adjust_brightness(np_img, tensor_img):
            result_cv2 = np.array(F.adjust_brightness(np_img, 1.2))
            result_tensor = F.adjust_brightness(tensor_img, 1.2).numpy()
            result_tensor = np.transpose(result_tensor * 255, (1, 2, 0)).astype(
                'uint8'
            )
            np.testing.assert_equal(result_cv2, result_tensor)

        # For adjust_contrast / adjust_saturation / adjust_hue the implement is kind
        # of different between PIL and Tensor. So the results can not equal exactly.

        def test_adjust_contrast(np_img, tensor_img):
            result_pil = np.array(F.adjust_contrast(np_img, 0.36))
            result_tensor = F.adjust_contrast(tensor_img, 0.36).numpy()
            result_tensor = np.transpose(result_tensor * 255, (1, 2, 0))
            diff = np.max(np.abs(result_tensor - result_pil))
            self.assertTrue(diff < 1.1)

        def test_adjust_saturation(np_img, tensor_img):
            result_pil = np.array(F.adjust_saturation(np_img, 1.0))
            result_tensor = F.adjust_saturation(tensor_img, 1.0).numpy()
            result_tensor = np.transpose(result_tensor * 255.0, (1, 2, 0))
            diff = np.max(np.abs(result_tensor - result_pil))
            self.assertTrue(diff < 1.1)

        def test_adjust_hue(np_img, tensor_img):
            result_pil = np.array(F.adjust_hue(np_img, 0.45))
            result_tensor = F.adjust_hue(tensor_img, 0.45).numpy()
            result_tensor = np.transpose(result_tensor * 255, (1, 2, 0))
            diff = np.max(np.abs(result_tensor - result_pil))
            self.assertTrue(diff <= 16.0)

        for place in places:
            paddle.set_device(place)

            test_adjust_brightness(np_img, tensor_img)
            test_adjust_contrast(np_img, tensor_img)
            test_adjust_saturation(np_img, tensor_img)
            test_adjust_hue(np_img, tensor_img)

    def test_pad(self):
        np_img = (np.random.rand(28, 24, 3) * 255).astype('uint8')
        pil_img = Image.fromarray(np_img)
        tensor_img = F.to_tensor(pil_img, 'CHW') * 255

        np_padded_img = F.pad(np_img, [1, 2], padding_mode='reflect')
        pil_padded_img = F.pad(pil_img, [1, 2], padding_mode='reflect')
        tensor_padded_img = F.pad(tensor_img, [1, 2], padding_mode='reflect')

        np.testing.assert_almost_equal(np_padded_img, np.array(pil_padded_img))
        np.testing.assert_almost_equal(
            np_padded_img,
            tensor_padded_img.numpy().transpose((1, 2, 0)),
            decimal=3,
        )

        tensor_padded_img = F.pad(tensor_img, 1, padding_mode='reflect')
        tensor_padded_img = F.pad(
            tensor_img, [1, 2, 1, 2], padding_mode='reflect'
        )

        pil_p_img = pil_img.convert('P')
        pil_padded_img = F.pad(pil_p_img, [1, 2])
        pil_padded_img = F.pad(pil_p_img, [1, 2], padding_mode='reflect')

    def test_resize(self):
        np_img = (np.zeros([28, 24, 3]) * 255).astype('uint8')
        pil_img = Image.fromarray(np_img)
        tensor_img = F.to_tensor(pil_img, 'CHW') * 255

        np_resized_img = F.resize(np_img, 40)
        pil_resized_img = F.resize(pil_img, 40)
        tensor_resized_img = F.resize(tensor_img, 40)
        tensor_resized_img2 = F.resize(tensor_img, (46, 40))

        np.testing.assert_almost_equal(
            np_resized_img, np.array(pil_resized_img)
        )
        np.testing.assert_almost_equal(
            np_resized_img,
            tensor_resized_img.numpy().transpose((1, 2, 0)),
            decimal=3,
        )
        np.testing.assert_almost_equal(
            np_resized_img,
            tensor_resized_img2.numpy().transpose((1, 2, 0)),
            decimal=3,
        )

        gray_img = (np.zeros([28, 32])).astype('uint8')
        gray_resize_img = F.resize(gray_img, 40)

    def test_to_tensor(self):
        np_img = (np.random.rand(28, 28) * 255).astype('uint8')
        pil_img = Image.fromarray(np_img)

        np_tensor = F.to_tensor(np_img, data_format='HWC')
        pil_tensor = F.to_tensor(pil_img, data_format='HWC')

        np.testing.assert_allclose(np_tensor.numpy(), pil_tensor.numpy())

        # test float dtype
        float_img = np.random.rand(28, 28)
        float_tensor = F.to_tensor(float_img)

        pil_img = Image.fromarray(np_img).convert('I')
        pil_tensor = F.to_tensor(pil_img)

        pil_img_16bit = Image.new('I;16', pil_img.size)
        pil_img_16bit.paste(pil_img)
        pil_tensor = F.to_tensor(pil_img)

        pil_img = Image.fromarray(np_img).convert('F')
        pil_tensor = F.to_tensor(pil_img)

        pil_img = Image.fromarray(np_img).convert('L')
        pil_tensor = F.to_tensor(pil_img)

        pil_img = Image.fromarray(np_img).convert('YCbCr')
        pil_tensor = F.to_tensor(pil_img)

    def test_erase(self):
        np_img = (np.random.rand(28, 28, 3) * 255).astype('uint8')
        pil_img = Image.fromarray(np_img).convert('RGB')

        expected = np_img.copy()
        expected[10:15, 10:15, :] = 0

        F.erase(np_img, 10, 10, 5, 5, 0, inplace=True)
        np.testing.assert_equal(np_img, expected)

        pil_result = F.erase(pil_img, 10, 10, 5, 5, 0)
        np.testing.assert_equal(np.array(pil_result), expected)

        np_data = np.random.rand(3, 28, 28).astype('float32')
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.device.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if paddle.device.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            tensor_img = paddle.to_tensor(np_data)
            expected_tensor = tensor_img.clone()
            expected_tensor[:, 10:15, 10:15] = paddle.to_tensor([0.88])

            tensor_result = F.erase(
                tensor_img, 10, 10, 5, 5, paddle.to_tensor([0.88])
            )
            np.testing.assert_equal(
                tensor_result.numpy(), expected_tensor.numpy()
            )

    def test_erase_backward(self):
        img = paddle.randn((3, 14, 14), dtype=np.float32)
        img.stop_gradient = False
        erased = F.erase(
            img, 3, 3, 5, 5, paddle.ones((1, 1, 1), dtype='float32')
        )
        loss = erased.sum()
        loss.backward()

        expected_grad = np.ones((3, 14, 14), dtype=np.float32)
        expected_grad[:, 3:8, 3:8] = 0.0
        np.testing.assert_equal(img.grad.numpy(), expected_grad)

    def test_image_load(self):
        fake_img = Image.fromarray(
            (np.random.random((32, 32, 3)) * 255).astype('uint8')
        )

        temp_dir = tempfile.TemporaryDirectory()
        path = os.path.join(temp_dir.name, 'temp.jpg')
        fake_img.save(path)

        set_image_backend('pil')

        pil_img = image_load(path).convert('RGB')

        print(type(pil_img))

        set_image_backend('cv2')

        np_img = image_load(path)

        temp_dir.cleanup()

    def test_affine(self):
        np_img = (np.random.rand(32, 26, 3) * 255).astype('uint8')
        pil_img = Image.fromarray(np_img).convert('RGB')
        tensor_img = F.to_tensor(pil_img, data_format='CHW') * 255

        np.testing.assert_almost_equal(
            np_img, tensor_img.transpose((1, 2, 0)), decimal=4
        )

        np_affined_img = F.affine(
            np_img, 45, translate=[0.2, 0.2], scale=0.5, shear=[-10, 10]
        )
        pil_affined_img = F.affine(
            pil_img, 45, translate=[0.2, 0.2], scale=0.5, shear=[-10, 10]
        )
        tensor_affined_img = F.affine(
            tensor_img, 45, translate=[0.2, 0.2], scale=0.5, shear=[-10, 10]
        )

        np.testing.assert_equal(
            np_affined_img.shape, np.array(pil_affined_img).shape
        )
        np.testing.assert_equal(
            np_affined_img.shape, tensor_affined_img.transpose((1, 2, 0)).shape
        )

        # Temporarily disable the test on Windows with numpy >= 2.0.0 to avoid
        # precision issue on PR-CI-Windows-Inference
        if os.name == "nt" and np.lib.NumpyVersion(np.__version__) >= "2.0.0":
            return

        np.testing.assert_almost_equal(
            np.array(pil_affined_img),
            tensor_affined_img.numpy().transpose((1, 2, 0)),
            decimal=4,
        )

    def test_rotate(self):
        np_img = (np.random.rand(28, 28, 3) * 255).astype('uint8')
        pil_img = Image.fromarray(np_img).convert('RGB')
        rotated_np_img = F.rotate(np_img, 80, expand=True)
        rotated_pil_img = F.rotate(pil_img, 80, expand=True)

        tensor_img = F.to_tensor(pil_img, 'CHW')

        rotated_tensor_img1 = F.rotate(tensor_img, 80, expand=True)

        rotated_tensor_img2 = F.rotate(
            tensor_img,
            80,
            interpolation='bilinear',
            center=(10, 10),
            expand=False,
        )

        np.testing.assert_equal(
            rotated_np_img.shape, np.array(rotated_pil_img).shape
        )
        np.testing.assert_equal(
            rotated_np_img.shape, rotated_tensor_img1.transpose((1, 2, 0)).shape
        )

    def test_rotate1(self):
        np_img = (np.random.rand(28, 28, 3) * 255).astype('uint8')
        pil_img = Image.fromarray(np_img).convert('RGB')

        rotated_np_img = F.rotate(
            np_img, 80, expand=True, center=[0, 0], fill=[0, 0, 0]
        )
        rotated_pil_img = F.rotate(
            pil_img, 80, expand=True, center=[0, 0], fill=[0, 0, 0]
        )

        np.testing.assert_equal(
            rotated_np_img.shape, np.array(rotated_pil_img).shape
        )

    def test_perspective(self):
        np_img = (np.random.rand(32, 26, 3) * 255).astype('uint8')
        pil_img = Image.fromarray(np_img).convert('RGB')
        tensor_img = F.to_tensor(pil_img, data_format='CHW') * 255

        np.testing.assert_almost_equal(
            np_img, tensor_img.transpose((1, 2, 0)), decimal=4
        )

        startpoints = [[0, 0], [13, 0], [13, 15], [0, 15]]
        endpoints = [[3, 2], [12, 3], [10, 14], [2, 15]]

        np_perspectived_img = F.perspective(np_img, startpoints, endpoints)
        pil_perspectived_img = F.perspective(pil_img, startpoints, endpoints)
        tensor_perspectived_img = F.perspective(
            tensor_img, startpoints, endpoints
        )

        np.testing.assert_equal(
            np_perspectived_img.shape, np.array(pil_perspectived_img).shape
        )
        np.testing.assert_equal(
            np_perspectived_img.shape,
            tensor_perspectived_img.transpose((1, 2, 0)).shape,
        )

        result_pil = np.array(pil_perspectived_img)
        result_tensor = (
            tensor_perspectived_img.numpy().transpose((1, 2, 0)).astype('uint8')
        )
        num_diff_pixels = (result_pil != result_tensor).sum() / 3.0
        ratio_diff_pixels = (
            num_diff_pixels / result_tensor.shape[0] / result_tensor.shape[1]
        )
        # Tolerance : less than 6% of different pixels
        assert ratio_diff_pixels < 0.06

    def test_batch_input(self):
        paddle.seed(777)
        batch_tensor = paddle.rand((2, 3, 8, 8), dtype=paddle.float32)

        def test_erase(batch_tensor):
            input1, input2 = paddle.unbind(batch_tensor, axis=0)
            target_result = paddle.stack(
                [
                    F.erase(input1, 1, 1, 2, 2, 0.5),
                    F.erase(input2, 1, 1, 2, 2, 0.5),
                ]
            )

            batch_result = F.erase(batch_tensor, 1, 1, 2, 2, 0.5)

            return paddle.allclose(batch_result, target_result)

        self.assertTrue(test_erase(batch_tensor))

        def test_affine(batch_tensor):
            input1, input2 = paddle.unbind(batch_tensor, axis=0)
            target_result = paddle.stack(
                [
                    F.affine(
                        input1,
                        45,
                        translate=[0.2, 0.2],
                        scale=0.5,
                        shear=[-10, 10],
                    ),
                    F.affine(
                        input2,
                        45,
                        translate=[0.2, 0.2],
                        scale=0.5,
                        shear=[-10, 10],
                    ),
                ]
            )
            batch_result = F.affine(
                batch_tensor,
                45,
                translate=[0.2, 0.2],
                scale=0.5,
                shear=[-10, 10],
            )

            return paddle.allclose(batch_result, target_result)

        self.assertTrue(test_affine(batch_tensor))

        def test_perspective(batch_tensor):
            input1, input2 = paddle.unbind(batch_tensor, axis=0)
            startpoints = [[0, 0], [3, 0], [4, 5], [6, 7]]
            endpoints = [[0, 1], [3, 1], [4, 4], [5, 7]]
            target_result = paddle.stack(
                [
                    F.perspective(input1, startpoints, endpoints),
                    F.perspective(input2, startpoints, endpoints),
                ]
            )

            batch_result = F.perspective(batch_tensor, startpoints, endpoints)

            return paddle.allclose(batch_result, target_result)

        self.assertTrue(test_perspective(batch_tensor))

        def test_adjust_brightness(batch_tensor):
            input1, input2 = paddle.unbind(batch_tensor, axis=0)
            target_result = paddle.stack(
                [
                    F.adjust_brightness(input1, 2.1),
                    F.adjust_brightness(input2, 2.1),
                ]
            )

            batch_result = F.adjust_brightness(batch_tensor, 2.1)

            return paddle.allclose(batch_result, target_result)

        self.assertTrue(test_adjust_brightness(batch_tensor))

        def test_adjust_contrast(batch_tensor):
            input1, input2 = paddle.unbind(batch_tensor, axis=0)
            target_result = paddle.stack(
                [F.adjust_contrast(input1, 0.3), F.adjust_contrast(input2, 0.3)]
            )

            batch_result = F.adjust_contrast(batch_tensor, 0.3)

            return paddle.allclose(batch_result, target_result)

        self.assertTrue(test_adjust_contrast(batch_tensor))

        def test_adjust_saturation(batch_tensor):
            input1, input2 = paddle.unbind(batch_tensor, axis=0)
            target_result = paddle.stack(
                [
                    F.adjust_saturation(input1, 1.1),
                    F.adjust_saturation(input2, 1.1),
                ]
            )

            batch_result = F.adjust_saturation(batch_tensor, 1.1)

            return paddle.allclose(batch_result, target_result)

        self.assertTrue(test_adjust_saturation(batch_tensor))

        def test_adjust_hue(batch_tensor):
            input1, input2 = paddle.unbind(batch_tensor, axis=0)
            target_result = paddle.stack(
                [F.adjust_hue(input1, -0.2), F.adjust_hue(input2, -0.2)]
            )

            batch_result = F.adjust_hue(batch_tensor, -0.2)

            return paddle.allclose(batch_result, target_result)

        self.assertTrue(test_adjust_hue(batch_tensor))


if __name__ == '__main__':
    unittest.main()
