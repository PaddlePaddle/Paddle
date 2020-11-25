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

import unittest
import os
import tempfile
import cv2
import shutil
import numpy as np
from PIL import Image

import paddle
from paddle.vision import get_image_backend, set_image_backend, image_load
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms
import paddle.vision.transforms.functional as F


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
                    fake_img = (np.random.random(
                        (280, 350, 3)) * 255).astype('uint8')
                else:
                    fake_img = (np.random.random(
                        (400, 300, 3)) * 255).astype('uint8')
                cv2.imwrite(os.path.join(sub_dir, str(j) + '.jpg'), fake_img)

    def get_backend(self):
        return 'cv2'

    def create_image(self, shape):
        if self.backend == 'cv2':
            return (np.random.rand(*shape) * 255).astype('uint8')
        elif self.backend == 'pil':
            return Image.fromarray((np.random.rand(*shape) * 255).astype(
                'uint8'))

    def get_shape(self, img):
        if self.backend == 'pil':
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
            std=[58.395, 57.120, 57.375], )
        trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.Transpose(),
            normalize,
        ])

        self.do_transform(trans)

    def test_normalize(self):
        normalize = transforms.Normalize(mean=0.5, std=0.5)
        trans = transforms.Compose([transforms.Transpose(), normalize])
        self.do_transform(trans)

    def test_trans_resize(self):
        trans = transforms.Compose([
            transforms.Resize(300),
            transforms.RandomResizedCrop((280, 280)),
            transforms.Resize(280),
            transforms.Resize((256, 200)),
            transforms.Resize((180, 160)),
            transforms.CenterCrop(128),
            transforms.CenterCrop((128, 128)),
        ])
        self.do_transform(trans)

    def test_flip(self):
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(1.0),
            transforms.RandomHorizontalFlip(0.0),
            transforms.RandomVerticalFlip(0.0),
            transforms.RandomVerticalFlip(1.0),
        ])
        self.do_transform(trans)

    def test_color_jitter(self):
        trans = transforms.Compose([
            transforms.BrightnessTransform(0.0),
            transforms.HueTransform(0.0),
            transforms.SaturationTransform(0.0),
            transforms.ContrastTransform(0.0),
        ])
        self.do_transform(trans)

    def test_rotate(self):
        trans = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.RandomRotation([-10, 10]),
            transforms.RandomRotation(
                45, expand=True),
            transforms.RandomRotation(
                10, expand=True, center=(60, 80)),
        ])
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
        trans = transforms.Compose([
            transforms.RandomCrop(200),
            transforms.RandomCrop((140, 160)),
        ])
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
            (180, 200), pad_if_needed=True)
        img = trans_random_crop_bigger(img)

        trans_random_crop_pad = transforms.RandomCrop((224, 256), 2, True)
        img = trans_random_crop_pad(img)

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

    def test_tranpose(self):
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
        trans_pad = transforms.Pad(10, keys=("image", ))
        fake_img_padded = trans_pad((fake_img1, fake_img2))

    def test_exception(self):
        trans = transforms.Compose([transforms.Resize(-1)])

        trans_batch = transforms.Compose([transforms.Resize(-1)])

        with self.assertRaises(Exception):
            self.do_transform(trans)

        with self.assertRaises(Exception):
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

    def test_info(self):
        str(transforms.Compose([transforms.Resize((224, 224))]))
        str(transforms.Compose([transforms.Resize((224, 224))]))


class TestTransformsPIL(TestTransformsCV2):
    def get_backend(self):
        return 'pil'


class TestFunctional(unittest.TestCase):
    def test_errors(self):
        with self.assertRaises(TypeError):
            F.to_tensor(1)

        with self.assertRaises(ValueError):
            fake_img = Image.fromarray((np.random.rand(28, 28, 3) * 255).astype(
                'uint8'))
            F.to_tensor(fake_img, data_format=1)

        with self.assertRaises(TypeError):
            fake_img = Image.fromarray((np.random.rand(28, 28, 3) * 255).astype(
                'uint8'))
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
            F.rotate(1, 0.1)

        with self.assertRaises(TypeError):
            F.to_grayscale(1)

        with self.assertRaises(ValueError):
            set_image_backend(1)

        with self.assertRaises(ValueError):
            image_load('tmp.jpg', backend=1)

    def test_normalize(self):
        np_img = (np.random.rand(28, 24, 3)).astype('uint8')
        pil_img = Image.fromarray(np_img)
        tensor_img = F.to_tensor(pil_img)
        tensor_img_hwc = F.to_tensor(pil_img, data_format='HWC')

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        normalized_img = F.normalize(tensor_img, mean, std)
        normalized_img = F.normalize(
            tensor_img_hwc, mean, std, data_format='HWC')

        normalized_img = F.normalize(pil_img, mean, std, data_format='HWC')
        normalized_img = F.normalize(
            np_img, mean, std, data_format='HWC', to_rgb=True)

    def test_center_crop(self):
        np_img = (np.random.rand(28, 24, 3)).astype('uint8')
        pil_img = Image.fromarray(np_img)

        np_cropped_img = F.center_crop(np_img, 4)
        pil_cropped_img = F.center_crop(pil_img, 4)

        np.testing.assert_almost_equal(np_cropped_img,
                                       np.array(pil_cropped_img))

    def test_pad(self):
        np_img = (np.random.rand(28, 24, 3)).astype('uint8')
        pil_img = Image.fromarray(np_img)

        np_padded_img = F.pad(np_img, [1, 2], padding_mode='reflect')
        pil_padded_img = F.pad(pil_img, [1, 2], padding_mode='reflect')

        np.testing.assert_almost_equal(np_padded_img, np.array(pil_padded_img))

        pil_p_img = pil_img.convert('P')
        pil_padded_img = F.pad(pil_p_img, [1, 2])
        pil_padded_img = F.pad(pil_p_img, [1, 2], padding_mode='reflect')

    def test_resize(self):
        np_img = (np.zeros([28, 24, 3])).astype('uint8')
        pil_img = Image.fromarray(np_img)

        np_reseized_img = F.resize(np_img, 40)
        pil_reseized_img = F.resize(pil_img, 40)

        np.testing.assert_almost_equal(np_reseized_img,
                                       np.array(pil_reseized_img))

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

        pil_img = Image.fromarray(np_img).convert('I;16')
        pil_tensor = F.to_tensor(pil_img)

        pil_img = Image.fromarray(np_img).convert('F')
        pil_tensor = F.to_tensor(pil_img)

        pil_img = Image.fromarray(np_img).convert('1')
        pil_tensor = F.to_tensor(pil_img)

        pil_img = Image.fromarray(np_img).convert('YCbCr')
        pil_tensor = F.to_tensor(pil_img)

    def test_image_load(self):
        fake_img = Image.fromarray((np.random.random((32, 32, 3)) * 255).astype(
            'uint8'))

        path = 'temp.jpg'
        fake_img.save(path)

        set_image_backend('pil')

        pil_img = image_load(path).convert('RGB')

        print(type(pil_img))

        set_image_backend('cv2')

        np_img = image_load(path)

        os.remove(path)


if __name__ == '__main__':
    unittest.main()
