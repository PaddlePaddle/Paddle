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
图像模型组件测试 / Image Model Component Tests

测试目标 / Test Target:
  paddle.vision.models 预训练模型组件

覆盖的模块 / Covered Modules:
  - paddle.vision.models.resnet: ResNet架构
  - paddle.vision.models.mobilenetv1/v2: MobileNet
  - paddle.vision.models.vgg: VGG架构
  - paddle.vision.models.alexnet: AlexNet

作用 / Purpose:
  补充视觉模型API的测试，提升覆盖率。
"""

import unittest

import paddle
from paddle.vision import models

paddle.disable_static()


class TestResNet(unittest.TestCase):
    """测试ResNet模型 / Test ResNet models"""

    def test_resnet18_inference(self):
        """测试ResNet18推理 / Test ResNet18 inference"""
        model = models.resnet18(pretrained=False)
        model.eval()
        x = paddle.randn([2, 3, 224, 224])
        with paddle.no_grad():
            output = model(x)
        self.assertEqual(output.shape, [2, 1000])

    def test_resnet50_structure(self):
        """测试ResNet50结构 / Test ResNet50 structure"""
        model = models.resnet50(pretrained=False)
        self.assertIsNotNone(model.fc)
        # Test with small batch
        model.eval()
        x = paddle.randn([1, 3, 224, 224])
        with paddle.no_grad():
            output = model(x)
        self.assertEqual(output.shape[1], 1000)

    def test_resnet_custom_classes(self):
        """测试自定义分类数ResNet / Test ResNet with custom num_classes"""
        model = models.resnet18(pretrained=False, num_classes=10)
        model.eval()
        x = paddle.randn([2, 3, 224, 224])
        with paddle.no_grad():
            output = model(x)
        self.assertEqual(output.shape, [2, 10])


class TestMobileNet(unittest.TestCase):
    """测试MobileNet / Test MobileNet"""

    def test_mobilenetv1(self):
        """测试MobileNetV1推理 / Test MobileNetV1 inference"""
        model = models.mobilenet_v1(pretrained=False)
        model.eval()
        x = paddle.randn([2, 3, 224, 224])
        with paddle.no_grad():
            output = model(x)
        self.assertEqual(output.shape, [2, 1000])

    def test_mobilenetv2(self):
        """测试MobileNetV2推理 / Test MobileNetV2 inference"""
        model = models.mobilenet_v2(pretrained=False)
        model.eval()
        x = paddle.randn([2, 3, 224, 224])
        with paddle.no_grad():
            output = model(x)
        self.assertEqual(output.shape, [2, 1000])


class TestVGG(unittest.TestCase):
    """测试VGG / Test VGG"""

    def test_vgg16(self):
        """测试VGG16推理 / Test VGG16 inference"""
        model = models.vgg16(pretrained=False)
        model.eval()
        x = paddle.randn([1, 3, 224, 224])
        with paddle.no_grad():
            output = model(x)
        self.assertEqual(output.shape, [1, 1000])


class TestAlexNet(unittest.TestCase):
    """测试AlexNet / Test AlexNet"""

    def test_alexnet(self):
        """测试AlexNet推理 / Test AlexNet inference"""
        model = models.alexnet(pretrained=False)
        model.eval()
        x = paddle.randn([1, 3, 224, 224])
        with paddle.no_grad():
            output = model(x)
        self.assertEqual(output.shape, [1, 1000])


class TestSqueezeNet(unittest.TestCase):
    """测试SqueezeNet / Test SqueezeNet"""

    def test_squeezenet1_0(self):
        """测试SqueezeNet1.0 / Test SqueezeNet1.0"""
        model = models.squeezenet1_0(pretrained=False)
        model.eval()
        x = paddle.randn([1, 3, 224, 224])
        with paddle.no_grad():
            output = model(x)
        self.assertEqual(output.shape, [1, 1000])


if __name__ == '__main__':
    unittest.main()
