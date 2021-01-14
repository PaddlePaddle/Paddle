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
import numpy as np

import paddle
from paddle.static import InputSpec
import paddle.vision.models as models


class TestVisonModels(unittest.TestCase):
    def models_infer(self, arch, pretrained=False, batch_norm=False):

        x = np.array(np.random.random((2, 3, 224, 224)), dtype=np.float32)
        if batch_norm:
            net = models.__dict__[arch](pretrained=pretrained, batch_norm=True)
        else:
            net = models.__dict__[arch](pretrained=pretrained)

        input = InputSpec([None, 3, 224, 224], 'float32', 'image')
        model = paddle.Model(net, input)
        model.prepare()

        model.predict_batch(x)

    def test_mobilenetv2_pretrained(self):
        self.models_infer('mobilenet_v2', pretrained=False)

    def test_mobilenetv1(self):
        self.models_infer('mobilenet_v1')

    def test_vgg11(self):
        self.models_infer('vgg11')

    def test_vgg13(self):
        self.models_infer('vgg13')

    def test_vgg16(self):
        self.models_infer('vgg16')

    def test_vgg16_bn(self):
        self.models_infer('vgg16', batch_norm=True)

    def test_vgg19(self):
        self.models_infer('vgg19')

    def test_resnet18(self):
        self.models_infer('resnet18')

    def test_resnet34(self):
        self.models_infer('resnet34')

    def test_resnet50(self):
        self.models_infer('resnet50')

    def test_resnet101(self):
        self.models_infer('resnet101')

    def test_resnet152(self):
        self.models_infer('resnet152')

    def test_vgg16_num_classes(self):
        vgg16 = models.__dict__['vgg16'](pretrained=False, num_classes=10)

    def test_lenet(self):
        input = InputSpec([None, 1, 28, 28], 'float32', 'x')
        lenet = paddle.Model(models.__dict__['LeNet'](), input)
        lenet.prepare()

        x = np.array(np.random.random((2, 1, 28, 28)), dtype=np.float32)
        lenet.predict_batch(x)


if __name__ == '__main__':
    unittest.main()
