# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import site
import unittest
import numpy as np


class TestCustomCPUPlugin(unittest.TestCase):

    def setUp(self):
        # compile so and set to current path
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = 'rm -rf PaddleCustomDevice && git clone https://github.com/PaddlePaddle/PaddleCustomDevice.git && cd PaddleCustomDevice/backends/custom_cpu && mkdir build && cd build && cmake .. && make -j8'
        os.system(cmd)

        # set environment for loading and registering compiled custom kernels
        # only valid in current process
        os.environ['CUSTOM_DEVICE_ROOT'] = os.path.join(
            cur_dir, 'PaddleCustomDevice/backends/custom_cpu/build')

    def test_custom_device_dataloader(self):
        import paddle

        with paddle.fluid.framework._test_eager_guard():
            self._test_custom_device_dataloader()
        self._test_custom_device_dataloader()

    def _test_custom_device_dataloader(self):
        import paddle

        paddle.set_device('custom_cpu')
        dataset = paddle.vision.datasets.MNIST(
            mode='test',
            transform=paddle.vision.transforms.Compose([
                paddle.vision.transforms.CenterCrop(20),
                paddle.vision.transforms.RandomResizedCrop(14),
                paddle.vision.transforms.Normalize(),
                paddle.vision.transforms.ToTensor()
            ]))
        loader = paddle.io.DataLoader(dataset,
                                      batch_size=32,
                                      num_workers=1,
                                      shuffle=True)
        for image, label in loader:
            self.assertTrue(image.place.is_custom_place())
            self.assertTrue(label.place.is_custom_place())
            break

    def test_custom_device_mnist(self):
        import paddle

        with paddle.fluid.framework._test_eager_guard():
            self._test_custom_device_mnist()
        self._test_custom_device_mnist()

    def _test_custom_device_mnist(self):
        import paddle

        class MNIST(paddle.nn.Layer):

            def __init__(self):
                super(MNIST, self).__init__()
                self.shape = 1 * 28 * 28
                self.size = 10
                self.output_weight = self.create_parameter(
                    [self.shape, self.size])
                self.accuracy = paddle.metric.Accuracy()

            def forward(self, inputs, label=None):
                x = paddle.reshape(inputs, shape=[-1, self.shape])
                x = paddle.matmul(x, self.output_weight)
                x = paddle.nn.functional.softmax(x)
                if label is not None:
                    self.accuracy.reset()
                    correct = self.accuracy.compute(x, label)
                    self.accuracy.update(correct)
                    acc = self.accuracy.accumulate()
                    return x, acc
                else:
                    return x

        paddle.set_device('custom_cpu')
        dataset = paddle.vision.datasets.MNIST(
            mode='train',
            transform=paddle.vision.transforms.Compose(
                [paddle.vision.transforms.ToTensor()]))
        loader = paddle.io.DataLoader(dataset,
                                      batch_size=64,
                                      num_workers=1,
                                      shuffle=True)

        mnist = MNIST()
        sgd = paddle.optimizer.SGD(learning_rate=0.01,
                                   parameters=mnist.parameters())

        data = next(loader())
        img = data[0]
        label = data[1]
        label_int32 = paddle.cast(label, 'int32')

        pred, acc = mnist(img, label_int32)
        avg_loss = paddle.nn.functional.cross_entropy(pred, label_int32)
        avg_loss.backward()
        sgd.step()
        sgd.clear_grad()

        self.assertTrue(pred.place.is_custom_place())

    def tearDown(self):
        del os.environ['CUSTOM_DEVICE_ROOT']


if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # only support Linux now
        exit()
    unittest.main()
