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
import tempfile
import unittest

import numpy as np


class TestCustomCPUPlugin(unittest.TestCase):
    def setUp(self):
        # compile so and set to current path
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_dir = tempfile.TemporaryDirectory()
        cmd = 'cd {} \
            && git clone --depth 1 {} \
            && cd PaddleCustomDevice \
            && git fetch origin \
            && git checkout {} -b dev \
            && cd backends/custom_cpu \
            && mkdir build && cd build && cmake .. -DPython_EXECUTABLE={} -DWITH_TESTING=OFF && make -j8'.format(
            self.temp_dir.name,
            os.getenv('PLUGIN_URL'),
            os.getenv('PLUGIN_TAG'),
            sys.executable,
        )
        os.system(cmd)

        # set environment for loading and registering compiled custom kernels
        # only valid in current process
        os.environ['CUSTOM_DEVICE_ROOT'] = os.path.join(
            cur_dir,
            f'{self.temp_dir.name}/PaddleCustomDevice/backends/custom_cpu/build',
        )

    def tearDown(self):
        self.temp_dir.cleanup()
        del os.environ['CUSTOM_DEVICE_ROOT']

    def test_custom_device(self):
        self._test_custom_device_dataloader()
        self._test_custom_device_mnist()
        self._test_eager_backward_api()
        self._test_eager_copy_to()
        self._test_fallback_kernel()
        self._test_scalar()
        self._test_custom_device_py_api()
        self._test_custom_device_mix_precision()

    def _test_custom_device_dataloader(self):
        import paddle

        paddle.set_device('custom_cpu')
        dataset = paddle.vision.datasets.MNIST(
            mode='test',
            transform=paddle.vision.transforms.Compose(
                [
                    paddle.vision.transforms.CenterCrop(20),
                    paddle.vision.transforms.RandomResizedCrop(14),
                    paddle.vision.transforms.Normalize(),
                    paddle.vision.transforms.ToTensor(),
                ]
            ),
        )
        loader = paddle.io.DataLoader(
            dataset, batch_size=32, num_workers=1, shuffle=True
        )
        for image, label in loader:
            self.assertTrue(image.place.is_custom_place())
            self.assertTrue(label.place.is_custom_place())
            break

    def _test_custom_device_mnist(self):
        import paddle

        class MNIST(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.shape = 1 * 28 * 28
                self.size = 10
                self.output_weight = self.create_parameter(
                    [self.shape, self.size]
                )
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
                [paddle.vision.transforms.ToTensor()]
            ),
        )
        loader = paddle.io.DataLoader(
            dataset, batch_size=64, num_workers=1, shuffle=True
        )

        mnist = MNIST()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=mnist.parameters()
        )

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

    def _test_eager_backward_api(self):
        x = np.random.random([2, 2]).astype("float32")
        y = np.random.random([2, 2]).astype("float32")
        grad = np.ones([2, 2]).astype("float32")

        import paddle

        paddle.set_device('custom_cpu')
        paddle.device.get_available_device()
        x_tensor = paddle.to_tensor(x, stop_gradient=False)
        y_tensor = paddle.to_tensor(y)
        z1_tensor = paddle.matmul(x_tensor, y_tensor)
        z2_tensor = paddle.matmul(x_tensor, y_tensor)

        grad_tensor = paddle.to_tensor(grad)
        paddle.autograd.backward([z1_tensor, z2_tensor], [grad_tensor, None])

        self.assertTrue(x_tensor.grad.place.is_custom_place())

    def _test_eager_copy_to(self):
        import paddle

        x = np.random.random([2, 2]).astype("float32")
        # cpu -> custom
        cpu_tensor = paddle.to_tensor(
            x, dtype='float32', place=paddle.CPUPlace()
        )
        custom_cpu_tensor = cpu_tensor._copy_to(
            paddle.CustomPlace('custom_cpu', 0), True
        )
        np.testing.assert_array_equal(custom_cpu_tensor, x)
        self.assertTrue(custom_cpu_tensor.place.is_custom_place())
        # custom -> custom
        another_custom_cpu_tensor = custom_cpu_tensor._copy_to(
            paddle.CustomPlace('custom_cpu', 0), True
        )
        np.testing.assert_array_equal(another_custom_cpu_tensor, x)
        self.assertTrue(another_custom_cpu_tensor.place.is_custom_place())
        # custom -> cpu
        another_cpu_tensor = custom_cpu_tensor._copy_to(paddle.CPUPlace(), True)
        np.testing.assert_array_equal(another_cpu_tensor, x)
        self.assertTrue(another_cpu_tensor.place.is_cpu_place())
        # custom -> custom self
        another_custom_cpu_tensor = another_custom_cpu_tensor._copy_to(
            paddle.CustomPlace('custom_cpu', 0), True
        )
        np.testing.assert_array_equal(another_custom_cpu_tensor, x)
        self.assertTrue(another_custom_cpu_tensor.place.is_custom_place())

    def _test_fallback_kernel(self):
        # using (custom_cpu, add, int16) which is not registered
        import paddle

        r = np.array([6, 6, 6], 'int16')
        x = paddle.to_tensor([5, 4, 3], 'int16')
        y = paddle.to_tensor([1, 2, 3], 'int16')
        z = paddle.add(x, y)
        np.testing.assert_array_equal(z, r)

    def _test_scalar(self):
        import paddle

        data_1 = paddle.to_tensor(
            [[[[1.0, 4.0, 5.0, 7.0], [3.0, 4.0, 5.0, 6.0]]]]
        )
        k_t = paddle.to_tensor([3], dtype="int32")
        value_1, indices_1 = paddle.topk(data_1, k=k_t)

    def _test_custom_device_gradient_accumulation(self):
        import paddle

        class MNIST(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.shape = 1 * 28 * 28
                self.size = 10
                self.output_weight = self.create_parameter(
                    [self.shape, self.size]
                )
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
                [paddle.vision.transforms.ToTensor()]
            ),
        )
        loader = paddle.io.DataLoader(
            dataset, batch_size=64, num_workers=1, shuffle=True
        )

        mnist = MNIST()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=mnist.parameters()
        )

        data = next(loader())
        img = data[0]
        label = data[1]
        label_int32 = paddle.cast(label, 'int32')

        pred, acc = mnist(img, label_int32)
        avg_loss = paddle.nn.functional.cross_entropy(pred, label_int32)
        avg_loss.backward(retain_graph=True)
        avg_loss = paddle.nn.functional.cross_entropy(pred, label_int32)
        avg_loss.backward()
        sgd.step()

    def _test_custom_device_mix_precision(self):
        import tempfile

        import paddle
        from paddle.inference import (
            PlaceType,
            PrecisionType,
            convert_to_mixed_precision,
        )
        from paddle.jit import to_static
        from paddle.static import InputSpec
        from paddle.vision.models import resnet50

        self.temp_dir = tempfile.TemporaryDirectory()
        model = resnet50(True)
        net = to_static(
            model,
            input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')],
            full_graph=True,
        )
        paddle.jit.save(
            net, os.path.join(self.temp_dir.name, 'resnet50/inference')
        )
        if paddle.framework.use_pir_api():
            return

        convert_to_mixed_precision(
            os.path.join(self.temp_dir.name, 'resnet50/inference.pdmodel'),
            os.path.join(self.temp_dir.name, 'resnet50/inference.pdiparams'),
            os.path.join(
                self.temp_dir.name, 'mixed_precision/inference.pdmodel'
            ),
            os.path.join(
                self.temp_dir.name, 'mixed_precision/inference.pdiparams'
            ),
            backend=PlaceType.CUSTOM,
            mixed_precision=PrecisionType.Half,
        )
        self.temp_dir.cleanup()

    def _test_custom_device_py_api(self):
        import paddle

        p = paddle.set_device('custom_cpu')
        paddle.device.synchronize('custom_cpu')

        s1 = paddle.device.Stream()
        s2 = paddle.device.Stream(p)

        s1 = paddle.device.current_stream()
        s2 = paddle.device.current_stream(p)

        e1 = paddle.device.Event()
        e2 = paddle.device.Event(p)

        s = paddle.device.Stream()
        e = paddle.device.Event()
        s.query()
        s.synchronize()
        s.wait_event(e)
        s.record_event(e)
        s.wait_stream(s)
        paddle.device.set_stream(s)

        e.query()
        e.synchronize()
        e.record(s)


if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # only support Linux now
        sys.exit()
    unittest.main()
