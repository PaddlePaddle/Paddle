# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import paddle
from paddle.fluid import layers


class TestAlphaDropout(unittest.TestCase):
    def __init__(self, p, place="cpu", dtype='float64'):
        super(TestAlphaDropout, self).__init__(methodName="runTest")
        self.p = p
        self.place = paddle.CPUPlace() if place == "cpu" else paddle.CUDAPlace(
            0)
        self.dtype = dtype

    def setUp(self) -> None:
        self.alpha_dropout = paddle.nn.AlphaDropout(self.p)

    def numpy_alpha_dropout(self, x, p, training=True):
        def f_scale(x, scale=1.0, bias=0.0):
            out = scale * x + bias
            return out

        if training:
            if p == 1:
                return f_scale(x, scale=0.)
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            alpha_p = -alpha * scale
            a = ((1 - p) * (1 + p * alpha_p**2))**-0.5
            b = -a * alpha_p * p

            dtype = x.dtype
            input_shape = x.shape
            random_tensor = layers.uniform_random(
                input_shape, dtype='float32', min=0., max=1.)
            random_tensor = random_tensor.numpy()
            p = np.ones(input_shape, dtype='float32') * p
            keep_mask = np.greater_equal(random_tensor, p)
            keep_mask = keep_mask.astype(dtype)
            drop_mask = np.subtract(np.ones(shape=input_shape), keep_mask)

            b = np.ones(input_shape, dtype=dtype) * b
            y = x * keep_mask + f_scale(drop_mask, scale=alpha_p)
            res = f_scale(y, scale=a) + b
            return res
        else:
            return x

    def test_normal_state(self):
        x = np.random.randn(2, 3, 3).astype(self.dtype)
        # to ensure has the same random mask
        paddle.seed(100)
        y_train1 = self.numpy_alpha_dropout(x, self.p)
        paddle.seed(100)
        y_train2 = self.alpha_dropout(
            paddle.to_tensor(
                x, place=self.place, dtype=self.dtype))
        np.testing.assert_allclose(y_train1, y_train2, atol=1e-8, rtol=1e-5)

        self.alpha_dropout.eval()
        paddle.seed(101)
        y_test1 = self.numpy_alpha_dropout(x, self.p, False)
        paddle.seed(101)
        y_test2 = self.alpha_dropout(
            paddle.to_tensor(
                x, place=self.place, dtype=self.dtype))
        np.testing.assert_allclose(y_test1, y_test2, atol=1e-8, rtol=1e-5)

    def runTest(self):
        self.test_normal_state()


class TestAlphaDropoutParams(unittest.TestCase):
    def __init__(self, place):
        super(TestAlphaDropoutParams, self).__init__(methodName="runTest")
        self.place = paddle.CPUPlace() if place == "cpu" else paddle.CUDAPlace(
            0)

    def setUp(self) -> None:
        self.x = np.array([[-3, 1], [-1, 1]]).astype('float32')
        self.x = paddle.to_tensor(self.x, place=self.place)

    def test_p_low_error(self):
        with self.assertRaises(ValueError):
            m = paddle.nn.AlphaDropout(p=-1)
            m(self.x)

    def test_p_high_error(self):
        with self.assertRaises(ValueError):
            m = paddle.nn.AlphaDropout(p=2)
            m(self.x)

    def test_p_type_error(self):
        with self.assertRaises(TypeError):
            m = paddle.nn.AlphaDropout(p='1')
            m(self.x)

    def runTest(self):
        self.test_p_low_error()
        self.test_p_high_error()
        self.test_p_type_error()


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    devices = ["cpu", "gpu"] if paddle.fluid.is_compiled_with_cuda() \
        else ["cpu"]
    for p in [0., 0.5, 1]:
        for device in devices:
            for dtpye in ["float32", "float64"]:
                for test_class in [TestAlphaDropout]:
                    suite.addTest(test_class(p, device, dtpye))

    for device in devices:
        for test_class in [TestAlphaDropoutParams]:
            suite.addTest(test_class(device))

    return suite


if __name__ == '__main__':
    unittest.main()
