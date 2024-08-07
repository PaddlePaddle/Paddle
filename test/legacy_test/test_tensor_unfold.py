#  Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import numpy as np

import paddle
from paddle import base


class TestTensorUnfold(unittest.TestCase):
    def setUp(self):
        self.shape = [5, 5]
        self.typelist = ['float32', 'float64', 'int32', 'int64', 'float16']
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))
            self.places.append(base.CUDAPinnedPlace())

    def test_tensor_unfold_forward(self):
        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in self.typelist:
                x_np = np.random.random(self.shape).astype(dtype)
                x = paddle.to_tensor(x_np, place=p)
                a = paddle.unfold(x, 0, 5, 1)
                np.testing.assert_allclose(a.numpy()[0], x_np.T)

    def test_tensor_unfold_backward(self):
        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in self.typelist:
                x_np = np.random.random(self.shape).astype(dtype)
                x = paddle.to_tensor(x_np, place=p)
                x.stop_gradient = False
                a = paddle.unfold(x, 0, 5, 1)
                b = a * 2
                b.retain_grads()
                loss = b.sum()
                loss.backward()
                self.assertEqual((b.grad.numpy() == 1).all().item(), True)


class TestTensorUnfold2(unittest.TestCase):
    def setUp(self):
        self.shape = [12]
        self.typelist = ['float32', 'float64', 'int32', 'int64', 'float16']
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))
            self.places.append(base.CUDAPinnedPlace())

    def test_tensor_unfold_forward(self):
        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in self.typelist:
                x_np = np.random.random(self.shape).astype(dtype)
                x = paddle.to_tensor(x_np, place=p)
                a = paddle.unfold(x, -1, 2, 5)
                target = np.stack((x_np[0:2], x_np[5:7], x_np[10:12]))
                np.testing.assert_allclose(a.numpy(), target)

    def test_tensor_unfold_backward(self):
        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            for dtype in self.typelist:
                x_np = np.random.random(self.shape).astype(dtype)
                x = paddle.to_tensor(x_np, place=p)
                x.stop_gradient = False
                a = paddle.unfold(x, -1, 2, 5)
                b = a * 2
                b.retain_grads()
                loss = b.sum()
                loss.backward()
                self.assertEqual((b.grad.numpy() == 1).all().item(), True)


if __name__ == '__main__':
    unittest.main()
