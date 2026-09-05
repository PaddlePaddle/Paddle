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

import unittest

import numpy as np
from op_test import get_device, get_places

import paddle
from paddle import base


class TestTensorUnfold(unittest.TestCase):
    def setUp(self):
        self.shape = [5, 5]
        self.typelist = ['float32', 'float64', 'int32', 'int64', 'float16']
        self.places = get_places()
        if base.core.is_compiled_with_cuda():
            self.places.append(base.CUDAPinnedPlace())

    def test_tensor_unfold_forward(self):
        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device(get_device())
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
                paddle.set_device(get_device())
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
        self.places = get_places()
        if base.core.is_compiled_with_cuda():
            self.places.append(base.CUDAPinnedPlace())

    def test_tensor_unfold_forward(self):
        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device(get_device())
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
                paddle.set_device(get_device())
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


class TestTensorUnfold_ZeroSize(TestTensorUnfold):
    def test_tensor_unfold_forward(self):
        self.shape = [5, 0]
        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device(get_device())
            for dtype in self.typelist:
                x_np = np.random.random(self.shape).astype(dtype)
                x = paddle.to_tensor(x_np, place=p)
                a = paddle.unfold(x, 0, 5, 1)
                np.testing.assert_allclose(a.numpy()[0], x_np.T)

    def test_tensor_unfold_backward(self):
        self.shape = [5, 0]
        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device(get_device())
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


class TestUnfoldAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape = [10, 10]
        self.dtype = "float32"
        self.init_data()

    def init_data(self):
        self.axis = 1
        self.size = 3
        self.step = 2

    def test_dygraph_compatibility(self):
        x = paddle.randn(self.shape, dtype=self.dtype)
        # Position args
        out1 = paddle.unfold(x, self.axis, self.size, self.step)
        # Key words args
        out2 = paddle.unfold(x, axis=self.axis, size=self.size, step=self.step)
        np.testing.assert_array_equal(out1.numpy(), out2.numpy())
        # Key words args for Alias
        out3 = paddle.unfold(
            x, dimension=self.axis, size=self.size, step=self.step
        )
        np.testing.assert_array_equal(out1.numpy(), out3.numpy())
        # Tensor method
        out4 = x.unfold(dimension=self.axis, size=self.size, step=self.step)
        np.testing.assert_array_equal(out1.numpy(), out4.numpy())


class TestTensorUnfoldOverlapBackward(unittest.TestCase):
    """With step < size the unfolded windows overlap, so every input element
    must receive the sum of the gradients of all windows containing it."""

    def setUp(self):
        self.places = get_places()

    def _check(self, shape, axis, size, step, dtype='float64'):
        x_np = np.random.random(shape).astype(dtype)
        expected = np.zeros(shape, dtype=dtype)
        num_window = (shape[axis] - size) // step + 1
        index = [slice(None)] * len(shape)
        for w in range(num_window):
            index[axis] = slice(w * step, w * step + size)
            expected[tuple(index)] += 1
        for place in self.places:
            with base.dygraph.guard(place):
                x = paddle.to_tensor(x_np)
                x.stop_gradient = False
                y = paddle.unfold(x, axis, size, step)
                y.backward(paddle.ones_like(y))
                np.testing.assert_allclose(x.grad.numpy(), expected)

    def test_overlapping_windows(self):
        self._check([5], 0, 3, 1)
        self._check([16], 0, 4, 1)
        self._check([8, 4], 0, 4, 2)

    def test_overlapping_windows_float32(self):
        self._check([5], 0, 3, 1, dtype='float32')

    def test_non_overlapping_windows(self):
        self._check([12], 0, 4, 4)
        self._check([12], -1, 2, 5)


if __name__ == '__main__':
    unittest.main()
