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
from op_test import get_device, get_places, is_custom_device

import paddle
from paddle import base


class TestAsStrided(unittest.TestCase):
    def setUp(self):
        self.shape = [32, 32]
        self.typelist = ['float32', 'float64', 'int32', 'int64', 'float16']
        self.places = get_places()
        if base.core.is_compiled_with_cuda() or is_custom_device():
            self.places.append(base.CUDAPinnedPlace())

    def test_as_strided_forward(self):
        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device(get_device())
            for dtype in self.typelist:
                x_np = np.random.random(self.shape).astype(dtype)
                x = paddle.to_tensor(x_np, place=p)
                a = paddle.as_strided(x, shape=(3, 4), stride=(32, 1))
                np.testing.assert_allclose(a.numpy(), x_np[:3, :4])

    def test_as_strided_backward(self):
        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device(get_device())
            for dtype in self.typelist:
                x_np = np.random.random(self.shape).astype(dtype)
                x = paddle.to_tensor(x_np, place=p)
                x.stop_gradient = False
                a = paddle.as_strided(x, shape=(3,), stride=(1,))
                b = a * 2
                b.retain_grads()
                loss = b.sum()
                loss.backward()
                self.assertEqual((b.grad.numpy() == 1).all().item(), True)


class TestAsStrided_ZeroSize(unittest.TestCase):
    def setUp(self):
        self.places = get_places()

    def test_as_strided_forward(self):
        for place in self.places:
            with base.dygraph.guard(place):
                a = paddle.to_tensor(
                    np.random.random([0, 32]).astype('float32')
                )
                a.stop_gradient = False
                b = paddle.as_strided(a, shape=(0, 4), stride=(32, 1))
                np.testing.assert_equal(b.shape, [0, 4])
                b.backward(paddle.ones_like(b))
                np.testing.assert_equal(a.grad.shape, [0, 32])

    def test_as_strided_error(self):
        for place in self.places:
            with base.dygraph.guard(place):
                self.assertRaises(
                    ValueError,
                    paddle.as_strided,
                    x=paddle.to_tensor(
                        np.random.random([0, 32]).astype('float32')
                    ),
                    shape=[3, 4],
                    stride=[32, 1],
                )


if __name__ == '__main__':
    unittest.main()
