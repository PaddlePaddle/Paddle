#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np

import paddle
import paddle.nn as nn


class SimpleNet(nn.Layer):
    def __init__(self, in_size, out_size):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(in_size, in_size)
        self.linear2 = nn.Linear(in_size, out_size)

    def forward(self, x, hook=None, register=False, remove=False):
        ret1 = self.linear1(x)
        if hook is not None:
            if register:
                h = ret.register_hook(hook)
                if remove:
                    h.remove()
        ret2 = self.linear2(ret)
        out = paddle.mean(ret, axis=-1)
        return ret1, out


class TestTensorRegisterHook(unittest.TestCase):
    def setUp(self):
        self.seed = 2021
        self.in_size = 10
        self.out_size = 10
        self.batch_size = 4
        self.data = np.random.uniform(
            size=[self.batch_size, self.in_size]).astype('float32')
        self.label = np.random.uniform(
            size=[self.batch_size, 1]).astype('float32')
        self.devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            self.devices.append("gpu")

        paddle.seed(self.seed)

    def test_hook_for_interior_var(self):
        def hook_fn(grad):
            grad = grad * 2
            print(grad)
            return grad

        for device in self.devices:
            x = paddle.to_tensor([0., 1., 2., 3.])
            y = paddle.to_tensor([4., 5., 6., 7.])
            x.stop_gradient = False
            y.stop_gradient = False

            w = x + y
            w.stop_gradient = False
            w.register_hook(hook_fn)

            z = paddle.to_tensor([1., 2., 3., 4.])
            z.stop_gradient = False

            o = z.matmul(w)

            print('=====Start backprop=====')
            o.backward()
            print('=====End backprop=====')
            print('x.grad:', x.grad)
            print('y.grad:', y.grad)
            print('w.grad:', w.grad)
            print('z.grad:', z.grad)

    def test_hook_for_leaf_var(self):
        pass

    def test_hook_for_accumulated_grad(self):
        pass

    def test_lambda_hook(self):
        pass

    def test_hook_in_model(self):
        def register_and_remove_hook(hook=None, register=False, remove=False):
            for device in self.devices:
                net = SimpleNet(self.in_size, self.out_size)
                loss_fn = nn.MSELoss()

                data = paddle.to_tensor(self.data)
                label = paddle.to_tensor(self.label)

                ret1, out = net(data)
                loss = loss_fn(out, label)
                loss.backward()

                return ret1.grad, net.linear1.weight.grad, net.linear1.bias.grad

        pass


if __name__ == '__main__':
    unittest.main()
