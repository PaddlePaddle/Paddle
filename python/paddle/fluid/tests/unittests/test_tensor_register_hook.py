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
                h = ret1.register_hook(hook)
                if remove:
                    h.remove()
        ret2 = self.linear2(ret1)
        out = paddle.mean(ret2, axis=-1)
        return ret1, out


class TestTensorRegisterHook(unittest.TestCase):
    def setUp(self):
        self.seed = 2021
        self.in_size = 10
        self.out_size = 10
        self.batch_size = 4
        self.devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            self.devices.append("gpu")

    def run_hook_for_interior_var(self, double_hook, removed=False):
        for device in self.devices:
            paddle.set_device(device)

            x = paddle.to_tensor([0., 1., 2., 3.])
            y = paddle.to_tensor([4., 5., 6., 7.])
            x.stop_gradient = False
            y.stop_gradient = False

            w = x + y
            w.stop_gradient = False
            helper = w.register_hook(double_hook)

            z = paddle.to_tensor([1., 2., 3., 4.])
            z.stop_gradient = False

            o = z.matmul(w)

            # remove hook before backward
            if removed:
                helper.remove()

            o.backward()

            # z.grad is not affected
            self.assertTrue(np.array_equal(z.grad, w.numpy()))
            # w.grad is not changed by hook
            self.assertTrue(np.array_equal(w.grad, z.numpy()))
            # x.grad and y.grad are changed if run hook
            self.assertTrue(
                np.array_equal(x.grad,
                               z.numpy() * 2 if not removed else z.numpy()))
            self.assertTrue(
                np.array_equal(y.grad,
                               z.numpy() * 2 if not removed else z.numpy()))

    def run_hook_for_leaf_var(self, double_hook, removed=False):
        for device in self.devices:
            paddle.set_device(device)

            x = paddle.to_tensor([0., 1., 2., 3.])
            y = paddle.to_tensor([4., 5., 6., 7.])
            x.stop_gradient = False
            y.stop_gradient = False
            helper = y.register_hook(double_hook)

            w = x + y
            w.stop_gradient = False

            z = paddle.to_tensor([1., 2., 3., 4.])
            z.stop_gradient = False

            o = z.matmul(w)

            # remove hook before backward
            if removed:
                helper.remove()

            o.backward()

            # z.grad, w.grad, x.grad is not affected
            self.assertTrue(np.array_equal(z.grad, w.numpy()))
            self.assertTrue(np.array_equal(w.grad, z.numpy()))
            self.assertTrue(np.array_equal(x.grad, z.numpy()))
            # y.grad are changed if run hook
            self.assertTrue(
                np.array_equal(y.grad,
                               z.numpy() * 2 if not removed else z.numpy()))

    def run_hook_for_accumulated_grad(self, double_hook, removed=False):
        for device in self.devices:
            paddle.set_device(device)

            a = paddle.to_tensor([0., 1., 1., 2.])
            b = paddle.to_tensor([0., 0., 1., 2.])
            a.stop_gradient = False
            b.stop_gradient = False

            helper1 = a.register_hook(double_hook)

            x = a + b
            x.stop_gradient = False

            helper2 = x.register_hook(double_hook)

            y = paddle.to_tensor([4., 5., 6., 7.])
            z = paddle.to_tensor([1., 2., 3., 4.])
            y.stop_gradient = False
            z.stop_gradient = False

            o1 = x + y
            o2 = x + z
            o1.stop_gradient = False
            o2.stop_gradient = False

            o = o1.matmul(o2)

            # remove hook before backward
            if removed:
                helper1.remove()
                helper2.remove()

            o.backward()

            base_grad = np.array([5., 9., 13., 19.])
            # x.grad is not changed
            self.assertTrue(np.array_equal(x.grad, base_grad))
            # b.grad is changed by x.hook
            self.assertTrue(
                np.array_equal(b.grad, base_grad * 2
                               if not removed else base_grad))
            # a.grad is changed by x.hook and a.hook
            self.assertTrue(
                np.array_equal(a.grad, base_grad * 4
                               if not removed else base_grad))

    def run_hook_in_model(self,
                          data,
                          label,
                          hook=None,
                          register=False,
                          remove=False):
        for device in self.devices:
            paddle.seed(self.seed)
            paddle.set_device(device)

            net = SimpleNet(self.in_size, self.out_size)
            loss_fn = nn.MSELoss()

            data = paddle.to_tensor(data)
            label = paddle.to_tensor(label)

            ret1, out = net(data, hook, register, remove)
            loss = loss_fn(out, label)
            loss.backward()

            return ret1.grad, net.linear1.weight.grad, net.linear1.bias.grad

    def test_func_hook_for_interior_var(self):
        def hook_fn(grad):
            grad = grad * 2
            print(grad)
            return grad

        # register hook
        self.run_hook_for_interior_var(hook_fn)
        # register hook and removed
        self.run_hook_for_interior_var(hook_fn, removed=True)

    def test_lambda_hook_for_interior_var(self):
        # register hook
        self.run_hook_for_interior_var(lambda grad: grad * 2)
        # register hook and removed
        self.run_hook_for_interior_var(lambda grad: grad * 2, removed=True)

    def test_hook_for_leaf_var(self):
        # register hook
        self.run_hook_for_leaf_var(lambda grad: grad * 2)
        # register hook and removed
        self.run_hook_for_leaf_var(lambda grad: grad * 2, removed=True)

    def test_hook_for_accumulated_grad(self):
        # register hook
        self.run_hook_for_accumulated_grad(lambda grad: grad * 2)
        # register hook and removed
        self.run_hook_for_accumulated_grad(lambda grad: grad * 2, removed=True)

    def test_hook_in_model(self):
        data = np.random.uniform(
            size=[self.batch_size, self.in_size]).astype('float32')
        label = np.random.uniform(size=[self.batch_size, 1]).astype('float32')

        # get original value
        ret1_grad, linear1_w_grad, linear1_b_grad = self.run_hook_in_model(
            data, label)
        # get value changed by hook
        ret1_grad_hook, linear1_w_grad_hook, linear1_b_grad_hook = self.run_hook_in_model(
            data, label, lambda grad: grad * 2, True)
        # get value after removing hook
        ret1_grad_rm, linear1_w_grad_rm, linear1_b_grad_rm = self.run_hook_in_model(
            data, label, lambda grad: grad * 2, True, True)

        # compare original value and with hook
        self.assertTrue(np.array_equal(ret1_grad, ret1_grad_hook))
        self.assertTrue(np.array_equal(linear1_w_grad * 2, linear1_w_grad_hook))
        self.assertTrue(np.array_equal(linear1_b_grad * 2, linear1_b_grad_hook))

        # compare original value and remove hook
        self.assertTrue(np.array_equal(ret1_grad, ret1_grad_rm))
        self.assertTrue(np.array_equal(linear1_w_grad, linear1_w_grad_rm))
        self.assertTrue(np.array_equal(linear1_b_grad, linear1_b_grad_rm))


if __name__ == '__main__':
    unittest.main()
