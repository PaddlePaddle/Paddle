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
from paddle.fluid.framework import _test_eager_guard
import paddle.fluid as fluid
import paddle.fluid.core as core


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


class SimpleNetForStatic(nn.Layer):
    def __init__(self, in_size, out_size):
        super(SimpleNetForStatic, self).__init__()
        self.linear1 = nn.Linear(in_size, in_size)
        self.linear2 = nn.Linear(in_size, out_size)

    def forward(self, x):
        ret1 = self.linear1(x)
        ret1.register_hook(lambda grad: grad * 2)

        ret2 = self.linear2(ret1)
        out = paddle.mean(ret2, axis=-1)
        return out


class TestTensorRegisterHook(unittest.TestCase):
    def setUp(self):
        self.seed = 2021
        self.in_size = 10
        self.out_size = 10
        self.batch_size = 4
        self.devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            self.devices.append("gpu")

    def func_hook_for_interior_var(self):
        def run_double_hook_for_interior_var(double_hook, removed=False):
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
                self.assertTrue(np.array_equal(z.grad.numpy(), w.numpy()))
                # w.grad is not changed by hook
                self.assertTrue(np.array_equal(w.grad.numpy(), z.numpy()))
                # x.grad and y.grad are changed if run hook
                self.assertTrue(
                    np.array_equal(x.grad.numpy(),
                                   z.numpy() * 2 if not removed else z.numpy()))
                self.assertTrue(
                    np.array_equal(y.grad.numpy(),
                                   z.numpy() * 2 if not removed else z.numpy()))

        def run_print_hook_for_interior_var(print_hook, removed=False):
            for device in self.devices:
                paddle.set_device(device)

                x = paddle.to_tensor([0., 1., 2., 3.])
                y = paddle.to_tensor([4., 5., 6., 7.])
                x.stop_gradient = False
                y.stop_gradient = False

                w = x + y
                w.stop_gradient = False
                helper = w.register_hook(print_hook)

                z = paddle.to_tensor([1., 2., 3., 4.])
                z.stop_gradient = False

                o = z.matmul(w)

                # remove hook before backward
                if removed:
                    helper.remove()

                o.backward()

                # all grads are not affected
                self.assertTrue(np.array_equal(z.grad.numpy(), w.numpy()))
                self.assertTrue(np.array_equal(w.grad.numpy(), z.numpy()))
                self.assertTrue(np.array_equal(x.grad.numpy(), z.numpy()))
                self.assertTrue(np.array_equal(y.grad.numpy(), z.numpy()))

        def double_hook(grad):
            grad = grad * 2
            print(grad)
            return grad

        def print_hook(grad):
            print(grad)

        # register hook
        run_double_hook_for_interior_var(double_hook)
        # register hook and removed
        run_double_hook_for_interior_var(double_hook, removed=True)

        # register hook
        run_double_hook_for_interior_var(lambda grad: grad * 2)
        # register hook and removed
        run_double_hook_for_interior_var(lambda grad: grad * 2, removed=True)

        # register hook
        run_print_hook_for_interior_var(print_hook)
        # register hook and removed
        run_print_hook_for_interior_var(print_hook, removed=True)

    def test_hook_for_interior_var(self):
        with _test_eager_guard():
            self.func_hook_for_interior_var()
        self.func_hook_for_interior_var()

    def func_hook_for_leaf_var(self):
        def run_double_hook_for_leaf_var(double_hook, removed=False):
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
                self.assertTrue(np.array_equal(z.grad.numpy(), w.numpy()))
                self.assertTrue(np.array_equal(w.grad.numpy(), z.numpy()))
                self.assertTrue(np.array_equal(x.grad.numpy(), z.numpy()))
                # y.grad are changed if run hook
                self.assertTrue(
                    np.array_equal(y.grad.numpy(),
                                   z.numpy() * 2 if not removed else z.numpy()))

        # register hook
        run_double_hook_for_leaf_var(lambda grad: grad * 2)
        # register hook and removed
        run_double_hook_for_leaf_var(lambda grad: grad * 2, removed=True)

    def test_hook_for_leaf_var(self):
        with _test_eager_guard():
            self.func_hook_for_leaf_var()
        self.func_hook_for_leaf_var()

    def func_hook_for_accumulated_grad_interior_var(self):
        def run_double_hook_for_accumulated_grad_interior_var(double_hook,
                                                              removed=False):
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
                self.assertTrue(np.array_equal(x.grad.numpy(), base_grad))
                # b.grad is changed by x.hook
                self.assertTrue(
                    np.array_equal(b.grad.numpy(), base_grad * 2
                                   if not removed else base_grad))
                # a.grad is changed by x.hook and a.hook
                self.assertTrue(
                    np.array_equal(a.grad.numpy(), base_grad * 4
                                   if not removed else base_grad))

        # register hook
        run_double_hook_for_accumulated_grad_interior_var(lambda grad: grad * 2)
        # register hook and removed
        run_double_hook_for_accumulated_grad_interior_var(
            lambda grad: grad * 2, removed=True)

    def test_hook_for_accumulated_grad_interior_var(self):
        with _test_eager_guard():
            self.func_hook_for_accumulated_grad_interior_var()
        self.func_hook_for_accumulated_grad_interior_var()

    def func_hook_for_accumulated_grad_leaf_var(self):
        def run_double_hook_for_accumulated_grad_leaf_var(double_hook,
                                                          removed=False):
            for device in self.devices:
                paddle.set_device(device)

                x = paddle.to_tensor([0., 1., 2., 4.])
                x.stop_gradient = False

                helper = x.register_hook(double_hook)

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
                    helper.remove()

                o.backward()

                base_grad = np.array([5., 9., 13., 19.])
                # x.grad is changed by x.hook
                self.assertTrue(
                    np.array_equal(x.grad.numpy(), base_grad * 2
                                   if not removed else base_grad))

        # register hook
        run_double_hook_for_accumulated_grad_leaf_var(lambda grad: grad * 2)
        # register hook and removed
        run_double_hook_for_accumulated_grad_leaf_var(
            lambda grad: grad * 2, removed=True)

    def test_hook_for_accumulated_grad_leaf_var(self):
        with _test_eager_guard():
            self.func_hook_for_accumulated_grad_leaf_var()
        self.func_hook_for_accumulated_grad_leaf_var()

    def func_hook_in_model(self):
        def run_double_hook_in_model(data,
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

                return (ret1.grad.numpy(), net.linear1.weight.grad.numpy(),
                        net.linear1.bias.grad.numpy())

        data = np.random.uniform(
            size=[self.batch_size, self.in_size]).astype('float32')
        label = np.random.uniform(size=[self.batch_size, 1]).astype('float32')

        # get original value
        ret1_grad, linear1_w_grad, linear1_b_grad = run_double_hook_in_model(
            data, label)
        # get value changed by hook
        ret1_grad_hook, linear1_w_grad_hook, linear1_b_grad_hook = run_double_hook_in_model(
            data, label, lambda grad: grad * 2, True)
        # get value after removing hook
        ret1_grad_rm, linear1_w_grad_rm, linear1_b_grad_rm = run_double_hook_in_model(
            data, label, lambda grad: grad * 2, True, True)

        # compare original value and with hook
        self.assertTrue(np.array_equal(ret1_grad, ret1_grad_hook))
        self.assertTrue(np.array_equal(linear1_w_grad * 2, linear1_w_grad_hook))
        self.assertTrue(np.array_equal(linear1_b_grad * 2, linear1_b_grad_hook))

        # compare original value and remove hook
        self.assertTrue(np.array_equal(ret1_grad, ret1_grad_rm))
        self.assertTrue(np.array_equal(linear1_w_grad, linear1_w_grad_rm))
        self.assertTrue(np.array_equal(linear1_b_grad, linear1_b_grad_rm))

    def test_func_hook_in_model(self):
        with _test_eager_guard():
            self.func_hook_in_model()
        self.func_hook_in_model()

    def func_multiple_hooks_for_interior_var(self):
        def run_multiple_hooks_for_interior_var(device,
                                                hooks,
                                                remove1=False,
                                                remove2=False,
                                                remove3=False):
            paddle.set_device(device)

            x = paddle.to_tensor([0., 1., 2., 3.])
            y = paddle.to_tensor([4., 5., 6., 7.])
            x.stop_gradient = False
            y.stop_gradient = False

            w = x + y
            w.stop_gradient = False

            helpers = []
            for hook in hooks:
                helper = w.register_hook(hook)
                helpers.append(helper)

            z = paddle.to_tensor([1., 2., 3., 4.])
            z.stop_gradient = False

            o = z.matmul(w)

            if remove1:
                helpers[0].remove()
            if remove2:
                helpers[1].remove()
            if remove3:
                helpers[2].remove()

            o.backward()

            return z.numpy(), w.grad.numpy(), x.grad.numpy(), y.grad.numpy()

        def double_hook(grad):
            return grad * 2

        hooks = [double_hook, double_hook, double_hook]

        for device in self.devices:
            z, w_grad, x_grad, y_grad = run_multiple_hooks_for_interior_var(
                device, hooks)

            self.assertTrue(np.array_equal(w_grad, z))
            self.assertTrue(np.array_equal(x_grad, z * 8))
            self.assertTrue(np.array_equal(y_grad, z * 8))

            z, w_grad, x_grad, y_grad = run_multiple_hooks_for_interior_var(
                device, hooks, remove1=True)

            self.assertTrue(np.array_equal(w_grad, z))
            self.assertTrue(np.array_equal(x_grad, z * 4))
            self.assertTrue(np.array_equal(y_grad, z * 4))

            z, w_grad, x_grad, y_grad = run_multiple_hooks_for_interior_var(
                device, hooks, remove2=True)

            self.assertTrue(np.array_equal(w_grad, z))
            self.assertTrue(np.array_equal(x_grad, z * 4))
            self.assertTrue(np.array_equal(y_grad, z * 4))

            z, w_grad, x_grad, y_grad = run_multiple_hooks_for_interior_var(
                device, hooks, remove3=True)

            self.assertTrue(np.array_equal(w_grad, z))
            self.assertTrue(np.array_equal(x_grad, z * 4))
            self.assertTrue(np.array_equal(y_grad, z * 4))

            z, w_grad, x_grad, y_grad = run_multiple_hooks_for_interior_var(
                device, hooks, remove1=True, remove2=True, remove3=True)

            self.assertTrue(np.array_equal(w_grad, z))
            self.assertTrue(np.array_equal(x_grad, z))
            self.assertTrue(np.array_equal(y_grad, z))

    def test_multiple_hooks_for_interior_var(self):
        with _test_eager_guard():
            self.func_multiple_hooks_for_interior_var()
        self.func_multiple_hooks_for_interior_var()

    def func_hook_in_double_grad(self):
        def double_print_hook(grad):
            grad = grad * 2
            print(grad)
            return grad

        x = paddle.ones(shape=[1], dtype='float32')
        x.stop_gradient = False

        # hook only works in backward
        # for forward var x, the x.grad generated in
        # paddle.grad will not deal with by hook
        x.register_hook(double_print_hook)

        y = x * x
        fluid.set_flags({'FLAGS_retain_grad_for_all_tensor': False})
        # Since y = x * x, dx = 2 * x
        dx = paddle.grad(
            outputs=[y], inputs=[x], create_graph=True, retain_graph=True)[0]
        fluid.set_flags({'FLAGS_retain_grad_for_all_tensor': True})

        z = y + dx
        self.assertTrue(x.grad is None)

        # If create_graph = True, the gradient of dx
        # would be backpropagated. Therefore,
        # z = x * x + dx = x * x + 2 * x, and
        # x.gradient() = 2 * x + 2 = 4.0
        # after changed by hook: 8.0

        # TODO(wuweilong): enable this case when DoubleGrad in eager mode is ready
        if fluid.in_dygraph_mode():
            pass
        else:
            z.backward()
            self.assertTrue(np.array_equal(x.grad.numpy(), np.array([8.])))

    def test_hook_in_double_grad(self):
        with _test_eager_guard():
            self.func_hook_in_double_grad()
        self.func_hook_in_double_grad()

    def func_remove_one_hook_multiple_times(self):
        for device in self.devices:
            paddle.set_device(device)

            x = paddle.to_tensor([1., 2., 3., 4.])
            x.stop_gradient = False

            h = x.register_hook(lambda grad: grad * 2)
            self.assertTrue(h.remove())
            self.assertFalse(h.remove())

    def test_remove_one_hook_multiple_times(self):
        with _test_eager_guard():
            self.func_remove_one_hook_multiple_times()
        self.func_remove_one_hook_multiple_times()

    def func_register_hook_for_stop_gradient_var(self):
        for device in self.devices:
            paddle.set_device(device)

            x = paddle.to_tensor([1., 2., 3., 4.])

            with self.assertRaises(RuntimeError):
                x.register_hook(lambda grad: grad * 2)

    def test_register_hook_for_stop_gradient_var(self):
        with _test_eager_guard():
            self.func_register_hook_for_stop_gradient_var()
        self.func_register_hook_for_stop_gradient_var()

    def test_register_hook_in_static_mode(self):
        paddle.enable_static()

        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(paddle.static.Scope()):
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name='x', shape=[None, self.in_size], dtype='float32')

                net = SimpleNetForStatic(self.in_size, self.out_size)
                with self.assertRaises(AssertionError):
                    out = net(x)

        paddle.disable_static()

    def func_register_hook_in_dy2static_mode(self):
        net = SimpleNetForStatic(self.in_size, self.out_size)
        jit_net = paddle.jit.to_static(
            net, input_spec=[paddle.static.InputSpec([None, self.in_size])])

        data = np.random.uniform(
            size=[self.batch_size, self.in_size]).astype('float32')
        data_t = paddle.to_tensor(data)

        with self.assertRaises(AssertionError):
            out = jit_net(data_t)

    def test_register_hook_in_dy2static_mode(self):
        with _test_eager_guard():
            self.func_register_hook_in_dy2static_mode()
        self.func_register_hook_in_dy2static_mode()


HOOK_INIT_VALUE = 10
HOOK_IS_CALLED = False


def global_void_hook():
    global HOOK_INIT_VALUE
    global HOOK_IS_CALLED
    HOOK_INIT_VALUE *= 2
    HOOK_IS_CALLED = True


class TestTensorRegisterBackwardHook(unittest.TestCase):
    def setUp(self):
        self.devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            self.devices.append("gpu")

    def func_register_backward_hook(self):
        global HOOK_INIT_VALUE
        global HOOK_IS_CALLED
        for device in self.devices:
            x = paddle.to_tensor(5., stop_gradient=False)
            x._register_backward_hook(global_void_hook)
            for i in range(5):
                y = paddle.pow(x, 4.0)
                y.backward()

            self.assertEqual(HOOK_INIT_VALUE, 320)
            self.assertTrue(HOOK_IS_CALLED)

            # reset initial value
            HOOK_INIT_VALUE = 10
            HOOK_IS_CALLED = False

    def test_register_backward_hook(self):
        with _test_eager_guard():
            self.func_register_backward_hook()
        self.func_register_backward_hook()

    def func_register_backward_hook_for_interior_var(self):
        x = paddle.to_tensor(5., stop_gradient=False)
        y = paddle.pow(x, 4.0)

        with self.assertRaises(ValueError):
            y._register_backward_hook(global_void_hook)

    def test_register_backward_hook_for_interior_var(self):
        with _test_eager_guard():
            self.func_register_backward_hook_for_interior_var()
        self.func_register_backward_hook_for_interior_var()

    def func_register_backward_hook_for_var_without_gradient(self):
        x = paddle.to_tensor(5.)
        y = paddle.pow(x, 4.0)

        with self.assertRaises(ValueError):
            x._register_backward_hook(global_void_hook)

    def test_register_backward_hook_for_var_without_gradient(self):
        with _test_eager_guard():
            self.func_register_backward_hook_for_var_without_gradient()
        self.func_register_backward_hook_for_var_without_gradient()


if __name__ == '__main__':
    unittest.main()
