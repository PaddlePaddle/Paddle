# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.autograd.py_layer import LegacyPyLayer, EagerPyLayer
from paddle.fluid.framework import _test_eager_guard, in_dygraph_mode


class FakeTensor(paddle.fluid.core.VarBase):

    def __init__(self):
        pass


class TestPyLayer(unittest.TestCase):

    def func_test_simple_pylayer_multiple_output(self):

        class tanh(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1, x2, func1, func2=paddle.square):
                ctx.func = func2
                y1 = func1(x1)
                y2 = func1(x2)
                ctx.save_for_backward(y1, y2)
                return y1, 1, y2, None

            @staticmethod
            def backward(ctx, dy1, dy2):
                y1, y2 = ctx.saved_tensor()
                re1 = dy1 * (1 - ctx.func(y1))
                re2 = dy2 * (1 - paddle.square(y2))
                return re1, re2

        input1 = paddle.randn([2, 3]).astype("float64")
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        z = tanh.apply(input1, input1, paddle.tanh, paddle.square)
        z = z[0] + z[2]
        z.mean().backward()

        z2 = paddle.tanh(input2) + paddle.tanh(input2)
        z2.mean().backward()

        self.assertTrue(
            np.max(np.abs((input1.grad.numpy() - input2.grad.numpy()))) < 1e-10)

    def test_simple_pylayer_multiple_output(self):
        with _test_eager_guard():
            self.func_test_simple_pylayer_multiple_output()
        self.func_test_simple_pylayer_multiple_output()

    def func_test_simple_pylayer_return_none_with_no_grad(self):

        class tanh(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1, x2, func1, func2=paddle.square):
                ctx.func = func2
                y1 = func1(x1)
                y2 = func1(x2)
                ctx.save_for_backward(y1, y2)
                return 1, None, y1, y2, ''

            @staticmethod
            def backward(ctx, dy1, dy2):
                y1, y2 = ctx.saved_tensor()
                re1 = dy1 * (1 - ctx.func(y1))
                re2 = dy2 * (1 - paddle.square(y2))
                return re1, None

        input1 = paddle.randn([2, 3]).astype("float64")
        input2 = input1.detach().clone()
        input3 = input1.detach().clone()
        input4 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        input3.stop_gradient = True
        input4.stop_gradient = True
        z = tanh.apply(input1, input3, paddle.tanh, paddle.square)
        z = z[2] + z[3]
        z.mean().backward()

        z2 = paddle.tanh(input2) + paddle.tanh(input4)
        z2.mean().backward()

        self.assertTrue(
            np.max(np.abs((input1.grad.numpy() - input2.grad.numpy()))) < 1e-10)

    def test_simple_pylayer_return_none_with_no_grad(self):
        with _test_eager_guard():
            self.func_test_simple_pylayer_return_none_with_no_grad()
        self.func_test_simple_pylayer_return_none_with_no_grad()

    def func_test_simple_pylayer_single_output(self):

        class tanh(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1, func1, func2=paddle.square):
                ctx.func = func2
                y1 = func1(x1)
                ctx.save_for_backward(y1)
                return y1

            @staticmethod
            def backward(ctx, dy1):
                y1, = ctx.saved_tensor()
                re1 = dy1 * (1 - ctx.func(y1))
                return re1

        input1 = paddle.randn([2, 3]).astype("float64")
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        z = tanh.apply(x1=input1, func1=paddle.tanh)
        z.mean().backward()
        z2 = paddle.tanh(input2)
        z2.mean().backward()

        self.assertTrue(
            np.max(np.abs((input1.grad.numpy() - input2.grad.numpy()))) < 1e-10)

    def test_simple_pylayer_single_output(self):
        with _test_eager_guard():
            self.func_test_simple_pylayer_single_output()
        self.func_test_simple_pylayer_single_output()

    def func_test_pylayer_num_output_match(self):

        class tanh(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(
                ctx,
                x1,
                x2,
            ):
                return x1 + x2

            @staticmethod
            def backward(ctx, dy1):
                return dy1 + 1

        input1 = paddle.randn([2, 3]).astype("float64")
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        z = tanh.apply(input1, input2)
        with self.assertRaises(ValueError):
            z.mean().backward()

    def test_pylayer_num_output_match(self):
        with _test_eager_guard():
            self.func_test_pylayer_num_output_match()
        self.func_test_pylayer_num_output_match()

    def func_test_pylayer_dtype(self):

        class tanh(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x, dtype):
                y = paddle.cast(x, dtype)
                return y

            @staticmethod
            def backward(ctx, dy1):
                return dy1

        dtypes = [
            'bool', 'float16', 'float32', 'float64', 'uint8', 'int32', 'int64'
        ]
        for dtype in dtypes:
            input1 = (paddle.randn([2, 3]))
            input1.stop_gradient = False
            self.assertTrue(input1.grad is None)

            z = tanh.apply(input1, dtype)
            z = paddle.cast(z, "float32")
            z.sum().backward()
            self.assertTrue(input1.grad is not None)

    def test_pylayer_dtype(self):
        with _test_eager_guard():
            self.func_test_pylayer_dtype()
        self.func_test_pylayer_dtype()

    def func_test_pylayer_Exception_forward(self):

        class Layer_None1(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, *args):
                return None

            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        with self.assertRaises(ValueError):
            z = Layer_None1.apply(input1)

        class Layer_None2(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, *args):
                return [None, args[0]]

            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        # return None
        z = Layer_None2.apply(input1)

        class Layer_one1(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, *args):
                return 1

            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        # At least one output of `PyLayer.backward` is a `Tensor`
        with self.assertRaises(ValueError):
            z = Layer_one1.apply(input1)

        class Layer_one2(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, *args):
                return [1, 2, args[0]]

            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        # return int
        z = Layer_one2.apply(input1)

        class Layer_no_fw(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        with self.assertRaises(NotImplementedError):
            z = Layer_no_fw.apply(input1)

    def test_pylayer_Exception_forward(self):
        with _test_eager_guard():
            self.func_test_pylayer_Exception_forward()
        self.func_test_pylayer_Exception_forward()

    def func_test_pylayer_nograd(self):

        class tanh(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1, func1, func2=paddle.square, xx=None):
                ctx.func = func2
                y1 = func1(x1)
                return y1

            @staticmethod
            def backward(ctx, x1, y1, dy1):
                re1 = dy1 * (1 - ctx.func(y1))
                return re1

        input1 = paddle.randn([2, 3]).astype("float64")
        z = tanh.apply(input1, paddle.tanh, paddle.square)
        z.mean().backward()
        self.assertTrue(z.grad is None)

    def test_pylayer_nograd(self):
        with _test_eager_guard():
            self.func_test_pylayer_nograd()
        self.func_test_pylayer_nograd()

    def func_test_pylayer_Exception_bk(self):

        class Layer_bk_none1(
                EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, dy1):
                return None

        input2 = paddle.randn([2, 3]).astype("float64")
        input2.stop_gradient = False
        z = Layer_bk_none1.apply(input2)

        with self.assertRaises(ValueError):
            z.sum().backward()

        class Layer_bk_none2(
                EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1, x2):
                return x1 + x2

            @staticmethod
            def backward(ctx, dy1):
                return None, dy1

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        z = Layer_bk_none2.apply(input1, input1)

        with self.assertRaises(ValueError):
            z.mean().backward()

        class Layer_bk_one1(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer
                            ):

            @staticmethod
            def forward(ctx, x):
                return x + x

            @staticmethod
            def backward(ctx, dy):
                return 1

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        z = Layer_bk_one1.apply(input1)

        with self.assertRaises(ValueError):
            z.mean().backward()

        class Layer_bk_one2(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer
                            ):

            @staticmethod
            def forward(ctx, x1, x2):
                return x1 * 2, x2 * 5

            @staticmethod
            def backward(ctx, *args):
                return 1, 1

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False

        y = Layer_bk_one2.apply(input1, input1)
        z = y[0] + y[1]
        with self.assertRaises(ValueError):
            z.mean().backward()

        class Layer_no_bk(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x):
                return x * 2, x * 5

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        z = Layer_no_bk.apply(input1)

        with self.assertRaises(OSError):
            z = z[0] + z[1]
            z.mean().backward()

        class Layer_bk_match(
                EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x):
                return x * 2, x * 5

            @staticmethod
            def backward(ctx, dy1, dy2):
                return dy2 * 2, dy1 * 2

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        z = Layer_bk_match.apply(input1)
        with self.assertRaises(ValueError):
            z = z[0] + z[1]
            z.mean().backward()

    def test_pylayer_Exception_bk(self):
        with _test_eager_guard():
            self.func_test_pylayer_Exception_bk()
        self.func_test_pylayer_Exception_bk()

    def func_test_pylayer_bk_return_none(self):

        class Layer_bk_none1(
                EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1, x2):
                return x1 + x2

            @staticmethod
            def backward(ctx, dy):
                return 1

        input1 = paddle.randn([2, 3]).astype("float64")
        input2 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = True
        input2.stop_gradient = False
        z = Layer_bk_none1.apply(input1, input2)

        with self.assertRaises(ValueError):
            z.mean().backward()

        class Layer_bk_none2(
                EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1, x2):
                return x1 * 2, x2 * 5

            @staticmethod
            def backward(ctx, *args):
                return 1, 1

        input1 = paddle.randn([2, 3]).astype("float64")
        input2 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = True
        input2.stop_gradient = False
        z = Layer_bk_none2.apply(input1, input2)
        z = z[0] + z[1]
        with self.assertRaises(ValueError):
            z.mean().backward()

    def test_pylayer_bk_return_none(self):
        with _test_eager_guard():
            self.func_test_pylayer_bk_return_none()
        self.func_test_pylayer_bk_return_none()

    def func_test_pylayer_inplace(self):

        class cus_tanh(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, dy):
                return dy

        class Layer(paddle.nn.Layer):

            def __init__(self):
                super(Layer, self).__init__()

            def forward(self, data):
                data = data**2
                z = paddle.tanh(data)
                z = cus_tanh.apply(data)
                return z.mean()

        for i in range(2):
            data = paddle.ones([2, 3], dtype="float64") / (i + 1)
            data.stop_gradient = False
            layer = Layer()
            z = layer(data)
            z.backward()
            self.assertTrue(data.grad is not None)

    def test_pylayer_inplace(self):
        with _test_eager_guard():
            self.func_test_pylayer_inplace()
        self.func_test_pylayer_inplace()

    def test_pylayer_inplace_backward_error(self):
        with _test_eager_guard():

            class cus_tanh(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer
                           ):

                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, dy):
                    return dy

            class Layer(paddle.nn.Layer):

                def __init__(self):
                    super(Layer, self).__init__()

                def forward(self, data):
                    var_b = data**2
                    var_c = var_b**2
                    z = cus_tanh.apply(var_b)
                    loss = paddle.nn.functional.relu(var_c)
                    return loss

            data = paddle.ones([2, 3], dtype="float64")
            data.stop_gradient = False
            layer = Layer()
            z = layer(data)
            with self.assertRaisesRegexp(
                    RuntimeError,
                    "received tensor_version:{} != wrapper_version_snapshot:{}".
                    format(1, 0)):
                z.backward()

    def test_pylayer_inplace_backward_success_1(self):
        with _test_eager_guard():

            class cus_tanh(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer
                           ):

                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, dy):
                    return dy

            class Layer(paddle.nn.Layer):

                def __init__(self):
                    super(Layer, self).__init__()

                def forward(self, data):
                    var_b = data**2
                    var_c = cus_tanh.apply(var_b)
                    var_d = var_c**2
                    loss = var_d.sum()
                    return loss

            for i in range(2):
                data = paddle.ones([2, 3], dtype="float64") / (i + 1)
                data.stop_gradient = False
                layer = Layer()
                z = layer(data)
                z.backward()
                self.assertTrue(data.grad is not None)

    def test_pylayer_inplace_backward_success_2(self):
        with _test_eager_guard():

            class cus_tanh(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer
                           ):

                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, dy):
                    return dy

            class Layer(paddle.nn.Layer):

                def __init__(self):
                    super(Layer, self).__init__()

                def forward(self, data):
                    var_b = data**2
                    var_c = cus_tanh.apply(var_b)
                    var_d = var_c + var_c
                    loss = var_d.sum()
                    return loss

            for i in range(2):
                data = paddle.ones([2, 3], dtype="float64") / (i + 1)
                data.stop_gradient = False
                layer = Layer()
                z = layer(data)
                z.backward()
                self.assertTrue(data.grad is not None)

    def func_test_pylayer_inplace_and_leaf_exception(self):

        class cus_pylayer_op(
                EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, dy):
                return dy

        class Layer(paddle.nn.Layer):

            def __init__(self):
                super(Layer, self).__init__()

            def forward(self, data):
                z = cus_pylayer_op.apply(data)
                return z.mean()

        for i in range(2):
            data = paddle.ones([2, 3], dtype="float64") / (i + 1)
            data.stop_gradient = False
            layer = Layer()

            with self.assertRaises(ValueError):
                z = layer(data)

    def test_pylayer_inplace_and_leaf_exception(self):
        with _test_eager_guard():
            self.func_test_pylayer_inplace_and_leaf_exception()
        self.func_test_pylayer_inplace_and_leaf_exception()

    def func_test_backward_in_backward(self):

        class cus_tanh(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x):
                temp = x.detach()
                ctx.inputs = temp
                return x.mean()

            @staticmethod
            def backward(ctx, dy):
                with paddle.set_grad_enabled(True):
                    temp = ctx.inputs
                    temp.stop_gradient = False
                    z = paddle.tanh(temp)
                    z.backward()
                    self.assertTrue(temp.grad is not None)
                    return paddle.to_tensor(temp.grad)

        for i in range(2):
            data = paddle.ones([2, 3], dtype="float32") / (i + 1)
            data.stop_gradient = False
            data = paddle.nn.functional.relu(data)
            z = paddle.tanh(data)
            z = cus_tanh.apply(data)

    def test_backward_in_backward(self):
        with _test_eager_guard():
            self.func_test_backward_in_backward()
        self.func_test_backward_in_backward()

    def func_test_return_to_tensor(self):

        class Tanh(EagerPyLayer if in_dygraph_mode() else LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1):
                y1 = paddle.tanh(x1)
                ctx.save_for_backward(y1)
                tensor_1 = paddle.to_tensor([1, 2], dtype='float32')
                return y1, 5, None, "helloworld", tensor_1

            @staticmethod
            def backward(ctx, dy1, dy2):
                y1, = ctx.saved_tensor()
                re1 = dy1 * (1 - paddle.square(y1))
                return dy1

        input1 = paddle.randn([2, 3]).astype("float32")
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        z, number, none_item, string_item, tensor1 = Tanh.apply(x1=input1)
        z.mean().backward()

    def test_return_to_tensor(self):
        with _test_eager_guard():
            self.func_test_return_to_tensor()
        self.func_test_return_to_tensor()

    def test_materialize_grads(self):
        with _test_eager_guard():

            class Tanh(EagerPyLayer):

                @staticmethod
                def forward(ctx, x):
                    ctx.mark_not_inplace(x)
                    return x, x + x

                @staticmethod
                def backward(ctx, grad, grad2):
                    self.assertEqual(grad2, paddle.zeros([1]))
                    return grad

            x = paddle.ones([1], dtype="float64")
            x.stop_gradient = False
            Tanh.apply(x)[0].backward()

    def test_dont_materialize_grads(self):
        with _test_eager_guard():

            class Tanh(EagerPyLayer):

                @staticmethod
                def forward(ctx, x):
                    ctx.mark_not_inplace(x)
                    ctx.set_materialize_grads(False)
                    return x, x + x

                @staticmethod
                def backward(ctx, grad, grad2):
                    self.assertIsNone(grad2)
                    return grad

            x = paddle.ones([1], dtype="float64")
            x.stop_gradient = False
            Tanh.apply(x)[0].backward()

    def test_mark_non_differentiable(self):
        with _test_eager_guard():

            class Tanh(EagerPyLayer):

                @staticmethod
                def forward(ctx, x):
                    a = x + x
                    ctx.mark_non_differentiable(a)
                    return a

                @staticmethod
                def backward(ctx, grad):
                    self.assertTrue(False)  # should not be call
                    return paddle.ones([1], dtype="float64")

            x = paddle.ones([1], dtype="float64")
            x.stop_gradient = False
            y = Tanh.apply(x)
            y.sum().backward()

    def test_mark_non_differentiable2(self):
        with _test_eager_guard():

            class Tanh(EagerPyLayer):

                @staticmethod
                def forward(ctx, x):
                    a = x + x
                    b = x + x + x
                    ctx.mark_non_differentiable(a)
                    return a, b

                @staticmethod
                def backward(ctx, grad_a, grad_b):
                    self.assertEqual(grad_a, paddle.zeros([1]))
                    self.assertEqual(grad_b, paddle.ones([1], dtype="float64"))
                    return grad_b

            x = paddle.ones([1], dtype="float64")
            x.stop_gradient = False
            a, b = Tanh.apply(x)
            b.sum().backward()
            self.assertEqual(x.grad, paddle.ones([1], dtype="float64"))


class TestPyLayerReturnType(unittest.TestCase):

    def test_forward_args_fake_tensor(self):

        class Tanh(LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1):
                y1 = FakeTensor()
                return y1, x1

            @staticmethod
            def backward(ctx, dy1, dy2):
                return dy1

        input1 = FakeTensor()

        with self.assertRaises(ValueError):
            y1, y2 = Tanh.apply(input1)

    def test_forward_kwargs_fake_tensor(self):

        class Tanh(LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1):

                return x1

            @staticmethod
            def backward(ctx, dy1, dy2):
                return dy1

        input1 = FakeTensor()

        with self.assertRaises(ValueError):
            y = Tanh.apply(x1=input1)

    def test_forward_return_fake_tensor(self):

        class Tanh(LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1):

                return FakeTensor()

            @staticmethod
            def backward(ctx, dy1, dy2):
                return dy1

        input1 = paddle.randn([3, 2])

        with self.assertRaises(ValueError):
            y = Tanh.apply(x1=input1)

    def test_forward_return_fake_tensor_tuple(self):

        class Tanh(LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1):

                return FakeTensor(), FakeTensor()

            @staticmethod
            def backward(ctx, dy1, dy2):
                return dy1

        input1 = paddle.randn([3, 2])

        with self.assertRaises(ValueError):
            y = Tanh.apply(x1=input1)

    def test_backward_return_fake_tensor_tuple(self):

        class Tanh(LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1, x2):
                return x1 + 1, x1 + 2

            @staticmethod
            def backward(ctx, dy1, dy2):

                return FakeTensor(), 2

        input1 = paddle.randn([3, 2])
        input1.stop_gradient = False
        y, _ = Tanh.apply(input1, 1 + input1)

        with self.assertRaises(ValueError):
            y.mean().backward()

    def test_backward_return_fake_tensor(self):

        class Tanh(LegacyPyLayer):

            @staticmethod
            def forward(ctx, x1):
                return x1 + 1, x1 + 2

            @staticmethod
            def backward(ctx, dy1, dy2):
                return FakeTensor()

        input1 = paddle.randn([3, 2])
        input1.stop_gradient = False
        y, _ = Tanh.apply(input1)

        with self.assertRaises(ValueError):
            y.mean().backward()


if __name__ == '__main__':
    unittest.main()
