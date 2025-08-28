#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_device_place

import paddle
from paddle import base

paddle.enable_static()


class TestMaxMinAmaxAminAPI(unittest.TestCase):
    def setUp(self):
        self.init_case()
        self.cal_np_out_and_gradient()
        self.place = get_device_place()

    def init_case(self):
        self.x_np = np.array([[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = 0
        self.keepdim = False

    # If there are multiple minimum or maximum elements, max/min/amax/amin is non-derivable,
    # its gradient check is not supported by unittest framework,
    # thus we calculate the gradient by numpy function.
    def cal_np_out_and_gradient(self):
        def _cal_np_out_and_gradient(func):
            if func == 'amax':
                out = np.amax(self.x_np, axis=self.axis, keepdims=self.keepdim)
            elif func == 'amin':
                out = np.amin(self.x_np, axis=self.axis, keepdims=self.keepdim)
            elif func == 'max':
                out = np.max(self.x_np, axis=self.axis, keepdims=self.keepdim)
            elif func == 'min':
                out = np.min(self.x_np, axis=self.axis, keepdims=self.keepdim)
            else:
                print(
                    'This unittest only test amax/amin/max/min, but now is',
                    func,
                )
            self.np_out[func] = out
            grad = np.zeros(self.shape)
            out_b = np.broadcast_to(out.view(), self.shape)
            grad[self.x_np == out_b] = 1
            grad_sum = grad.sum(self.axis).reshape(out.shape)
            grad_b = np.broadcast_to(grad_sum, self.shape)
            grad /= grad_sum

            self.np_grad[func] = grad

        self.np_out = {}
        self.np_grad = {}
        _cal_np_out_and_gradient('amax')
        _cal_np_out_and_gradient('amin')
        _cal_np_out_and_gradient('max')
        _cal_np_out_and_gradient('min')

    def _choose_paddle_func(self, func, x):
        if func == 'amax':
            out = paddle.amax(x, self.axis, self.keepdim)
        elif func == 'amin':
            out = paddle.amin(x, self.axis, self.keepdim)
        elif func == 'max':
            out = paddle.max(x, self.axis, self.keepdim)
        elif func == 'min':
            out = paddle.min(x, self.axis, self.keepdim)
        else:
            print('This unittest only test amax/amin/max/min, but now is', func)
        return out

    # We check the output between paddle API and numpy in static graph.

    def test_static_graph(self):
        def _test_static_graph(func):
            paddle.enable_static()
            startup_program = base.Program()
            train_program = base.Program()
            with base.program_guard(startup_program, train_program):
                x = paddle.static.data(
                    name='input', dtype=self.dtype, shape=self.shape
                )
                x.stop_gradient = False
                out = self._choose_paddle_func(func, x)

                exe = base.Executor(self.place)
                res = exe.run(
                    feed={'input': self.x_np},
                    fetch_list=[out],
                )
                self.assertTrue((np.array(res[0]) == self.np_out[func]).all())
            paddle.disable_static()

        _test_static_graph('amax')
        _test_static_graph('amin')
        _test_static_graph('max')
        _test_static_graph('min')

    # As dygraph is easy to compute gradient, we check the gradient between
    # paddle API and numpy in dygraph.
    def test_dygraph(self):
        def _test_dygraph(func):
            paddle.disable_static()
            x = paddle.to_tensor(
                self.x_np, dtype=self.dtype, stop_gradient=False
            )
            out = self._choose_paddle_func(func, x)
            grad_tensor = paddle.ones_like(x)
            paddle.autograd.backward([out], [grad_tensor], True)

            np.testing.assert_allclose(
                self.np_out[func], out.numpy(), rtol=1e-05
            )
            np.testing.assert_allclose(self.np_grad[func], x.grad, rtol=1e-05)
            paddle.enable_static()

        _test_dygraph('amax')
        _test_dygraph('amin')
        _test_dygraph('max')
        _test_dygraph('min')

    # test two minimum or maximum elements


class TestMaxMinAmaxAminAPI_AxisWithOne1(TestMaxMinAmaxAminAPI):
    def init_case(self):
        self.x_np = np.random.randn(1, 5, 10).astype(np.float32)
        self.shape = [1, 5, 10]
        self.dtype = 'float32'
        self.axis = 0
        self.keepdim = False


class TestMaxMinAmaxAminAPI_AxisWithOne2(TestMaxMinAmaxAminAPI):
    def init_case(self):
        self.x_np = np.random.randn(1, 5, 10).astype(np.float32)
        self.shape = [1, 5, 10]
        self.dtype = 'float32'
        self.axis = 0
        self.keepdim = True


class TestMaxMinAmaxAminAPI_AxisWithOne3(TestMaxMinAmaxAminAPI):
    def init_case(self):
        self.x_np = np.random.randn(1, 1, 10).astype(np.float32)
        self.shape = [1, 1, 10]
        self.dtype = 'float32'
        self.axis = (0, 1)
        self.keepdim = False


class TestMaxMinAmaxAminAPI_ZeroDim(TestMaxMinAmaxAminAPI):
    def init_case(self):
        self.x_np = np.array(0.5)
        self.shape = []
        self.dtype = 'float64'
        self.axis = None
        self.keepdim = False


class TestMaxMinAmaxAminAPI2(TestMaxMinAmaxAminAPI):
    def init_case(self):
        self.x_np = np.array([[0.2, 0.3, 0.9, 0.9], [0.1, 0.1, 0.6, 0.7]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = None
        self.keepdim = False


# test different axis
class TestMaxMinAmaxAminAPI3(TestMaxMinAmaxAminAPI):
    def init_case(self):
        self.x_np = np.array([[0.2, 0.3, 0.9, 0.9], [0.1, 0.1, 0.6, 0.7]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = 0
        self.keepdim = False


# test keepdim = True
class TestMaxMinAmaxAminAPI4(TestMaxMinAmaxAminAPI):
    def init_case(self):
        self.x_np = np.array([[0.2, 0.3, 0.9, 0.9], [0.1, 0.1, 0.6, 0.7]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = 1
        self.keepdim = True


# test axis is tuple
class TestMaxMinAmaxAminAPI5(TestMaxMinAmaxAminAPI):
    def init_case(self):
        self.x_np = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(
            np.int32
        )
        self.shape = [2, 2, 2]
        self.dtype = 'int32'
        self.axis = (0, 1)
        self.keepdim = False


# test multiple minimum or maximum elements
class TestMaxMinAmaxAminAPI6(TestMaxMinAmaxAminAPI):
    def init_case(self):
        self.x_np = np.array([[0.2, 0.9, 0.9, 0.9], [0.9, 0.9, 0.2, 0.2]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = None
        self.keepdim = False


# test input grad when out is operated like multiply
class TestMaxMinAmaxAminAPI7(TestMaxMinAmaxAminAPI):
    def init_case(self):
        self.x_np = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(
            np.int32
        )
        self.shape = [2, 2, 2]
        self.dtype = 'int32'
        self.axis = (0, 1)
        self.keepdim = False

    # As dygraph is easy to compute gradient, we check the gradient between
    # paddle API and numpy in dygraph.
    def test_dygraph(self):
        def _test_dygraph(func):
            paddle.disable_static()
            x = paddle.to_tensor(
                self.x_np, dtype=self.dtype, stop_gradient=False
            )
            out = self._choose_paddle_func(func, x)
            loss = out * 2
            grad_tensor = paddle.ones_like(x)
            paddle.autograd.backward([loss], [grad_tensor], True)

            np.testing.assert_allclose(
                self.np_out[func], out.numpy(), rtol=1e-05
            )
            np.testing.assert_allclose(
                self.np_grad[func] * 2, x.grad, rtol=1e-05
            )
            paddle.enable_static()

        _test_dygraph('amax')
        _test_dygraph('amin')
        _test_dygraph('max')
        _test_dygraph('min')


class TestMaxMinAmaxAminAPI_ZeroSize(TestMaxMinAmaxAminAPI):
    def init_case(self):
        self.x_np = np.random.randn(1, 0, 10).astype(np.float32)
        self.shape = [1, 0, 10]
        self.dtype = 'float32'
        self.axis = 0
        self.keepdim = False


class TestMaxMinAmaxAminAPI_ZeroSize2(TestMaxMinAmaxAminAPI):
    def init_case(self):
        self.x_np = np.random.randn(1, 0, 10).astype(np.float32)
        self.shape = [1, 0, 10]
        self.dtype = 'float32'
        self.axis = -1
        self.keepdim = True


class TestAmaxAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        paddle_dygraph_out = []
        # Position args (args)
        out1 = paddle.amax(x, 1, True)
        paddle_dygraph_out.append(out1)
        # Key words args (kwargs) for paddle
        out2 = paddle.amax(x=x, axis=1, keepdim=True)
        paddle_dygraph_out.append(out2)
        # Key words args for torch
        out3 = paddle.amax(input=x, dim=1, keepdim=True)
        paddle_dygraph_out.append(out3)
        # Combined args and kwargs
        out4 = paddle.amax(x, dim=1, keepdim=True)
        paddle_dygraph_out.append(out4)
        # Tensor method args
        out5 = x.amax(1, True)
        paddle_dygraph_out.append(out5)
        # Tensor method kwargs
        out6 = x.amax(dim=1, keepdim=True)
        paddle_dygraph_out.append(out6)
        # Test out
        out7 = paddle.empty([])
        paddle.amax(x, 1, True, out=out7)
        paddle_dygraph_out.append(out7)
        # Test default value
        out8 = x.amax(1)
        # Numpy reference  out
        ref_out = np.amax(self.np_input, 1, keepdims=True)
        # Check
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy())
        ref_out = np.amax(self.np_input, 1, keepdims=False)
        np.testing.assert_allclose(ref_out, out8.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            # Position args (args)
            out1 = paddle.amax(x, 1, True)
            # Key words args (kwargs) for paddle
            out2 = paddle.amax(x=x, axis=1, keepdim=True)
            # Key words args for torch
            out3 = paddle.amax(input=x, dim=1, keepdim=True)
            # Combined args and kwargs
            out4 = paddle.amax(x, dim=1, keepdim=True)
            # Tensor method args
            out5 = x.amax(1, True)
            # Tensor method kwargs
            out6 = x.amax(dim=1, keepdim=True)
            # Do not support out in static
            # out7 = paddle.empty([])
            # paddle.all(x, 1, True, out=out7)
            # Test default value
            out8 = x.amax()
            exe = base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_input},
                fetch_list=[out1, out2, out3, out4, out5, out6, out8],
            )
            ref_out = np.amax(self.np_input, 1, keepdims=True)
            for out in fetches[:-1]:
                np.testing.assert_allclose(out, ref_out)
            ref_out = np.amax(self.np_input)
            np.testing.assert_allclose(*fetches[-1:], ref_out)


class TestAminAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        paddle_dygraph_out = []
        # Position args (args)
        out1 = paddle.amin(x, 1, True)
        paddle_dygraph_out.append(out1)
        # Key words args (kwargs) for paddle
        out2 = paddle.amin(x=x, axis=1, keepdim=True)
        paddle_dygraph_out.append(out2)
        # Key words args for torch
        out3 = paddle.amin(input=x, dim=1, keepdim=True)
        paddle_dygraph_out.append(out3)
        # Combined args and kwargs
        out4 = paddle.amin(x, dim=1, keepdim=True)
        paddle_dygraph_out.append(out4)
        # Tensor method args
        out5 = x.amin(1, True)
        paddle_dygraph_out.append(out5)
        # Tensor method kwargs
        out6 = x.amin(dim=1, keepdim=True)
        paddle_dygraph_out.append(out6)
        # Test out
        out7 = paddle.empty([])
        paddle.amin(x, 1, True, out=out7)
        paddle_dygraph_out.append(out7)
        # Test default value
        out8 = x.amin(1)
        # Numpy reference  out
        ref_out = np.amin(self.np_input, 1, keepdims=True)
        # Check
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy())
        ref_out = np.amin(self.np_input, 1, keepdims=False)
        np.testing.assert_allclose(ref_out, out8.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            # Position args (args)
            out1 = paddle.amin(x, 1, True)
            # Key words args (kwargs) for paddle
            out2 = paddle.amin(x=x, axis=1, keepdim=True)
            # Key words args for torch
            out3 = paddle.amin(input=x, dim=1, keepdim=True)
            # Combined args and kwargs
            out4 = paddle.amin(x, dim=1, keepdim=True)
            # Tensor method args
            out5 = x.amin(1, True)
            # Tensor method kwargs
            out6 = x.amin(dim=1, keepdim=True)
            # Do not support out in static
            # out7 = paddle.empty([])
            # paddle.all(x, 1, True, out=out7)
            # Test default value
            out8 = x.amin()
            exe = base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_input},
                fetch_list=[out1, out2, out3, out4, out5, out6, out8],
            )
            ref_out = np.amin(self.np_input, 1, keepdims=True)
            for out in fetches[:-1]:
                np.testing.assert_allclose(out, ref_out)
            ref_out = np.amin(self.np_input)
            np.testing.assert_allclose(*fetches[-1:], ref_out)


class TestAmaxAminOutAPI(unittest.TestCase):
    def _run_api(self, api, x, case):
        out_buf = paddle.zeros([], dtype=x.dtype)
        out_buf.stop_gradient = False
        if case == 'return':
            y = api(x)
        elif case == 'input_out':
            api(x, out=out_buf)
            y = out_buf
        elif case == 'both_return':
            y = api(x, out=out_buf)
        elif case == 'both_input_out':
            _ = api(x, out=out_buf)
            y = out_buf
        else:
            raise AssertionError
        return y

    def test_amax_out_in_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            np.array([[0.1, 0.9, 0.9, 0.9], [0.9, 0.9, 0.6, 0.7]]).astype(
                'float64'
            ),
            stop_gradient=False,
        )
        ref = paddle._C_ops.amax(x, None, False)
        outs = []
        grads = []
        for case in ['return', 'input_out', 'both_return', 'both_input_out']:
            y = self._run_api(paddle.amax, x, case)
            np.testing.assert_allclose(
                y.numpy(), ref.numpy(), rtol=1e-6, atol=1e-6
            )
            loss = (y * 2).mean()
            loss.backward()
            outs.append(y.numpy())
            grads.append(x.grad.numpy())
            x.clear_gradient()
        for i in range(1, 4):
            np.testing.assert_allclose(outs[0], outs[i], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(grads[0], grads[i], rtol=1e-6, atol=1e-6)
        paddle.enable_static()

    def test_amin_out_in_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            np.array([[0.2, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.7]]).astype(
                'float64'
            ),
            stop_gradient=False,
        )
        ref = paddle._C_ops.amin(x, None, False)
        outs = []
        grads = []
        for case in ['return', 'input_out', 'both_return', 'both_input_out']:
            y = self._run_api(paddle.amin, x, case)
            np.testing.assert_allclose(
                y.numpy(), ref.numpy(), rtol=1e-6, atol=1e-6
            )
            loss = (y * 2).mean()
            loss.backward()
            outs.append(y.numpy())
            grads.append(x.grad.numpy())
            x.clear_gradient()
        for i in range(1, 4):
            np.testing.assert_allclose(outs[0], outs[i], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(grads[0], grads[i], rtol=1e-6, atol=1e-6)
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
