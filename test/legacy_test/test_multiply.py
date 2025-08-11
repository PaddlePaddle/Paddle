# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import static, tensor
from paddle.base.framework import in_pir_mode


class TestMultiplyApi(unittest.TestCase):
    def _run_static_graph_case(self, x_data, y_data):
        with static.program_guard(static.Program(), static.Program()):
            paddle.enable_static()
            x = paddle.static.data(
                name='x', shape=x_data.shape, dtype=x_data.dtype
            )
            y = paddle.static.data(
                name='y', shape=y_data.shape, dtype=y_data.dtype
            )
            res = tensor.multiply(x, y)

            place = get_device_place()
            exe = paddle.static.Executor(place)
            outs = exe.run(
                paddle.static.default_main_program(),
                feed={'x': x_data, 'y': y_data},
                fetch_list=[res],
            )
            res = outs[0]
            return res

    def _run_dynamic_graph_case(self, x_data, y_data):
        paddle.disable_static()
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        res = paddle.multiply(x, y)
        return res.numpy()

    def test_multiply(self):
        np.random.seed(7)

        # test static computation graph: 1-d array
        x_data = np.random.rand(200)
        y_data = np.random.rand(200)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.multiply(x_data, y_data), rtol=1e-05)

        # test static computation graph: 2-d array
        x_data = np.random.rand(2, 500)
        y_data = np.random.rand(2, 500)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.multiply(x_data, y_data), rtol=1e-05)

        # test static computation graph: broadcast
        x_data = np.random.rand(2, 500)
        y_data = np.random.rand(500)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.multiply(x_data, y_data), rtol=1e-05)

        # test static computation graph: boolean
        x_data = np.random.choice([True, False], size=[200])
        y_data = np.random.choice([True, False], size=[200])
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.multiply(x_data, y_data), rtol=1e-05)

        # test dynamic computation graph: 1-d array
        x_data = np.random.rand(200)
        y_data = np.random.rand(200)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.multiply(x_data, y_data), rtol=1e-05)

        # test dynamic computation graph: 2-d array
        x_data = np.random.rand(20, 50)
        y_data = np.random.rand(20, 50)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.multiply(x_data, y_data), rtol=1e-05)

        # test dynamic computation graph: broadcast
        x_data = np.random.rand(2, 500)
        y_data = np.random.rand(500)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.multiply(x_data, y_data), rtol=1e-05)

        # test dynamic computation graph: boolean
        x_data = np.random.choice([True, False], size=[200])
        y_data = np.random.choice([True, False], size=[200])
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.multiply(x_data, y_data), rtol=1e-05)


class TestMultiplyError(unittest.TestCase):
    def test_errors(self):
        # test static computation graph: dtype can not be int8
        paddle.enable_static()
        with static.program_guard(static.Program(), static.Program()):
            x = paddle.static.data(name='x', shape=[100], dtype=np.int8)
            y = paddle.static.data(name='y', shape=[100], dtype=np.int8)
            if not in_pir_mode():
                self.assertRaises(TypeError, tensor.multiply, x, y)

        # test static computation graph: inputs must be broadcastable
        with static.program_guard(static.Program(), static.Program()):
            x = paddle.static.data(name='x', shape=[20, 50], dtype=np.float64)
            y = paddle.static.data(name='y', shape=[20], dtype=np.float64)
            self.assertRaises(ValueError, tensor.multiply, x, y)

        np.random.seed(7)
        # test dynamic computation graph: dtype can not be int8
        paddle.disable_static()
        x_data = np.random.randn(200).astype(np.int8)
        y_data = np.random.randn(200).astype(np.int8)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        self.assertRaises(RuntimeError, paddle.multiply, x, y)

        # test dynamic computation graph: inputs must be broadcastable
        x_data = np.random.rand(200, 5)
        y_data = np.random.rand(200)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        self.assertRaises(ValueError, paddle.multiply, x, y)

        # test dynamic computation graph: inputs must be broadcastable(python)
        x_data = np.random.rand(200, 5)
        y_data = np.random.rand(200)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        self.assertRaises(ValueError, paddle.multiply, x, y)

        # test dynamic computation graph: dtype must be same
        x_data = np.random.randn(200).astype(np.int64)
        y_data = np.random.randn(200).astype(np.float64)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        self.assertRaises(TypeError, paddle.multiply, x, y)

        # test dynamic computation graph: dtype must be Tensor type
        x_data = np.random.randn(200).astype(np.int64)
        y_data = np.random.randn(200).astype(np.float64)
        y = paddle.to_tensor(y_data)
        self.assertRaises(ValueError, paddle.multiply, x_data, y)

        # test dynamic computation graph: dtype must be Tensor type
        x_data = np.random.randn(200).astype(np.int64)
        y_data = np.random.randn(200).astype(np.float64)
        x = paddle.to_tensor(x_data)
        self.assertRaises(ValueError, paddle.multiply, x, y_data)

        # test dynamic computation graph: dtype must be Tensor type
        x_data = np.random.randn(200).astype(np.float32)
        y_data = np.random.randn(200).astype(np.float32)
        x = paddle.to_tensor(x_data)
        self.assertRaises(ValueError, paddle.multiply, x, y_data)

        # test dynamic computation graph: dtype must be Tensor type
        x_data = np.random.randn(200).astype(np.float32)
        y_data = np.random.randn(200).astype(np.float32)
        x = paddle.to_tensor(x_data)
        self.assertRaises(ValueError, paddle.multiply, x_data, y)

        # test dynamic computation graph: dtype must be Tensor type
        x_data = np.random.randn(200).astype(np.float32)
        y_data = np.random.randn(200).astype(np.float32)
        self.assertRaises(ValueError, paddle.multiply, x_data, y_data)


class TestMultiplyInplaceApi(TestMultiplyApi):
    def _run_static_graph_case(self, x_data, y_data):
        with static.program_guard(static.Program(), static.Program()):
            paddle.enable_static()
            x = paddle.static.data(
                name='x', shape=x_data.shape, dtype=x_data.dtype
            )
            y = paddle.static.data(
                name='y', shape=y_data.shape, dtype=y_data.dtype
            )
            res = x.multiply_(y)

            place = get_device_place()
            exe = paddle.static.Executor(place)
            outs = exe.run(
                paddle.static.default_main_program(),
                feed={'x': x_data, 'y': y_data},
                fetch_list=[res],
            )
            res = outs[0]
            return res

    def _run_dynamic_graph_case(self, x_data, y_data):
        paddle.disable_static()
        with paddle.no_grad():
            x = paddle.to_tensor(x_data)
            y = paddle.to_tensor(y_data)
            x.multiply_(y)
        return x.numpy()


class TestMultiplyInplaceError(unittest.TestCase):
    def test_errors(self):
        paddle.disable_static()
        # test dynamic computation graph: inputs must be broadcastable
        x_data = np.random.rand(3, 4)
        y_data = np.random.rand(2, 3, 4)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)

        def multiply_shape_error():
            with paddle.no_grad():
                x.multiply_(y)

        self.assertRaises(ValueError, multiply_shape_error)
        paddle.enable_static()


class TestMultiplyApiZeroSize(TestMultiplyApi):

    # only support the 0 size tensor
    def _test_grad(self, x_data, y_data):
        paddle.disable_static()
        x = paddle.to_tensor(x_data, stop_gradient=False)
        y = paddle.to_tensor(y_data, stop_gradient=False)
        z = paddle.multiply(x, y)
        loss = z.sum()
        loss.backward()
        np.testing.assert_allclose(
            x.grad.numpy(), np.zeros(self.x_shape).astype('float32'), rtol=1e-05
        )
        np.testing.assert_allclose(
            y.grad.numpy(), np.zeros(self.y_shape).astype('float32'), rtol=1e-05
        )

    def init_shapes(self):
        self.x_shape = [0, 4]
        self.y_shape = [0, 1]

    def test_multiply(self):
        np.random.seed(7)
        self.init_shapes()

        # test static computation graph
        x_data = np.random.rand(*(self.x_shape)).astype('float32')
        y_data = np.random.rand(*(self.y_shape)).astype('float32')
        expected_res = np.multiply(x_data, y_data)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, expected_res, rtol=1e-05)
        # test dynamic computation graph
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, expected_res, rtol=1e-05)
        # test gradient
        self._test_grad(x_data, y_data)


class TestMultiplyApiZeroSize1(TestMultiplyApiZeroSize):
    def init_shapes(self):
        self.x_shape = [6, 0]
        self.y_shape = [6, 0]


class TestMultiplyApiZeroSize2(TestMultiplyApiZeroSize):
    def init_shapes(self):
        self.x_shape = [1, 8]
        self.y_shape = [0, 1]


class TestMultiplyApiZeroSize3(TestMultiplyApiZeroSize):
    def init_shapes(self):
        self.x_shape = [5, 0]
        self.y_shape = [5, 1]


class TestMultiplyApiBF16(unittest.TestCase):
    # Now only check the successful run of multiply with bfloat16 and backward.
    def setUp(self):
        paddle.device.set_device('cpu')

    def test_multiply(self):
        self.x_shape = [1, 1024, 32, 128]
        self.y_shape = [1, 1024, 1, 128]
        x = paddle.rand(self.x_shape, dtype='bfloat16')
        x.stop_gradient = False
        y = paddle.rand(self.y_shape, dtype='bfloat16')
        y.stop_gradient = False
        res = paddle.multiply(x, y)
        loss = res.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.dtype == paddle.bfloat16
        assert y.grad is not None
        assert y.grad.dtype == paddle.bfloat16


if __name__ == '__main__':
    unittest.main()
