#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

# Note:
# 0D Tensor indicates that the tensor's dimension is 0
# 0D Tensor's shape is always [], numel is 1
# which can be created by paddle.rand([])

import unittest

import numpy as np

import paddle
from paddle.framework import use_pir_api

binary_api_list = [
    {'func': paddle.add, 'cls_method': '__add__'},
    {'func': paddle.subtract, 'cls_method': '__sub__'},
    {'func': paddle.multiply, 'cls_method': '__mul__'},
    {'func': paddle.divide, 'cls_method': '__div__'},
    {'func': paddle.pow, 'cls_method': '__pow__'},
    {'func': paddle.equal, 'cls_method': '__eq__'},
    {'func': paddle.not_equal, 'cls_method': '__ne__'},
    {'func': paddle.greater_equal, 'cls_method': '__ge__'},
    {'func': paddle.greater_than, 'cls_method': '__gt__'},
    {'func': paddle.less_equal, 'cls_method': '__le__'},
    {'func': paddle.less_than, 'cls_method': '__lt__'},
    {'func': paddle.remainder, 'cls_method': '__mod__'},
    paddle.mod,
    paddle.floor_mod,
    paddle.logical_and,
    paddle.logical_or,
    paddle.logical_xor,
    paddle.maximum,
    paddle.minimum,
    paddle.fmax,
    paddle.fmin,
    paddle.complex,
    paddle.kron,
    paddle.logaddexp,
    paddle.nextafter,
    paddle.ldexp,
    paddle.polar,
    paddle.heaviside,
]

binary_int_api_list = [
    paddle.bitwise_and,
    paddle.bitwise_or,
    paddle.bitwise_xor,
    paddle.gcd,
    paddle.lcm,
]


inplace_binary_api_list = [
    paddle.tensor.add_,
    paddle.tensor.subtract_,
    paddle.tensor.multiply_,
    paddle.tensor.remainder_,
    paddle.tensor.remainder_,
]


# Use to test zero-dim of binary API
class TestBinaryAPI(unittest.TestCase):
    def test_dygraph_binary(self):
        paddle.disable_static()
        for api in binary_api_list:
            # 1) x is 0D, y is 0D
            x = paddle.rand([])
            y = paddle.rand([])
            x.stop_gradient = False
            y.stop_gradient = False
            if isinstance(api, dict):
                out = api['func'](x, y)
                out_cls = getattr(paddle.Tensor, api['cls_method'])(x, y)
                np.testing.assert_array_equal(out_cls.numpy(), out.numpy())
            else:
                out = api(x, y)

            out.retain_grads()
            out.backward()

            self.assertEqual(x.shape, [])
            self.assertEqual(y.shape, [])
            self.assertEqual(out.shape, [])
            if x.grad is not None:
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(y.grad.shape, [])
                self.assertEqual(out.grad.shape, [])

            # 2) x is ND, y is 0D
            x = paddle.rand([2, 3, 4])
            y = paddle.rand([])
            x.stop_gradient = False
            y.stop_gradient = False
            if isinstance(api, dict):
                out = api['func'](x, y)
                out_cls = getattr(paddle.Tensor, api['cls_method'])(x, y)
                np.testing.assert_array_equal(out_cls.numpy(), out.numpy())
            else:
                out = api(x, y)

            out.retain_grads()
            out.backward()

            self.assertEqual(x.shape, [2, 3, 4])
            self.assertEqual(y.shape, [])
            self.assertEqual(out.shape, [2, 3, 4])
            if x.grad is not None:
                self.assertEqual(x.grad.shape, [2, 3, 4])
                self.assertEqual(y.grad.shape, [])
                self.assertEqual(out.grad.shape, [2, 3, 4])

            # 3) x is 0D , y is ND
            x = paddle.rand([])
            y = paddle.rand([2, 3, 4])
            x.stop_gradient = False
            y.stop_gradient = False
            if isinstance(api, dict):
                out = api['func'](x, y)
                out_cls = getattr(paddle.Tensor, api['cls_method'])(x, y)
                np.testing.assert_array_equal(out_cls.numpy(), out.numpy())
            else:
                out = api(x, y)

            out.retain_grads()
            out.backward()

            self.assertEqual(x.shape, [])
            self.assertEqual(y.shape, [2, 3, 4])
            self.assertEqual(out.shape, [2, 3, 4])
            if x.grad is not None:
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(y.grad.shape, [2, 3, 4])
                self.assertEqual(out.grad.shape, [2, 3, 4])

            # 4) x is 0D , y is scalar
            x = paddle.rand([])
            x.stop_gradient = False
            y = 0.5
            if isinstance(api, dict):
                out = getattr(paddle.Tensor, api['cls_method'])(x, y)

                out.retain_grads()
                out.backward()

                self.assertEqual(x.shape, [])
                self.assertEqual(out.shape, [])
                if x.grad is not None:
                    self.assertEqual(x.grad.shape, [])
                    self.assertEqual(out.grad.shape, [])

        for api in binary_int_api_list:
            # 1) x is 0D, y is 0D
            x_np = np.random.randint(-10, 10, [])
            y_np = np.random.randint(-10, 10, [])
            out_np = eval(f'np.{api.__name__}(x_np, y_np)')

            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            out = api(x, y)

            self.assertEqual(out.shape, [])
            np.testing.assert_array_equal(out.numpy(), out_np)

            # 2) x is ND, y is 0D
            x_np = np.random.randint(-10, 10, [3, 5])
            y_np = np.random.randint(-10, 10, [])
            out_np = eval(f'np.{api.__name__}(x_np, y_np)')

            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            out = api(x, y)

            self.assertEqual(out.shape, [3, 5])
            np.testing.assert_array_equal(out.numpy(), out_np)

            # 3) x is 0D , y is ND
            x_np = np.random.randint(-10, 10, [])
            y_np = np.random.randint(-10, 10, [3, 5])
            out_np = eval(f'np.{api.__name__}(x_np, y_np)')

            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            out = api(x, y)

            self.assertEqual(out.shape, [3, 5])
            np.testing.assert_array_equal(out.numpy(), out_np)

        for api in inplace_binary_api_list:
            with paddle.no_grad():
                x = paddle.rand([])
                y = paddle.rand([])
                out = api(x, y)
                self.assertEqual(x.shape, [])
                self.assertEqual(out.shape, [])

                x = paddle.rand([3, 5])
                y = paddle.rand([])
                out = api(x, y)
                self.assertEqual(x.shape, [3, 5])
                self.assertEqual(out.shape, [3, 5])

        paddle.enable_static()

    def assertShapeEqual(self, out, target_tuple):
        if not use_pir_api():
            out_shape = list(out.shape)
        else:
            out_shape = out.shape
        self.assertEqual(out_shape, target_tuple)

    def test_static_binary_0D_0D(self):
        paddle.enable_static()
        for api in binary_api_list:
            main_prog = paddle.static.Program()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                # 1) x is 0D, y is 0D
                x = paddle.rand([])
                y = paddle.rand([])
                x.stop_gradient = False
                y.stop_gradient = False
                if isinstance(api, dict):
                    out = api['func'](x, y)
                    out_cls = getattr(
                        (
                            paddle.pir.Value
                            if use_pir_api()
                            else paddle.static.Variable
                        ),
                        api['cls_method'],
                    )(x, y)
                    self.assertEqual(out.shape, out_cls.shape)
                else:
                    out = api(x, y)
                grad_list = paddle.static.append_backward(
                    out, parameter_list=[x, y, out]
                )

                self.assertShapeEqual(x, [])
                self.assertShapeEqual(y, [])
                self.assertShapeEqual(out, [])

                if len(grad_list) != 0 and grad_list[0][1] is not None:
                    # x_grad
                    self.assertShapeEqual(grad_list[0][1], [])
                    # y_grad
                    self.assertShapeEqual(grad_list[1][1], [])
                    # out_grad
                    self.assertShapeEqual(grad_list[2][1], [])

        paddle.disable_static()

    def test_static_binary_0D_ND(self):
        paddle.enable_static()
        for api in binary_api_list:
            main_prog = paddle.static.Program()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                # 2) x is 0D, y is ND
                x = paddle.rand([])
                y = paddle.rand([2, 3, 4])
                x.stop_gradient = False
                y.stop_gradient = False
                if isinstance(api, dict):
                    out = api['func'](x, y)
                    out_cls = getattr(
                        (
                            paddle.pir.Value
                            if use_pir_api()
                            else paddle.static.Variable
                        ),
                        api['cls_method'],
                    )(x, y)
                    self.assertEqual(out.shape, out_cls.shape)
                else:
                    out = api(x, y)
                grad_list = paddle.static.append_backward(
                    out, parameter_list=[x, y, out]
                )

                self.assertShapeEqual(x, [])
                self.assertShapeEqual(y, [2, 3, 4])
                self.assertShapeEqual(out, [2, 3, 4])

                if len(grad_list) != 0 and grad_list[0][1] is not None:
                    # x_grad
                    self.assertShapeEqual(grad_list[0][1], [])
                    # y_grad
                    self.assertShapeEqual(grad_list[1][1], [2, 3, 4])
                    # out_grad
                    self.assertShapeEqual(grad_list[2][1], [2, 3, 4])
        paddle.disable_static()

    def test_static_binary_ND_0D(self):
        paddle.enable_static()
        for api in binary_api_list:
            main_prog = paddle.static.Program()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                # 3) x is ND, y is 0d
                x = paddle.rand([2, 3, 4])
                y = paddle.rand([])
                x.stop_gradient = False
                y.stop_gradient = False
                if isinstance(api, dict):
                    out = api['func'](x, y)
                    out_cls = getattr(
                        (
                            paddle.pir.Value
                            if use_pir_api()
                            else paddle.static.Variable
                        ),
                        api['cls_method'],
                    )(x, y)
                    self.assertEqual(out.shape, out_cls.shape)
                else:
                    out = api(x, y)
                grad_list = paddle.static.append_backward(
                    out, parameter_list=[x, y, out]
                )

                self.assertShapeEqual(x, [2, 3, 4])
                self.assertShapeEqual(y, [])
                self.assertShapeEqual(out, [2, 3, 4])

                if len(grad_list) != 0 and grad_list[0][1] is not None:
                    # x_grad
                    self.assertShapeEqual(grad_list[0][1], [2, 3, 4])
                    # y_grad
                    self.assertShapeEqual(grad_list[1][1], [])
                    # out_grad
                    self.assertShapeEqual(grad_list[2][1], [2, 3, 4])
        paddle.disable_static()

    def test_static_binary_0D_scalar(self):
        paddle.enable_static()
        for api in binary_api_list:
            main_prog = paddle.static.Program()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                # 4) x is 0D , y is scalar
                x = paddle.rand([])
                x.stop_gradient = False
                y = 0.5
                if isinstance(api, dict):
                    out = getattr(
                        (
                            paddle.pir.Value
                            if use_pir_api()
                            else paddle.static.Variable
                        ),
                        api['cls_method'],
                    )(x, y)
                    grad_list = paddle.static.append_backward(
                        out, parameter_list=[x, out]
                    )
                    self.assertShapeEqual(x, [])
                    self.assertShapeEqual(out, [])

                    if len(grad_list) != 0 and grad_list[0][1] is not None:
                        # x_grad
                        self.assertShapeEqual(grad_list[0][1], [])
                        # out_grad
                        self.assertShapeEqual(grad_list[1][1], [])
        paddle.disable_static()

    def test_static_binary_int_api(self):
        paddle.enable_static()
        for api in binary_int_api_list:
            main_prog = paddle.static.Program()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                # 1) x is 0D, y is 0D
                x = paddle.randint(-10, 10, [])
                y = paddle.randint(-10, 10, [])
                out = api(x, y)
                self.assertShapeEqual(out, [])

                # 2) x is ND , y is 0D
                x = paddle.randint(-10, 10, [3, 5])
                y = paddle.randint(-10, 10, [])
                out = api(x, y)
                self.assertShapeEqual(out, [3, 5])

                # 3) x is 0D , y is ND
                x = paddle.randint(-10, 10, [])
                y = paddle.randint(-10, 10, [3, 5])
                out = api(x, y)
                self.assertShapeEqual(out, [3, 5])

        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
