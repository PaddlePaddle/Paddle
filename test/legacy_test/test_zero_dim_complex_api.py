#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

unary_apis_with_complex_input = [
    paddle.real,
    paddle.imag,
    paddle.angle,
    paddle.conj,
]


class AssertShapeEqualMixin:
    def assertShapeEqual(self, out, target_tuple):
        if not paddle.framework.in_pir_mode():
            out_shape = list(out.shape)
        else:
            out_shape = out.shape
        self.assertEqual(out_shape, target_tuple)


class TestUnaryElementwiseAPIWithComplexInput(unittest.TestCase):
    def test_dygraph_unary(self):
        paddle.disable_static()
        for api in unary_apis_with_complex_input:
            x = paddle.rand([]) + 1j * paddle.rand([])
            x.stop_gradient = False
            x.retain_grads()
            out = api(x)
            out.retain_grads()
            out.backward()

            self.assertEqual(x.shape, [])
            self.assertEqual(out.shape, [])
            if x.grad is not None:
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(out.grad.shape, [])

        paddle.enable_static()

    def test_static_unary(self):
        paddle.enable_static()
        for api in unary_apis_with_complex_input:
            main_prog = paddle.static.Program()
            block = main_prog.global_block()
            exe = paddle.static.Executor()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                x = paddle.complex(paddle.rand([]), paddle.rand([]))
                x.stop_gradient = False
                out = api(x)

                [(_, x_grad), (_, out_grad)] = paddle.static.append_backward(
                    out, parameter_list=[x, out]
                )

                res = exe.run(main_prog, fetch_list=[x, out, x_grad, out_grad])
                for item in res:
                    self.assertEqual(item.shape, ())

        paddle.disable_static()


class TestAsReal(unittest.TestCase, AssertShapeEqualMixin):
    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.rand([]) + 1j * paddle.rand([])
        x.stop_gradient = False
        x.retain_grads()
        out = paddle.as_real(x)
        out.retain_grads()
        out.backward()

        self.assertEqual(x.shape, [])
        self.assertEqual(out.shape, [2])
        if x.grad is not None:
            self.assertEqual(x.grad.shape, [])
            self.assertEqual(out.grad.shape, [2])

        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()

        main_prog = paddle.static.Program()
        block = main_prog.global_block()
        exe = paddle.static.Executor()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            x = paddle.complex(paddle.rand([]), paddle.rand([]))
            x.stop_gradient = False
            out = paddle.as_real(x)
            self.assertShapeEqual(x, [])
            self.assertShapeEqual(out, [2])
            [(_, x_grad), (_, out_grad)] = paddle.static.append_backward(
                out.sum(), parameter_list=[x, out]
            )

            res = exe.run(main_prog, fetch_list=[x, out, x_grad, out_grad])
            self.assertEqual(res[0].shape, ())
            self.assertEqual(res[1].shape, (2,))
            self.assertEqual(res[2].shape, ())
            self.assertEqual(res[3].shape, (2,))

        paddle.disable_static()


class TestAsComplex(unittest.TestCase, AssertShapeEqualMixin):
    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.rand([2])
        x.stop_gradient = False
        x.retain_grads()
        out = paddle.as_complex(x)
        out.retain_grads()
        out.backward()

        self.assertEqual(x.shape, [2])
        self.assertEqual(out.shape, [])
        if x.grad is not None:
            self.assertEqual(x.grad.shape, [2])
            self.assertEqual(out.grad.shape, [])

        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        block = main_prog.global_block()
        exe = paddle.static.Executor()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            x = paddle.rand([2])
            x.stop_gradient = False
            out = paddle.as_complex(x)
            self.assertShapeEqual(x, [2])
            self.assertShapeEqual(out, [])
            [(_, x_grad), (_, out_grad)] = paddle.static.append_backward(
                out.sum(), parameter_list=[x, out]
            )

            res = exe.run(main_prog, fetch_list=[x, out, x_grad, out_grad])
            self.assertEqual(res[0].shape, (2,))
            self.assertEqual(res[1].shape, ())
            self.assertEqual(res[2].shape, (2,))
            self.assertEqual(res[3].shape, ())

        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
