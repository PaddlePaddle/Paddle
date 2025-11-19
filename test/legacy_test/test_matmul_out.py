# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base


def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        if X.ndim == 1:
            X = X.reshape((X.size,))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = list(range(len(X.shape)))
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_Y:
        if Y.ndim == 1:
            Y = Y.reshape((Y.size,))
        else:
            dim = list(range(len(Y.shape)))
            dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
            Y = np.transpose(Y, tuple(dim))

    Out = np.matmul(X, Y)
    return Out


class TestMatmulOutAndParamDecorator(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_shape = [3, 4]
        self.y_shape = [4, 3]
        self.x_np = np.random.rand(*self.x_shape).astype(np.float32)
        self.y_np = np.random.rand(*self.y_shape).astype(np.float32)

        self.apis = [paddle.matmul, paddle.linalg.matmul]
        self.test_types = [
            # "decorator1",
            # "decorator2",
            "out",
            # "out_decorator",
        ]

    def do_test(self, api, test_type):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)
        out = paddle.empty((3, 3), dtype='float32')
        out.stop_gradient = False

        if test_type == 'raw':
            result = api(x, y)
            result.mean().backward()
            return result, x.grad, y.grad
        elif test_type == 'decorator1':
            result = api(x, y)
            result.mean().backward()
            return result, x.grad, y.grad
        elif test_type == 'decorator2':
            result = api(input=x, other=y)
            result.mean().backward()
            return result, x.grad, y.grad
        elif test_type == 'out':
            api(x, y, out=out)
            out.mean().backward()
            return out, x.grad, y.grad
        elif test_type == 'out_decorator':
            api(input=x, other=y, out=out)
            out.mean().backward()
            return out, x.grad, y.grad
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def test_matmul_out(self):
        out_std, grad_std, y_grad_std = self.do_test(paddle.matmul, 'raw')
        for test_type in self.test_types:
            out, grad, y_grad = self.do_test(paddle.matmul, test_type)
            np.testing.assert_allclose(out.numpy(), out_std.numpy(), rtol=1e-20)
            np.testing.assert_allclose(
                grad.numpy(), grad_std.numpy(), rtol=1e-20
            )
            np.testing.assert_allclose(
                y_grad.numpy(), y_grad_std.numpy(), rtol=1e-20
            )


class TestMatMulAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.x_shape = [5, 6]
        self.y_shape = [6, 4]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x_input = np.random.randint(0, 8, self.x_shape).astype(
            self.dtype
        )
        self.np_y_input = np.random.randint(3, 9, self.y_shape).astype(
            self.dtype
        )

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x_input)
        y = paddle.to_tensor(self.np_y_input)
        paddle_dygraph_out = []
        # Position args (args)
        out1 = paddle.matmul(x, y)
        paddle_dygraph_out.append(out1)
        # Keywords args (kwargs) for paddle
        out2 = paddle.matmul(x=x, y=y)
        paddle_dygraph_out.append(out2)
        # Keywords args for torch
        out3 = paddle.matmul(input=x, other=y)
        paddle_dygraph_out.append(out3)
        # Combined args and kwargs
        out4 = paddle.matmul(x, other=y)
        paddle_dygraph_out.append(out4)
        # Tensor method args
        out5 = x.matmul(y)
        paddle_dygraph_out.append(out5)
        # Tensor method kwargs
        out6 = x.matmul(other=y)
        paddle_dygraph_out.append(out6)
        # Test out
        out7 = paddle.empty([])
        paddle.matmul(x, other=y, out=out7)
        paddle_dygraph_out.append(out7)
        # Numpy reference  out
        ref_out = reference_matmul(self.np_x_input, self.np_y_input)
        # Check
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.x_shape, dtype=self.dtype
            )
            y = paddle.static.data(
                name="y", shape=self.y_shape, dtype=self.dtype
            )
            # Position args (args)
            out1 = paddle.matmul(x, y)
            # Keywords args (kwargs) for paddle
            out2 = paddle.matmul(x=x, y=y)
            # Keywords args for torch
            out3 = paddle.matmul(input=x, other=y)
            # Combined args and kwargs
            out4 = paddle.matmul(x, other=y)
            # Tensor method args
            out5 = x.matmul(y)
            # Tensor method kwargs
            out6 = x.matmul(other=y)
            exe = base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x_input, "y": self.np_y_input},
                fetch_list=[out1, out2, out3, out4, out5, out6],
            )
            ref_out = reference_matmul(self.np_x_input, self.np_y_input)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


if __name__ == "__main__":
    unittest.main()
