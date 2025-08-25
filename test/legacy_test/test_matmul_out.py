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


if __name__ == "__main__":
    unittest.main()
