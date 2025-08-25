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


class TestTakeAlongAxisOutAndParamDecorator(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.input_shape = [2, 3, 4]
        self.axis = 1
        self.indices = paddle.to_tensor([[[0]]], dtype='int64')
        self.out_shape = [2, 2, 4]
        self.x_np = np.random.rand(*self.input_shape).astype(np.float32)

        self.apis = [
            paddle.take_along_dim,
            paddle.take_along_axis,
        ]
        self.test_types = [
            "decorator1",
            "decorator2",
            "out",
            "out_decorator",
        ]

    def do_test(self, api, test_type):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        out = paddle.empty(self.out_shape, dtype='float32')
        out.stop_gradient = False

        if test_type == 'raw':
            out = api(x, self.indices, self.axis)
            out.mean().backward()
            return out, x.grad
        elif test_type == 'decorator1':
            out = api(x, dim=self.axis, indices=self.indices)
            out.mean().backward()
            return out, x.grad
        elif test_type == 'decorator2':
            out = api(dim=self.axis, indices=self.indices, input=x)
            out.mean().backward()
            return out, x.grad
        elif test_type == 'out':
            api(x, self.indices, self.axis, out=out)
            out.mean().backward()
            return out, x.grad
        elif test_type == 'out_decorator':
            api(input=x, indices=self.indices, dim=self.axis, out=out)
            out.mean().backward()
            return out, x.grad
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def test_take_along_dim(self):
        out_std, grad_std = self.do_test(paddle.take_along_dim, 'raw')
        for test_type in self.test_types:
            out, grad = self.do_test(paddle.take_along_dim, test_type)
            np.testing.assert_allclose(out.numpy(), out_std.numpy(), rtol=1e-20)
            np.testing.assert_allclose(
                grad.numpy(), grad_std.numpy(), rtol=1e-20
            )


if __name__ == "__main__":
    unittest.main()
