# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
)

import paddle


class ManualPyLayerRecompute(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, fn, x, y):
        ctx.fn = fn
        ctx.save_for_backward(x, y)
        with paddle.no_grad():
            out = fn(x, y)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx.saved_tensor()
        with paddle.enable_grad():
            x.stop_gradient = False
            y.stop_gradient = False
            out = ctx.fn(x, y)
        grad_inputs = paddle.autograd.grad(out, [x, y], grad_out)
        return (*grad_inputs,)


class TestManualPyLayerRecompute(Dy2StTestBase):
    def test_recompute(self):
        @paddle.jit.to_static()
        def fn(x, y):
            return x * y + paddle.sin(x)

        x = paddle.randn([3, 3])
        x.stop_gradient = False
        y = paddle.randn([3, 3])
        y.stop_gradient = False

        out = ManualPyLayerRecompute.apply(fn, x, y)
        out.backward()
        grad_x1 = x.grad.numpy()
        grad_y1 = y.grad.numpy()

        x.clear_gradient()
        y.clear_gradient()

        out2 = fn(x, y)
        out2.backward()
        grad_x2 = x.grad.numpy()
        grad_y2 = y.grad.numpy()

        x.clear_gradient()
        y.clear_gradient()

        np.testing.assert_allclose(grad_x1, grad_x2, rtol=1e-05)
        np.testing.assert_allclose(grad_y1, grad_y2, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
