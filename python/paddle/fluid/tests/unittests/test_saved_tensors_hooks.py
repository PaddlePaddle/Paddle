# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.autograd import PyLayer


class TestSavedTensorsHooks(unittest.TestCase):

    def test_save_for_multiply(self):

        def pack_hook(x):
            return x.numpy()

        def unpack_hook(x):
            return paddle.to_tensor(x)

        a = paddle.ones([3, 3])
        b = paddle.ones([3, 3]) * 2
        a.stop_gradient = False
        b.stop_gradient = False
        with paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook):
            y = paddle.multiply(a, b)
        y.sum().backward()

        aa = paddle.ones([3, 3])
        bb = paddle.ones([3, 3]) * 2
        aa.stop_gradient = False
        bb.stop_gradient = False
        yy = paddle.multiply(aa, bb)
        yy.sum().backward()

        self.assertTrue(paddle.equal_all(aa.grad, a.grad))
        self.assertTrue(paddle.equal_all(bb.grad, b.grad))

    def test_save_for_pylayer(self):

        class cus_multiply(PyLayer):

            @staticmethod
            def forward(ctx, a, b):
                y = paddle.multiply(a, b)
                ctx.save_for_backward(a, b)
                return y

            @staticmethod
            def backward(ctx, dy):
                a, b = ctx.saved_tensor()
                grad_a = dy * a
                grad_b = dy * b
                return grad_a, grad_b

        def pack_hook(x):
            return x.numpy()

        def unpack_hook(x):
            return paddle.to_tensor(x)

        a = paddle.ones([3, 3])
        b = paddle.ones([3, 3]) * 2
        a.stop_gradient = False
        b.stop_gradient = False
        with paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook):
            y = cus_multiply.apply(a, b)
        y.sum().backward()

        aa = paddle.ones([3, 3])
        bb = paddle.ones([3, 3]) * 2
        aa.stop_gradient = False
        bb.stop_gradient = False
        yy = cus_multiply.apply(aa, bb)
        yy.sum().backward()

        self.assertTrue(paddle.equal_all(aa.grad, a.grad))
        self.assertTrue(paddle.equal_all(bb.grad, b.grad))


if __name__ == '__main__':
    unittest.main()
