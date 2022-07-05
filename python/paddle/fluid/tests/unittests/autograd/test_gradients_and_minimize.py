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
from paddle.incubate.autograd.primx import prim2orig
from paddle.incubate.autograd.utils import enable_prim, disable_prim, prim_enabled

paddle.enable_static()


class TestGradients(unittest.TestCase):

    def test_third_order(self):
        enable_prim()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name='x', shape=[1], dtype='float32')
            x2 = paddle.multiply(x, x)
            x3 = paddle.multiply(x2, x)
            x4 = paddle.multiply(x3, x)

            grad1, = paddle.static.gradients([x4], [x])
            grad2, = paddle.static.gradients([grad1], [x])
            grad3, = paddle.static.gradients([grad2], [x])

            prim2orig(main.block(0))

        feed = {x.name: np.array([2.]).astype('float32')}
        fetch_list = [grad3.name]
        result = [np.array([48.])]

        place = paddle.CPUPlace()
        if paddle.device.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        exe.run(startup)
        outs = exe.run(main, feed=feed, fetch_list=fetch_list)
        np.allclose(outs, result)
        disable_prim()

    def test_fourth_order(self):
        enable_prim()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name='x', shape=[1], dtype='float32')
            x2 = paddle.multiply(x, x)
            x3 = paddle.multiply(x2, x)
            x4 = paddle.multiply(x3, x)
            x5 = paddle.multiply(x4, x)
            out = paddle.sqrt(x5 + x4)

            grad1, = paddle.static.gradients([out], [x])
            grad2, = paddle.static.gradients([grad1], [x])
            grad3, = paddle.static.gradients([grad2], [x])
            grad4, = paddle.static.gradients([grad3], [x])

            prim2orig(main.block(0))

        feed = {
            x.name: np.array([2.]).astype('float32'),
        }
        fetch_list = [grad4.name]
        # (3*(-5*x^2-16*x-16))/(16*(x+1)^3.5)
        result = [np.array([-0.27263762711])]

        place = paddle.CPUPlace()
        if paddle.device.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        exe.run(startup)
        outs = exe.run(main, feed=feed, fetch_list=fetch_list)
        np.allclose(outs, result)
        disable_prim()


class TestMinimize(unittest.TestCase):

    def model(self, x, w, bias, opt):
        paddle.seed(0)
        place = paddle.CPUPlace()
        if paddle.device.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input_x = paddle.static.data('x', x.shape, dtype=x.dtype)
            input_x.stop_gradient = False
            params_w = paddle.static.create_parameter(shape=w.shape,
                                                      dtype=w.dtype,
                                                      is_bias=False)
            params_bias = paddle.static.create_parameter(shape=bias.shape,
                                                         dtype=bias.dtype,
                                                         is_bias=True)
            y = paddle.tanh(paddle.matmul(input_x, params_w) + params_bias)
            loss = paddle.norm(y, p=2)
            opt = opt
            _, grads = opt.minimize(loss)
            if prim_enabled():
                prim2orig(main.block(0))
        exe.run(startup)
        grads = exe.run(main,
                        feed={
                            'x': x,
                            'w': w,
                            'bias': bias
                        },
                        fetch_list=grads)
        return grads

    def test_adam(self):
        x = np.random.rand(2, 20)
        w = np.random.rand(20, 2)
        bias = np.random.rand(2)
        enable_prim()
        prim_grads = self.model(x, w, bias, paddle.optimizer.Adam(0.01))
        disable_prim()
        orig_grads = self.model(x, w, bias, paddle.optimizer.Adam(0.01))
        for orig, prim in zip(orig_grads, prim_grads):
            np.testing.assert_allclose(orig, prim)

    def test_sgd(self):
        x = np.random.rand(2, 20)
        w = np.random.rand(20, 2)
        bias = np.random.rand(2)
        enable_prim()
        prim_grads = self.model(x, w, bias, paddle.optimizer.SGD(0.01))
        disable_prim()
        orig_grads = self.model(x, w, bias, paddle.optimizer.SGD(0.01))
        for orig, prim in zip(orig_grads, prim_grads):
            np.testing.assert_allclose(orig, prim)


if __name__ == '__main__':
    unittest.main()
