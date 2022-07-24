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
from paddle.incubate.autograd.utils import (disable_prim, enable_prim,
                                            prim_enabled)

paddle.enable_static()


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
