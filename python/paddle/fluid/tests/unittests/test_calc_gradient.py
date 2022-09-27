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

import paddle
import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.backward import calc_gradient


class TestCalcGradient(unittest.TestCase):

    def test_calc_gradient(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            x = layers.create_parameter(dtype="float32", shape=[5, 10])
            y = layers.create_parameter(dtype="float32", shape=[10, 8])
            mul_out = layers.mul(x=x, y=y)
            mean_out = paddle.mean(mul_out)
            a = calc_gradient(mean_out, mul_out)
            b = calc_gradient(mean_out, x)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup)
        exe.run(main, feed={}, fetch_list=[a, b])


class TestDoubleGrad(unittest.TestCase):

    def test1(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            net = lambda x: x * x
            x = fluid.layers.create_parameter(
                name='x',
                shape=[1],
                dtype='float32',
                default_initializer=fluid.initializer.Constant(3))
            grad1, = fluid.gradients(net(x), x)  # 2x = 6
            z = net(x - grad1)
            grad2, = fluid.gradients(z, x)  # gradients( (x - 2x)^2) = 2x = 6

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup)
        out = exe.run(main, fetch_list=[grad1.name, grad2.name])
        self.assertEqual(6, out[0][0])
        self.assertEqual(6, out[1][0])

    def test2(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            x = fluid.layers.create_parameter(
                name='x',
                shape=[1],
                dtype='float32',
                default_initializer=fluid.initializer.Constant(1))
            y = x * x
            dx1, = fluid.gradients(y, x)
            z = dx1 * dx1 + y * y
            dx2, = fluid.gradients(z, x)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup)
        out, = exe.run(main, fetch_list=[dx2])
        self.assertEqual(12, out[0])


class TestGradientWithPrune(unittest.TestCase):

    def test_prune(self):
        with paddle.fluid.scope_guard(paddle.static.Scope()):
            x = fluid.data(name='x', shape=[3], dtype='float32')
            x.stop_gradient = False
            x1, x2, x3 = fluid.layers.split(x, dim=0, num_or_sections=3)
            y = x1 * 2
            x1_grad = fluid.gradients(y, x)

            exe = fluid.Executor(fluid.CPUPlace())
            main = fluid.default_main_program()
            exe.run(fluid.default_startup_program())
            out = exe.run(main,
                          feed={'x': np.ones([3]).astype('float32')},
                          fetch_list=[x1_grad])
            np.testing.assert_array_equal(out[0], [2.0, 0.0, 0.0])


class TestDoubleGradient(unittest.TestCase):

    def build_program(self):
        start_prog = paddle.static.Program()
        main_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, start_prog):
            x = paddle.static.data('x', shape=[2, 2])
            x.stop_gradient = False
            y = x * x

            v = paddle.ones([2, 2])
            v.stop_gradient = False

            grad_y = paddle.zeros_like(y)
            grad_y.stop_gradient = False
            grad_x = paddle.static.gradients(y, x, grad_y)
            # test with single targets
            jvp = paddle.static.gradients(grad_x, grad_y, v)

        return start_prog, main_prog, [grad_x, jvp]

    def test_calc_gradient(self):
        with paddle.fluid.scope_guard(paddle.static.Scope()):
            start_prog, main_prog, fetch_list = self.build_program()
            exe = paddle.static.Executor()
            exe.run(start_prog)
            ans = exe.run(main_prog,
                          feed={'x': np.ones([2, 2]).astype(np.float32)},
                          fetch_list=fetch_list)
            self.assertEqual(len(ans), 2)
            self.assertListEqual(ans[0].tolist(), [[0., 0.], [0., 0.]])
            self.assertListEqual(ans[1].tolist(), [[2., 2.], [2., 2.]])


class TestDoubleGradient2(unittest.TestCase):

    def build_program(self):
        start_prog = paddle.static.Program()
        main_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, start_prog):
            x = paddle.static.data('x', shape=[2, 2])
            x.stop_gradient = False
            y = x * x
            y2 = y + x

            v = paddle.ones([2, 2])
            v.stop_gradient = False

            grad_y = paddle.zeros_like(y)
            grad_y.stop_gradient = False
            grad_x = paddle.static.gradients(y, x, grad_y)
            grad_x2 = paddle.static.gradients(y2, x, grad_y)
            # test with multi targets
            jvp = paddle.static.gradients([grad_x[0], grad_x2[0]], grad_y,
                                          [v, v])

        return start_prog, main_prog, [grad_x, jvp]

    def test_calc_gradient(self):
        with paddle.fluid.scope_guard(paddle.static.Scope()):
            start_prog, main_prog, fetch_list = self.build_program()
            exe = paddle.static.Executor()
            exe.run(start_prog)
            ans = exe.run(main_prog,
                          feed={'x': np.ones([2, 2]).astype(np.float32)},
                          fetch_list=fetch_list)
            self.assertEqual(len(ans), 2)
            self.assertListEqual(ans[0].tolist(), [[0., 0.], [0., 0.]])
            self.assertListEqual(ans[1].tolist(), [[5., 5.], [5., 5.]])


if __name__ == "__main__":
    unittest.main()
