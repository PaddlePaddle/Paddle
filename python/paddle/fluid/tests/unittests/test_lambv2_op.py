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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
from paddle.fluid import core
from paddle.fluid.op import Operator
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers


class LAMBOptimizer(paddle.optimizer.Lamb):
    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, fluid.framework.Block)
        block.program._use_lamb = True

        m = moment1 = self._get_accumulator(self._moment1_acc_str,
                                            param_and_grad[0])
        v = self._get_accumulator(self._moment2_acc_str, param_and_grad[0])
        beta_1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                               param_and_grad[0])
        beta_2_pow_acc = self._get_accumulator(self._beta2_pow_acc_str,
                                               param_and_grad[0])

        beta_1 = layers.fill_constant(
            dtype='float32', shape=[1], value=self._beta1, name='lamb_beta_1')
        beta_2 = layers.fill_constant(
            dtype='float32', shape=[1], value=self._beta2, name='lamb_beta_2')
        epsilon = layers.fill_constant(
            dtype='float32', shape=[1], value=self._epsilon, name='epsilon')

        one = paddle.ones(shape=[1]).astype('float32')
        zero = paddle.zeros(shape=[1]).astype('float32')

        next_m = paddle.multiply(m, beta_1) + paddle.multiply(param_and_grad[1],
                                                              one - beta_1)
        next_v = paddle.multiply(v, beta_2) + paddle.multiply(
            paddle.pow(param_and_grad[1], 2), one - beta_2)

        beta1_correction = one - beta_1_pow_acc
        beta2_correction = one - beta_2_pow_acc

        next_m_unbiased = next_m / beta1_correction
        next_v_unbiased = next_v / beta2_correction

        update = next_m_unbiased / (paddle.sqrt(next_v_unbiased) + epsilon)

        if self._exclude_from_weight_decay_fn is not None and self._exclude_from_weight_decay_fn(
                param_and_grad[0]):
            self._lamb_weight_decay = 0.0
        update += self._lamb_weight_decay * param_and_grad[0]

        w_norm = paddle.norm(param_and_grad[0], p=2)
        g_norm = paddle.norm(update, p=2)

        learning_rate = self._create_param_lr(param_and_grad)

        ratio = paddle.where(
            paddle.greater_than(w_norm, zero),
            paddle.where(
                paddle.greater_than(g_norm, zero), (w_norm / g_norm), one), one)
        update_with_lr = ratio * learning_rate * update
        next_param = param_and_grad[0] - update_with_lr

        beta_1_pow_acc *= beta_1
        beta_2_pow_acc *= beta_2

        paddle.assign(next_m, m)
        paddle.assign(next_v, v)
        paddle.assign(next_param, param_and_grad[0])

        return None


class TestLambOpV2(unittest.TestCase):
    def test_lamb_op(self):
        shape = [2, 4, 8, 8]
        data = paddle.to_tensor(np.random.random(size=shape).astype("float32"))
        conv = paddle.nn.Conv2D(4, 6, (3, 3))
        data = conv(data)
        loss = paddle.mean(data)
        opt = paddle.optimizer.Lamb(
            learning_rate=1e-5, epsilon=1e-8, parameters=conv.parameters())
        loss.backward()
        opt.minimize(loss)

        assert loss.numpy() is not None


class TestLambOpWithCombinedOp(unittest.TestCase):
    def test_lamb_op_with_multi_steps(self):
        paddle.enable_static()

        def _build_static_model(main, startup, seed=100):
            with fluid.program_guard(main, startup):
                main.random_seed = seed
                startup.random_seed = seed
                x = fluid.layers.data(name='X', shape=[13], dtype='float32')
                y = fluid.layers.data(name='Y', shape=[1], dtype='float32')
                prediction = fluid.layers.fc(input=x, size=1, act=None)
                loss = fluid.layers.square_error_cost(input=prediction, label=y)
                avg_loss = fluid.layers.mean(loss)
            return avg_loss

        place = fluid.CPUPlace()
        num_steps = 10

        for i in range(num_steps):
            feed_x = np.random.random(size=(10, 13)).astype('float32')
            feed_y = np.random.random(size=(10, 1)).astype('float32')

            main_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(main_program, startup_program):
                avg_loss = _build_static_model(main_program, startup_program)
                lamb_kernel = paddle.optimizer.Lamb(learning_rate=0.2)
                lamb_kernel.minimize(avg_loss)

            executor = fluid.Executor(place)
            executor.run(startup_program)
            output = executor.run(program=main_program,
                                  feed={'X': feed_x,
                                        'Y': feed_y},
                                  fetch_list=[avg_loss.name])

            main = fluid.Program()
            startup = fluid.Program()
            with fluid.program_guard(main, startup):
                loss = _build_static_model(main, startup)
                lamb = LAMBOptimizer(learning_rate=0.2)
                lamb.minimize(loss)

            exe = fluid.Executor(place)
            exe.run(startup)
            out = exe.run(program=main,
                          feed={'X': feed_x,
                                'Y': feed_y},
                          fetch_list=[loss.name])

            self.assertTrue(np.allclose(out, output))


class TestLambOpV2Group(TestLambOpV2):
    def test_lamb_op(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.Lamb(
            learning_rate=0.01,
            parameters=[{
                'params': linear_1.parameters()
            }, {
                'params': linear_2.parameters(),
                'lamb_weight_decay': 0.001,
                'beta1': 0.9,
                'beta2': 0.99
            }],
            lamb_weight_decay=0.01)
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        adam.step()
        adam.clear_gradients()


if __name__ == "__main__":
    unittest.main()
