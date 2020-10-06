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
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from op_test import OpTest
import paddle
import paddle.fluid as fluid


class TestMomentumOp1(OpTest):
    def setUp(self):
        self.op_type = "momentum"
        self.dtype = np.float32
        self.init_dtype()

        param = np.random.random((123, 321)).astype(self.dtype)
        grad = np.random.random((123, 321)).astype(self.dtype)
        velocity = np.zeros((123, 321)).astype(self.dtype)
        learning_rate = np.array([0.001]).astype(self.dtype)
        mu = 0.0001
        use_nesterov = False

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Velocity': velocity,
            'LearningRate': learning_rate
        }

        self.attrs = {'mu': mu}

        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = param - grad * learning_rate - \
                        velocity_out * mu * learning_rate
        else:
            param_out = param - learning_rate * velocity_out

        self.outputs = {'ParamOut': param_out, 'VelocityOut': velocity_out}

    def init_dtype(self):
        pass

    def test_check_output(self):
        self.check_output()


class TestMomentumV2(unittest.TestCase):
    def test_momentum_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.Momentum(
            learning_rate=0.01, momentum=0.9, parameters=linear.parameters())
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()

    def test_momentum(self):
        place = fluid.CPUPlace()
        main = fluid.Program()
        with fluid.program_guard(main):
            x = fluid.layers.data(name='x', shape=[13], dtype='float32')
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            y_predict = fluid.layers.fc(input=x, size=1, act=None)
            cost = fluid.layers.square_error_cost(input=y_predict, label=y)
            avg_cost = fluid.layers.mean(cost)

            rms_optimizer = paddle.optimizer.Momentum(
                learning_rate=0.1, momentum=0.9)
            rms_optimizer.minimize(avg_cost)

            fetch_list = [avg_cost]
            train_reader = paddle.batch(
                paddle.dataset.uci_housing.train(), batch_size=1)
            feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for data in train_reader():
                exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    def test_raise_error(self):
        self.assertRaises(
            ValueError, paddle.optimizer.Momentum, learning_rate=None)
        self.assertRaises(ValueError, paddle.optimizer.Momentum, momentum=None)


if __name__ == "__main__":
    unittest.main()
