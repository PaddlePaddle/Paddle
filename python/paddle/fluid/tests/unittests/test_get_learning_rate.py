# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid.framework as framework
import paddle.fluid.optimizer as optimizer
from paddle.fluid.backward import append_backward
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.nn import FC
from paddle.fluid.optimizer import SGDOptimizer
import os
import pickle
import numpy.testing as npt
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay
import random
import math
import paddle.fluid.layers as layers


class MLP(fluid.Layer):
    def __init__(self, name_scope):
        super(MLP, self).__init__(name_scope)
        self._fc1 = FC(self.full_name(), 10)
        self._fc2 = FC(self.full_name(), 10)

    def forward(self, inputs):
        y = self._fc1(inputs)
        y = self._fc2(y)
        return y


class TestGetLearnignRate(unittest.TestCase):
    def test_get_learning_rate(self):
        with fluid.dygraph.guard():
            mlp_model = MLP("mlp")
            lr = 0.1
            decay_steps = 2
            decay_rate = 0.5
            staircase = False

            sgd = SGDOptimizer(learning_rate=fluid.layers.exponential_decay(
                learning_rate=0.1,
                decay_steps=2,
                decay_rate=0.5,
                staircase=False))

            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=128, drop_last=True)

            def expected_exp_decay_lr(learning_rate,
                                      global_step,
                                      decay_steps,
                                      decay_rate,
                                      staircase=False):
                exponent = global_step / decay_steps
                if staircase:
                    exponent = math.floor(exponent)
                return learning_rate * decay_rate**exponent

            def expected_natural_exp_decay_lr(learning_rate,
                                              global_step,
                                              decay_steps,
                                              decay_rate,
                                              staircase=False):
                exponent = float(global_step) / float(decay_steps)
                if staircase:
                    exponent = math.floor(exponent)
                return learning_rate * math.exp(-1 * decay_rate * exponent)

            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                    128, 1)
                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label._stop_gradient = True
                cost = mlp_model(img)
                avg_loss = fluid.layers.reduce_mean(cost)
                avg_loss.backward()
                sgd.minimize(avg_loss)
                self.assertAlmostEqual(
                    sgd.learning_rate()[0],
                    expected_exp_decay_lr(lr, batch_id, decay_steps, decay_rate,
                                          staircase))
                if batch_id == 2:
                    break


class TestGetLearnignRateRaise(unittest.TestCase):
    def test_get_learning_rate(self):
        train_data = np.array([[1.0], [2.0], [3.0], [4.0]]).astype('float32')
        y_true = np.array([[2.0], [4.0], [6.0], [8.0]]).astype('float32')
        x = fluid.layers.data(name="x", shape=[1], dtype='float32')
        y = fluid.layers.data(name="y", shape=[1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)
        optimizer = fluid.optimizer.Adam(learning_rate=1e-3)
        self.assertRaises(TypeError, optimizer.learning_rate)


if __name__ == '__main__':
    unittest.main()
