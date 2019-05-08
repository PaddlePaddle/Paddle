# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import SGDOptimizer, Adam
from paddle.fluid.dygraph.nn import FC
from paddle.fluid.dygraph.base import to_variable


class MLP(fluid.Layer):
    def __init__(self, name_scope, param_attr=None, bias_attr=None):
        super(MLP, self).__init__(name_scope)

        self._fc1 = FC(self.full_name(), 10)
        self._fc2 = FC(self.full_name(), 10)

    def forward(self, inputs):
        y = self._fc1(inputs)
        y = self._fc2(y)
        return y


class TestImperativeOptimizerBase(unittest.TestCase):
    def setUp(self):
        self.batch_num = 20

    def get_optimizer(self):
        raise NotImplementedError()

    def _check_mlp(self):
        seed = 90
        with fluid.dygraph.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            mlp = MLP('mlp')
            optimizer = self.get_optimizer()
            optimizer2 = SGDOptimizer(
                learning_rate=fluid.layers.natural_exp_decay(
                    learning_rate=0.1,
                    decay_steps=10000,
                    decay_rate=0.5,
                    staircase=True))
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=128, drop_last=True)

            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                    128, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label._stop_gradient = True

                cost = mlp(img)
                avg_loss = fluid.layers.reduce_mean(cost)

                avg_loss.backward()
                optimizer.minimize(avg_loss)
                optimizer2.minimize(avg_loss)
                mlp.clear_gradients()
                fluid.dygraph.save_persistables(
                    mlp.state_dict(), [optimizer, optimizer2], "save_dir_2")
                if batch_id == 2:
                    break

        with fluid.dygraph.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            mlp_load = MLP('mlp')
            optimizer_load1 = self.get_optimizer()
            optimizer_load2 = SGDOptimizer(
                learning_rate=fluid.layers.natural_exp_decay(
                    learning_rate=0.1,
                    decay_steps=10000,
                    decay_rate=0.5,
                    staircase=True))
            parameters, optimizers = fluid.dygraph.load_persistables(
                "save_dir_2")
            mlp_load.load_dict(parameters)
            optimizer_load1.load_dict(optimizers)
            optimizer_load2.load_dict(optimizers)

        self.assertTrue(optimizer._learning_rate.__dict__ ==
                        optimizer_load1._learning_rate.__dict__)
        self.assertTrue(optimizer2._learning_rate.__dict__ ==
                        optimizer_load2._learning_rate.__dict__)


class TestImperativeOptimizerPiecewiseDecay(TestImperativeOptimizerBase):
    def get_optimizer(self):
        bd = [3, 6, 9]
        optimizer = SGDOptimizer(learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=[0.1 * (0.1**i) for i in range(len(bd) + 1)]))
        return optimizer

    def test_sgd(self):
        self._check_mlp()


class TestImperativeOptimizerNaturalExpDecay(TestImperativeOptimizerBase):
    def get_optimizer(self):
        optimizer = SGDOptimizer(learning_rate=fluid.layers.natural_exp_decay(
            learning_rate=0.1,
            decay_steps=10000,
            decay_rate=0.5,
            staircase=True))
        return optimizer

    def test_sgd(self):
        self._check_mlp()


class TestImperativeOptimizerExponentialDecay(TestImperativeOptimizerBase):
    def get_optimizer(self):
        optimizer = SGDOptimizer(learning_rate=fluid.layers.exponential_decay(
            learning_rate=0.1,
            decay_steps=10000,
            decay_rate=0.5,
            staircase=True))
        return optimizer

    def test_sgd(self):
        self._check_mlp()


class TestImperativeOptimizerInverseTimeDecay(TestImperativeOptimizerBase):
    def get_optimizer(self):
        optimizer = Adam(learning_rate=fluid.layers.inverse_time_decay(
            learning_rate=0.1,
            decay_steps=10000,
            decay_rate=0.5,
            staircase=True))
        return optimizer

    def test_adam(self):
        self._check_mlp()


class TestImperativeOptimizerPolynomialDecay(TestImperativeOptimizerBase):
    def get_optimizer(self):
        optimizer = SGDOptimizer(learning_rate=fluid.layers.polynomial_decay(
            learning_rate=0.1, decay_steps=5, cycle=self.cycle))
        return optimizer

    def test_sgd_cycle(self):
        self.cycle = True
        self._check_mlp()

    def test_sgd(self):
        self.cycle = False
        self._check_mlp()


class TestImperativeOptimizerCosineDecay(TestImperativeOptimizerBase):
    def get_optimizer(self):
        optimizer = SGDOptimizer(learning_rate=fluid.layers.cosine_decay(
            learning_rate=0.1, step_each_epoch=10000, epochs=120))
        return optimizer

    def test_sgd(self):
        self._check_mlp()


class TestImperativeOptimizerNoamDecay(TestImperativeOptimizerBase):
    def get_optimizer(self):
        optimizer = SGDOptimizer(learning_rate=fluid.layers.noam_decay(
            d_model=512, warmup_steps=8000))
        return optimizer

    def test_sgd(self):
        self._check_mlp()


if __name__ == '__main__':
    unittest.main()
