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

import contextlib
import unittest
import numpy as np
import six

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.optimizer import SGDOptimizer, Adam
from paddle.fluid.dygraph.nn import FC
from paddle.fluid.dygraph.base import to_variable
from test_imperative_base import new_program_scope


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

    def prepare_places(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        return places

    def _check_mlp(self):
        seed = 90
        batch_size = 128
        places = self.prepare_places()

        with fluid.dygraph.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            mlp = MLP('mlp')
            optimizer = self.get_optimizer()
            image = to_variable(np.array([], dtype='float32'), name='image')
            label = to_variable(np.array([], dtype='int64'), name='label')

            py_reader = fluid.io.PyReader(
                feed_list=[image, label],
                capacity=batch_size,
                iterable=True,
                use_double_buffer=True)
            py_reader.decorate_batch_generator(
                paddle.dataset.mnist.train(), places=places)
            batch_py_reader = paddle.batch(
                py_reader, batch_size=batch_size, drop_last=True)

            dy_param_init_value = {}
            for batch_id, data in enumerate(batch_py_reader()):
                if batch_id >= self.batch_num:
                    break

                dy_x_data = np.array([np.array(x[0]['image']).reshape(1, 28, 28) for x in data]) \
                    .astype('float32')
                y_data = np.array([np.array(x[0]['label']) for x in data]) \
                    .astype('int64').reshape(batch_size, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label._stop_gradient = True

                cost = mlp(img)
                avg_loss = fluid.layers.reduce_mean(cost)
                dy_out = avg_loss.numpy()

                if batch_id == 0:
                    for param in mlp.parameters():
                        dy_param_init_value[param.name] = param.numpy()

                avg_loss.backward()
                optimizer.minimize(avg_loss)
                mlp.clear_gradients()
                dy_param_value = {}
                for param in mlp.parameters():
                    dy_param_value[param.name] = param.numpy()

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))

            mlp = MLP('mlp')
            optimizer = self.get_optimizer()
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=128, drop_last=True)

            img = fluid.layers.data(
                name='pixel', shape=[1, 28, 28], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            cost = mlp(img)
            avg_loss = fluid.layers.reduce_mean(cost)
            optimizer.minimize(avg_loss)

            # initialize params and fetch them
            static_param_init_value = {}
            static_param_name_list = []
            for param in mlp.parameters():
                static_param_name_list.append(param.name)

            out = exe.run(fluid.default_startup_program(),
                          fetch_list=static_param_name_list)

            for i in range(len(static_param_name_list)):
                static_param_init_value[static_param_name_list[i]] = out[i]

            for batch_id, data in enumerate(train_reader()):
                if batch_id >= self.batch_num:
                    break

                static_x_data = np.array(
                    [x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                    [128, 1])

                fetch_list = [avg_loss.name]
                fetch_list.extend(static_param_name_list)
                out = exe.run(fluid.default_main_program(),
                              feed={"pixel": static_x_data,
                                    "label": y_data},
                              fetch_list=fetch_list)

                static_param_value = {}
                static_out = out[0]
                for i in range(1, len(out)):
                    static_param_value[static_param_name_list[i - 1]] = out[i]

        for key, value in six.iteritems(static_param_init_value):
            self.assertTrue(np.allclose(value, dy_param_init_value[key]))

        self.assertTrue(np.allclose(static_out, dy_out))

        for key, value in six.iteritems(static_param_value):
            self.assertTrue(np.allclose(value, dy_param_value[key]))


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
