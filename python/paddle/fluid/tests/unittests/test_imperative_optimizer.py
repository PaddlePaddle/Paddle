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
from paddle.fluid.optimizer import SGDOptimizer, Adam, MomentumOptimizer, LarsMomentumOptimizer, AdagradOptimizer, AdamaxOptimizer, DpsgdOptimizer, DecayedAdagradOptimizer, AdadeltaOptimizer, RMSPropOptimizer, FtrlOptimizer, LambOptimizer
from paddle.fluid.optimizer import ModelAverage, DGCMomentumOptimizer, ExponentialMovingAverage, PipelineOptimizer, LookaheadOptimizer, RecomputeOptimizer
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph.base import to_variable
from test_imperative_base import new_program_scope

# Note(wangzhongpu)
# In dygraph, don't support ModelAverage, DGCMomentumOptimizer, ExponentialMovingAverage, PipelineOptimizer, LookaheadOptimizer, RecomputeOptimizer.


class MLP(fluid.Layer):
    def __init__(self, param_attr=None, bias_attr=None):
        super(MLP, self).__init__()

        self._fc1 = Linear(784, 10)
        self._fc2 = Linear(10, 10)

    def forward(self, inputs):
        y = self._fc1(inputs)
        y = self._fc2(y)
        return y


class TestImperativeOptimizerBase(unittest.TestCase):
    def setUp(self):
        self.batch_num = 20

    def get_optimizer_dygraph(self, parameter_list):
        raise NotImplementedError()

    def get_optimizer(self):
        raise NotImplementedError()

    def reader_decorator(self, reader):
        def _reader_imple():
            for item in reader():
                image = np.array(item[0]).reshape(1, 784)
                label = np.array(item[1]).astype('int64').reshape(1)
                yield image, label

        return _reader_imple

    def _check_exception(self, exception_message, place=None):
        seed = 90
        batch_size = 128
        if place == None:
            place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()

        with fluid.dygraph.guard(place):
            try:
                fluid.default_startup_program().random_seed = seed
                fluid.default_main_program().random_seed = seed
                mlp = MLP()
                optimizer = self.get_optimizer_dygraph(
                    parameter_list=mlp.parameters())
            except Exception as e:
                assert str(e) == exception_message

    def _check_mlp(self, place=None):
        seed = 90
        batch_size = 128

        if place == None:
            place = fluid.CPUPlace() if not core.is_compiled_with_cuda(
            ) else fluid.CUDAPlace(0)

        with fluid.dygraph.guard(place):
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            mlp = MLP()
            optimizer = self.get_optimizer_dygraph(
                parameter_list=mlp.parameters())

            batch_py_reader = fluid.io.PyReader(capacity=1)
            batch_py_reader.decorate_sample_list_generator(
                paddle.batch(
                    self.reader_decorator(paddle.dataset.mnist.train()),
                    batch_size=batch_size,
                    drop_last=True),
                places=fluid.CPUPlace())

            dy_param_init_value = {}
            for batch_id, data in enumerate(batch_py_reader()):
                if batch_id >= self.batch_num:
                    break

                img = data[0]
                label = data[1]
                label.stop_gradient = True

                img = fluid.layers.reshape(img, shape=[batch_size, -1])
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

            if place == None:
                place = fluid.CPUPlace() if not core.is_compiled_with_cuda(
                ) else fluid.CUDAPlace(0)

            exe = fluid.Executor(place)

            mlp = MLP()
            optimizer = self.get_optimizer()
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=128, drop_last=True)

            img = fluid.layers.data(
                name='pixel', shape=[1, 28, 28], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            img = fluid.layers.reshape(img, shape=[batch_size, 784])
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
    def get_optimizer_dygraph(self, parameter_list):
        bd = [3, 6, 9]
        optimizer = SGDOptimizer(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd,
                values=[0.1 * (0.1**i) for i in range(len(bd) + 1)]),
            parameter_list=parameter_list)
        return optimizer

    def get_optimizer(self):
        bd = [3, 6, 9]
        optimizer = SGDOptimizer(learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=[0.1 * (0.1**i) for i in range(len(bd) + 1)]))
        return optimizer

    def test_sgd(self):
        self._check_mlp()


class TestImperativeOptimizerNaturalExpDecay(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = SGDOptimizer(
            learning_rate=fluid.layers.natural_exp_decay(
                learning_rate=0.1,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True),
            parameter_list=parameter_list)
        return optimizer

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
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = SGDOptimizer(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=0.1,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True),
            parameter_list=parameter_list)
        return optimizer

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
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = Adam(
            learning_rate=fluid.layers.inverse_time_decay(
                learning_rate=0.1,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True),
            parameter_list=parameter_list)
        return optimizer

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
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = SGDOptimizer(
            learning_rate=fluid.layers.polynomial_decay(
                learning_rate=0.1, decay_steps=5, cycle=self.cycle),
            parameter_list=parameter_list)
        return optimizer

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
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = SGDOptimizer(
            learning_rate=fluid.layers.cosine_decay(
                learning_rate=0.1, step_each_epoch=10000, epochs=120),
            parameter_list=parameter_list)
        return optimizer

    def get_optimizer(self):
        optimizer = SGDOptimizer(learning_rate=fluid.layers.cosine_decay(
            learning_rate=0.1, step_each_epoch=10000, epochs=120))
        return optimizer

    def test_sgd(self):
        self._check_mlp()


class TestImperativeOptimizerNoamDecay(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = SGDOptimizer(
            learning_rate=fluid.layers.noam_decay(
                d_model=512, warmup_steps=8000),
            parameter_list=parameter_list)
        return optimizer

    def get_optimizer(self):
        optimizer = SGDOptimizer(learning_rate=fluid.layers.noam_decay(
            d_model=512, warmup_steps=8000))
        return optimizer

    def test_sgd(self):
        self._check_mlp()


class TestOptimizerLearningRate(unittest.TestCase):
    def test_constant_lr(self):
        with fluid.dygraph.guard():
            a = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")

            linear = fluid.dygraph.nn.Linear(10, 10)

            a = fluid.dygraph.to_variable(a)

            b = linear(a)

            loss = fluid.layers.reduce_mean(b)

            adam = fluid.optimizer.Adam(
                0.001, parameter_list=linear.parameters())

            self.assertTrue(
                np.allclose(
                    adam.current_step_lr(), 0.001, rtol=1e-06, atol=0.0))

            for i in range(10):
                adam.minimize(loss)
                lr = adam.current_step_lr()

                self.assertTrue(np.allclose(lr, 0.001, rtol=1e-06, atol=0.0))

    def test_lr_decay(self):
        with fluid.dygraph.guard():
            a = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")

            linear = fluid.dygraph.nn.Linear(10, 10)

            a = fluid.dygraph.to_variable(a)

            b = linear(a)

            loss = fluid.layers.reduce_mean(b)

            bd = [2, 4, 6, 8]
            value = [0.2, 0.4, 0.6, 0.8, 1.0]

            adam = fluid.optimizer.Adam(
                fluid.dygraph.PiecewiseDecay(bd, value, 0),
                parameter_list=linear.parameters())

            self.assertTrue(
                np.allclose(
                    adam.current_step_lr(), 0.2, rtol=1e-06, atol=0.0))

            ret = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
            for i in range(12):
                adam.minimize(loss)
                lr = adam.current_step_lr()

                self.assertTrue(np.allclose(lr, ret[i], rtol=1e-06, atol=0.0))

    def test_lr_decay_natural_exp(self):
        with fluid.dygraph.guard():
            a = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")

            linear = fluid.dygraph.nn.Linear(10, 10)

            a = fluid.dygraph.to_variable(a)

            b = linear(a)

            loss = fluid.layers.reduce_mean(b)
            base_lr = 1.0

            adam = fluid.optimizer.Adam(
                fluid.dygraph.NaturalExpDecay(
                    learning_rate=base_lr,
                    decay_steps=3,
                    decay_rate=0.5,
                    staircase=True),
                parameter_list=linear.parameters())

            self.assertTrue(
                np.allclose(
                    adam.current_step_lr(), 1.0, rtol=1e-06, atol=0.0))

            ret = [1.0, 1.0, 1.0, np.exp(-0.5), np.exp(-0.5)]
            for i in range(5):
                adam.minimize(loss)
                lr = adam.current_step_lr()

                self.assertTrue(np.allclose(lr, ret[i], rtol=1e-06, atol=0.0))


class TestImperativeMomentumOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = MomentumOptimizer(
            learning_rate=0.001, momentum=0.9, parameter_list=parameter_list)
        return optimizer

    def get_optimizer(self):
        optimizer = MomentumOptimizer(learning_rate=0.001, momentum=0.9)
        return optimizer

    def test_momentum(self):
        self._check_mlp()


class TestImperativeLarsMomentumOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = LarsMomentumOptimizer(
            learning_rate=0.001, momentum=0.9, parameter_list=parameter_list)
        return optimizer

    def get_optimizer(self):
        optimizer = LarsMomentumOptimizer(learning_rate=0.001, momentum=0.9)
        return optimizer

    def test_larsmomentum(self):
        self._check_mlp()


class TestImperativeAdagradOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = AdagradOptimizer(
            learning_rate=0.2, parameter_list=parameter_list)
        return optimizer

    def get_optimizer(self):
        optimizer = AdagradOptimizer(learning_rate=0.2)
        return optimizer

    def test_adagrad(self):
        self._check_mlp()


class TestImperativeAdamaxOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = AdamaxOptimizer(
            learning_rate=0.2, parameter_list=parameter_list)
        return optimizer

    def get_optimizer(self):
        optimizer = AdamaxOptimizer(learning_rate=0.2)
        return optimizer

    def test_adamax(self):
        self._check_mlp()


class TestImperativeDpsgdOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = DpsgdOptimizer(
            learning_rate=0.01,
            clip=10.0,
            batch_size=16.0,
            sigma=1.0,
            parameter_list=parameter_list)
        optimizer._seed = 100
        return optimizer

    def get_optimizer(self):
        optimizer = DpsgdOptimizer(
            learning_rate=0.01, clip=10.0, batch_size=16.0, sigma=1.0)
        optimizer._seed = 100
        return optimizer

    def test_dpsgd(self):
        self._check_mlp(place=fluid.CPUPlace())


class TestImperativeDecayedAdagradOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = DecayedAdagradOptimizer(
            learning_rate=0.2, parameter_list=parameter_list)
        return optimizer

    def get_optimizer(self):
        optimizer = DecayedAdagradOptimizer(learning_rate=0.2)
        return optimizer

    def test_decayadagrad(self):
        self._check_mlp()


class TestImperativeAdadeltaOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = AdadeltaOptimizer(
            learning_rate=0.0003,
            epsilon=1.0e-6,
            rho=0.95,
            parameter_list=parameter_list)
        return optimizer

    def get_optimizer(self):
        optimizer = AdadeltaOptimizer(
            learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)
        return optimizer

    def test_adadelta(self):
        self._check_mlp()


class TestImperativeRMSPropOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = RMSPropOptimizer(
            learning_rate=0.1, parameter_list=parameter_list)
        return optimizer

    def get_optimizer(self):
        optimizer = RMSPropOptimizer(learning_rate=0.1)
        return optimizer

    def test_rmsprop(self):
        self._check_mlp()


class TestImperativeFtrlOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = FtrlOptimizer(
            learning_rate=0.1, parameter_list=parameter_list)
        return optimizer

    def get_optimizer(self):
        optimizer = FtrlOptimizer(learning_rate=0.1)
        return optimizer

    def test_ftrl(self):
        self._check_mlp()


def exclude_fn(param):
    return param.name.endswith('.b_0')


class TestImperativeLambOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = LambOptimizer(
            learning_rate=0.002,
            exclude_from_weight_decay_fn=exclude_fn,
            parameter_list=parameter_list)
        return optimizer

    def get_optimizer(self):
        optimizer = LambOptimizer(
            learning_rate=0.002, exclude_from_weight_decay_fn=exclude_fn)
        return optimizer

    def test_lamb(self):
        self._check_mlp()


class TestImperativeModelAverage(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = ModelAverage(
            0.15, min_average_window=10000, max_average_window=12500)
        return optimizer

    def test_modelaverage(self):
        exception_message = "In dygraph, don't support ModelAverage."
        self._check_exception(exception_message)


class TestImperativeDGCMomentumOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = DGCMomentumOptimizer(
            learning_rate=0.0001,
            momentum=0.9,
            rampup_step=1000,
            rampup_begin_step=1252,
            sparsity=[0.999, 0.999])
        return optimizer

    def test_dgcmomentum(self):
        exception_message = "In dygraph, don't support DGCMomentumOptimizer."
        self._check_exception(exception_message)


class TestImperativeExponentialMovingAverage(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = ExponentialMovingAverage(0.999)
        return optimizer

    def test_exponentialmoving(self):
        exception_message = "In dygraph, don't support ExponentialMovingAverage."
        self._check_exception(exception_message)


class TestImperativePipelineOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = fluid.optimizer.SGD(learning_rate=0.5,
                                        parameter_list=parameter_list)
        optimizer = PipelineOptimizer(optimizer)
        return optimizer

    def test_pipline(self):
        exception_message = "In dygraph, don't support PipelineOptimizer."
        self._check_exception(exception_message)


class TestImperativeLookaheadOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = fluid.optimizer.SGD(learning_rate=0.5,
                                        parameter_list=parameter_list)
        optimizer = LookaheadOptimizer(optimizer, alpha=0.5, k=5)
        return optimizer

    def test_lookahead(self):
        exception_message = "In dygraph, don't support LookaheadOptimizer."
        self._check_exception(exception_message)


class TestImperativeRecomputeOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = fluid.optimizer.SGD(learning_rate=0.5,
                                        parameter_list=parameter_list)
        optimizer = RecomputeOptimizer(optimizer)
        return optimizer

    def test_recompute(self):
        exception_message = "In dygraph, don't support RecomputeOptimizer."
        self._check_exception(exception_message)


if __name__ == '__main__':
    unittest.main()
