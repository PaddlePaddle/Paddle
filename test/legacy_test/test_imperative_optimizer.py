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

import unittest

import numpy as np
from test_imperative_base import new_program_scope

import paddle
from paddle import base
from paddle.base import core
from paddle.distributed.fleet.meta_optimizers import DGCMomentumOptimizer

# Note(wangzhongpu)
# In dygraph, don't support ModelAverage, DGCMomentumOptimizer, ExponentialMovingAverage, PipelineOptimizer, LookaheadOptimizer, RecomputeOptimizer.


class MLP(paddle.nn.Layer):
    def __init__(self, param_attr=None, bias_attr=None):
        super().__init__()

        self._fc1 = paddle.nn.Linear(784, 10)
        self._fc2 = paddle.nn.Linear(10, 10)

    def forward(self, inputs):
        y = self._fc1(inputs)
        y = self._fc2(y)
        return y


class TestImperativeOptimizerBase(unittest.TestCase):
    def setUp(self):
        self.batch_num = 20

    def get_optimizer_dygraph(self, parameter_list):
        raise NotImplementedError

    def get_optimizer(self):
        raise NotImplementedError

    def reader_decorator(self, reader):
        def _reader_simple():
            for item in reader():
                image = np.array(item[0]).reshape(1, 784)
                label = np.array(item[1]).astype('int64').reshape(1)
                yield image, label

        return _reader_simple

    def _check_exception(self, exception_message, place=None):
        seed = 90
        batch_size = 128
        if place is None:
            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )

        with base.dygraph.guard(place):
            try:
                paddle.seed(seed)
                paddle.framework.random._manual_program_seed(seed)
                mlp = MLP()
                optimizer = self.get_optimizer_dygraph(
                    parameter_list=mlp.parameters()
                )
            except Exception as e:
                assert str(e) == exception_message

    def _check_mlp(self, place=None):
        seed = 90
        batch_size = 128

        if place is None:
            place = (
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )

        with base.dygraph.guard(place):
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            mlp = MLP()
            optimizer = self.get_optimizer_dygraph(
                parameter_list=mlp.parameters()
            )

            batch_py_reader = base.io.PyReader(capacity=1)
            batch_py_reader.decorate_sample_list_generator(
                paddle.batch(
                    self.reader_decorator(paddle.dataset.mnist.train()),
                    batch_size=batch_size,
                    drop_last=True,
                ),
                places=base.CPUPlace(),
            )

            dy_param_init_value = {}
            for batch_id, data in enumerate(batch_py_reader()):
                if batch_id >= self.batch_num:
                    break

                img = data[0]
                label = data[1]
                label.stop_gradient = True

                img = paddle.reshape(img, shape=[batch_size, -1])
                cost = mlp(img)
                avg_loss = paddle.mean(cost)
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
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            if place is None:
                place = (
                    base.CPUPlace()
                    if not core.is_compiled_with_cuda()
                    else base.CUDAPlace(0)
                )

            exe = base.Executor(place)

            mlp = MLP()
            optimizer = self.get_optimizer()
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=128, drop_last=True
            )

            img = paddle.static.data(
                name='pixel', shape=[-1, 1, 28, 28], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[-1, 1], dtype='int64'
            )
            img = paddle.reshape(img, shape=[batch_size, 784])
            cost = mlp(img)
            avg_loss = paddle.mean(cost)
            optimizer.minimize(avg_loss)

            # initialize params and fetch them
            static_param_init_value = {}
            static_param_name_list = []
            for param in mlp.parameters():
                static_param_name_list.append(param.name)

            out = exe.run(
                base.default_startup_program(),
                fetch_list=static_param_name_list,
            )

            for i in range(len(static_param_name_list)):
                static_param_init_value[static_param_name_list[i]] = out[i]

            for batch_id, data in enumerate(train_reader()):
                if batch_id >= self.batch_num:
                    break

                static_x_data = np.array(
                    [x[0].reshape(1, 28, 28) for x in data]
                ).astype('float32')
                y_data = (
                    np.array([x[1] for x in data])
                    .astype('int64')
                    .reshape([128, 1])
                )

                fetch_list = [avg_loss.name]
                fetch_list.extend(static_param_name_list)
                out = exe.run(
                    base.default_main_program(),
                    feed={"pixel": static_x_data, "label": y_data},
                    fetch_list=fetch_list,
                )

                static_param_value = {}
                static_out = out[0]
                for i in range(1, len(out)):
                    static_param_value[static_param_name_list[i - 1]] = out[i]

        for key, value in static_param_init_value.items():
            np.testing.assert_allclose(
                value, dy_param_init_value[key], rtol=1e-05
            )

        if core.is_compiled_with_rocm():
            np.testing.assert_allclose(
                static_out, dy_out, rtol=1e-05, atol=0.001
            )
        else:
            np.testing.assert_allclose(static_out, dy_out, rtol=1e-05)

        for key, value in static_param_value.items():
            if core.is_compiled_with_rocm():
                np.testing.assert_allclose(
                    value, dy_param_value[key], rtol=1e-05, atol=0.001
                )
            else:
                np.testing.assert_allclose(
                    value, dy_param_value[key], rtol=1e-05
                )


class TestOptimizerLearningRate(unittest.TestCase):
    def test_constant_lr(self):
        with base.dygraph.guard():
            a = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")

            linear = paddle.nn.Linear(10, 10)

            a = paddle.to_tensor(a)

            b = linear(a)

            loss = paddle.mean(b)

            adam = paddle.optimizer.Adam(0.001, parameters=linear.parameters())

            np.testing.assert_allclose(
                adam.get_lr(), 0.001, rtol=1e-06, atol=0.0
            )

            for i in range(10):
                adam.minimize(loss)
                lr = adam.get_lr()

                np.testing.assert_allclose(lr, 0.001, rtol=1e-06, atol=0.0)

    def test_lr_decay(self):
        with base.dygraph.guard():
            a = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")

            linear = paddle.nn.Linear(10, 10)

            a = paddle.to_tensor(a)

            b = linear(a)

            loss = paddle.mean(b)

            bd = [2, 4, 6, 8]
            value = [0.2, 0.4, 0.6, 0.8, 1.0]

            scheduler = paddle.optimizer.lr.PiecewiseDecay(bd, value)
            adam = paddle.optimizer.Adam(
                scheduler,
                parameters=linear.parameters(),
            )

            np.testing.assert_allclose(adam.get_lr(), 0.2, rtol=1e-06, atol=0.0)

            ret = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
            for i in range(12):
                adam.minimize(loss)
                lr = adam.get_lr()
                adam.step()
                scheduler.step()
                np.testing.assert_allclose(lr, ret[i], rtol=1e-06, atol=0.0)

    def test_lr_decay_natural_exp(self):
        with base.dygraph.guard():
            a = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")

            linear = paddle.nn.Linear(10, 10)

            a = paddle.to_tensor(a)

            b = linear(a)

            loss = paddle.mean(b)
            base_lr = 1.0

            scheduler = paddle.optimizer.lr.NaturalExpDecay(
                learning_rate=base_lr,
                gamma=0.5,
            )
            adam = paddle.optimizer.Adam(
                learning_rate=scheduler,
                parameters=linear.parameters(),
            )

            np.testing.assert_allclose(adam.get_lr(), 1.0, rtol=1e-06, atol=0.0)

            ret = [1.0, 1.0, 1.0, np.exp(-0.5), np.exp(-0.5)]
            counter = 0
            for i in range(5):
                adam.minimize(loss)
                lr = adam.get_lr()
                counter += 1
                if counter % 3 == 0:
                    adam.step()
                    scheduler.step()
                np.testing.assert_allclose(lr, ret[i], rtol=1e-06, atol=0.0)

    def test_set_lr(self):
        with base.dygraph.guard():
            a = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")

            linear = paddle.nn.Linear(10, 10)

            a = paddle.to_tensor(a)

            b = linear(a)

            loss = paddle.mean(b)

            adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

            lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
            for i in range(5):
                adam.set_lr(lr_list[i])
                adam.minimize(loss)
                lr = adam.get_lr()
                np.testing.assert_allclose(lr, lr_list[i], rtol=1e-06, atol=0.0)

            with self.assertRaises(RuntimeError):
                adam = paddle.optimizer.Adam(
                    paddle.optimizer.lr.NaturalExpDecay(
                        learning_rate=0.1,
                        gamma=0.5,
                    ),
                    parameters=linear.parameters(),
                )
                adam.set_lr(0.01)


def exclude_fn(param):
    return param.name.endswith('.b_0')


class TestImperativeDGCMomentumOptimizer(TestImperativeOptimizerBase):
    def get_optimizer_dygraph(self, parameter_list):
        optimizer = DGCMomentumOptimizer(
            learning_rate=0.0001,
            momentum=0.9,
            rampup_step=1000,
            rampup_begin_step=1252,
            sparsity=[0.999, 0.999],
        )
        return optimizer

    def test_dgcmomentum(self):
        exception_message = "In dygraph, don't support DGCMomentumOptimizer."
        self._check_exception(exception_message)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
