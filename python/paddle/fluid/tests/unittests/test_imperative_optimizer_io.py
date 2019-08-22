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

from __future__ import print_function

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


class MLP(fluid.Layer):
    def __init__(self, name_scope):
        super(MLP, self).__init__(name_scope)
        self._fc1 = FC(self.full_name(), 10)
        self._fc2 = FC(self.full_name(), 10)

    def forward(self, inputs):
        y = self._fc1(inputs)
        y = self._fc2(y)
        return y


class TestOptimizerIO(unittest.TestCase):
    def setUp(self):
        self.save_dir = "optimizer"
        self.train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=128, drop_last=True)
        self.save_filenames = [
            'sgd', 'moment', 'dgc_moment', 'lars_moment', 'adagrad', 'adam',
            'decay_adagrad', 'adadelta', 'rms', 'ftrl', 'lamb'
        ]

    def test_optmizer_io(self):
        with fluid.dygraph.guard():
            mlp = MLP("mlp")
            sgd = SGDOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                learning_rate=0.1,
                decay_steps=1,
                decay_rate=0.5,
                staircase=True))
            moment = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.9, use_nesterov=True)
            dgc_moment = fluid.optimizer.DGCMomentumOptimizer(
                learning_rate=0.0001,
                momentum=0.9,
                rampup_step=1000,
                rampup_begin_step=1252,
                sparsity=[0.999, 0.999])
            lars_moment = fluid.optimizer.LarsMomentum(
                learning_rate=0.2, momentum=0.1, lars_weight_decay=0.001)
            adagrad = fluid.optimizer.Adagrad(learning_rate=0.2)
            adam = fluid.optimizer.AdamOptimizer(0.01)
            decay_adagrad = fluid.optimizer.DecayedAdagrad(learning_rate=0.2)
            adadelta = fluid.optimizer.Adadelta(
                learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)
            rms = fluid.optimizer.RMSProp(learning_rate=0.1)
            ftrl = fluid.optimizer.Ftrl(learning_rate=0.1)
            lamb = fluid.optimizer.Lamb(learning_rate=0.002)

            optimizers = [
                sgd, moment, dgc_moment, lars_moment, adagrad, adam,
                decay_adagrad, adadelta, rms, ftrl, lamb
            ]

            for batch_id, data in enumerate(self.train_reader()):
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

                optimizers_state = []
                for i, optimizer in enumerate(optimizers):
                    optimizer.minimize(avg_loss)
                    optimizer.save(self.save_dir, self.save_filenames[i])
                    optimizers_state.append(optimizer.state_dict())
                break

            sgd_loaded = SGDOptimizer(
                learning_rate=fluid.layers.natural_exp_decay(
                    learning_rate=random.uniform(0.001, 1.0),
                    decay_steps=random.randint(0, 9),
                    decay_rate=random.uniform(0.001, 1.0),
                    staircase=random.choice([True, False])))
            moment_loaded = fluid.optimizer.MomentumOptimizer(
                learning_rate=random.uniform(0.001, 1.0),
                momentum=random.uniform(0.001, 1.0),
                use_nesterov=True)
            dgc_moment_loaded = fluid.optimizer.DGCMomentumOptimizer(
                learning_rate=random.uniform(0.001, 1.0),
                momentum=random.uniform(0.001, 1.0),
                rampup_step=random.randint(0, 1000),
                rampup_begin_step=random.randint(0, 1000),
                sparsity=[
                    random.uniform(0.001, 1.0), random.uniform(0.001, 1.0)
                ])
            lars_moment_loaded = fluid.optimizer.LarsMomentum(
                learning_rate=random.uniform(0.001, 1.0),
                momentum=random.uniform(0.001, 1.0),
                lars_weight_decay=random.uniform(0.001, 1.0))
            adagrad_loaded = fluid.optimizer.Adagrad(
                learning_rate=random.uniform(0.001, 1.0))
            adam_loaded = fluid.optimizer.AdamOptimizer(
                random.uniform(0.001, 1.0))
            decay_adagrad_loaded = fluid.optimizer.DecayedAdagrad(
                learning_rate=random.uniform(0.001, 1.0))
            adadelta_loaded = fluid.optimizer.Adadelta(
                learning_rate=random.uniform(0.001, 1.0),
                epsilon=random.uniform(0.001, 1.0),
                rho=random.uniform(0.001, 1.0))
            rms_loaded = fluid.optimizer.RMSProp(learning_rate=random.uniform(
                0.001, 1.0))
            ftrl_loaded = fluid.optimizer.Ftrl(learning_rate=random.uniform(
                0.001, 1.0))
            lamb_loaded = fluid.optimizer.Lamb(learning_rate=random.uniform(
                0.001, 1.0))

            optimizers_loaded = [
                sgd_loaded, moment_loaded, dgc_moment_loaded,
                lars_moment_loaded, adagrad_loaded, adam_loaded,
                decay_adagrad_loaded, adadelta_loaded, rms_loaded, ftrl_loaded,
                lamb_loaded
            ]
            optimizers_loaded_state = []

            for batch_id, data in enumerate(self.train_reader()):
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

                for i, optimizer_loaded in enumerate(optimizers_loaded):
                    optimizer_loaded.load_from_path(
                        os.path.join("./", self.save_dir, self.save_filenames[
                            i]))
                    optimizer_loaded.minimize(avg_loss)
                    optimizers_loaded_state.append(optimizer_loaded.state_dict(
                    ))
                break

            def compare_approximate(first, second):
                if first.keys() != second.keys():
                    return False
                    for key1 in first:
                        if first[key1].keys() != second[key1].keys():
                            return False
                        for key2 in first[key1].keys():
                            if not np.allclose(first[key1][key2],
                                               second[key1][key2]):
                                return False
                return True

            def compare_properties(first, second):
                if isinstance(first['_learning_rate'], LearningRateDecay):
                    # Since the loaded optimizer performs the minimize operation after loading state dict, 
                    # the current step_num has increased by 1 step
                    first['_learning_rate'].__dict__['step_num'] += 1
                    self.assertTrue(first['_learning_rate'].__dict__ ==
                                    second['_learning_rate'].__dict__)
                else:
                    self.assertTrue(
                        np.allclose(first['_learning_rate'], second[
                            '_learning_rate']))
                first.pop('_learning_rate')
                second.pop('_learning_rate')
                return first == second

            for i in range(len(optimizers_state)):
                self.assertTrue(
                    compare_properties(optimizers_state[i]['properties'],
                                       optimizers_loaded_state[i][
                                           'properties']))
                self.assertTrue(
                    compare_approximate(optimizers_state[i][
                        'accumulators_state'], optimizers_loaded_state[i][
                            'accumulators_state']))

    def test_optmizer_io_exception(self):
        with fluid.dygraph.guard():
            mlp = MLP("mlp")
            sgd = SGDOptimizer(learning_rate=0.001)
            for batch_id, data in enumerate(self.train_reader()):
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
                optimizers_state = []
                sgd.minimize(avg_loss)
                sgd.save(self.save_dir, "saved_sgd")
                break

            moment_loaded = fluid.optimizer.MomentumOptimizer(
                learning_rate=random.uniform(0.001, 1.0),
                momentum=random.uniform(0.001, 1.0),
                use_nesterov=True)
            self.assertRaises(TypeError, moment_loaded.load_from_path,
                              os.path.join(self.save_dir, "saved_sgd"))

    def test_io_in_static_exception(self):
        sgd = SGDOptimizer(learning_rate=fluid.layers.natural_exp_decay(
            learning_rate=0.1, decay_steps=1, decay_rate=0.5, staircase=True))
        moment = fluid.optimizer.MomentumOptimizer(
            learning_rate=0.001, momentum=0.9, use_nesterov=True)
        dgc_moment = fluid.optimizer.DGCMomentumOptimizer(
            learning_rate=0.0001,
            momentum=0.9,
            rampup_step=1000,
            rampup_begin_step=1252,
            sparsity=[0.999, 0.999])
        lars_moment = fluid.optimizer.LarsMomentum(
            learning_rate=0.2, momentum=0.1, lars_weight_decay=0.001)
        adagrad = fluid.optimizer.Adagrad(learning_rate=0.2)
        adam = fluid.optimizer.AdamOptimizer(0.01)
        decay_adagrad = fluid.optimizer.DecayedAdagrad(learning_rate=0.2)
        adadelta = fluid.optimizer.Adadelta(
            learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)
        rms = fluid.optimizer.RMSProp(learning_rate=0.1)
        ftrl = fluid.optimizer.Ftrl(learning_rate=0.1)
        lamb = fluid.optimizer.Lamb(learning_rate=0.002)

        optimizers = [
            sgd, moment, dgc_moment, lars_moment, adagrad, adam, decay_adagrad,
            adadelta, rms, ftrl, lamb
        ]

        for i, optimizer in enumerate(optimizers):
            self.assertRaises(TypeError, optimizer.save, self.save_dir,
                              self.save_filenames[i])
            self.assertRaises(TypeError, optimizer.load_from_path,
                              os.path.join(self.save_dir, "random_file_name"))
            self.assertRaises(TypeError, optimizer.load_state_dict, dict())


if __name__ == '__main__':
    unittest.main()
