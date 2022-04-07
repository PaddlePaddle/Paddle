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
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import paddle.fluid.dygraph.nn as nn
from paddle.fluid.dygraph.base import to_variable
from test_imperative_base import new_program_scope
from paddle.fluid.framework import _test_eager_guard


class Policy(fluid.dygraph.Layer):
    def __init__(self, input_size):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(input_size, 128)
        self.affine2 = nn.Linear(128, 2)
        self.dropout_ratio = 0.6

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, inputs):
        x = fluid.layers.reshape(inputs, shape=[-1, 4])
        x = self.affine1(x)
        x = fluid.layers.dropout(x, self.dropout_ratio)
        x = fluid.layers.relu(x)
        action_scores = self.affine2(x)
        return fluid.layers.softmax(action_scores, axis=1)


class TestImperativeMnist(unittest.TestCase):
    def test_mnist_float32(self):
        seed = 90
        epoch_num = 1

        state = np.random.normal(size=4).astype("float32")
        state_list = state.tolist()
        reward = np.random.random(size=[1, 1]).astype("float32")
        reward_list = reward.tolist()
        action_list = [1]
        action = np.array(action_list).astype("float32")
        mask_list = [[0, 1]]
        mask = np.array(mask_list).astype("float32")

        def run_dygraph():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            policy = Policy(input_size=4)

            dy_state = fluid.dygraph.base.to_variable(state)
            dy_state.stop_gradient = True
            loss_probs = policy(dy_state)

            dy_mask = fluid.dygraph.base.to_variable(mask)
            dy_mask.stop_gradient = True

            loss_probs = fluid.layers.log(loss_probs)
            loss_probs = fluid.layers.elementwise_mul(loss_probs, dy_mask)
            loss_probs = fluid.layers.reduce_sum(loss_probs, dim=-1)

            dy_reward = fluid.dygraph.base.to_variable(reward)
            dy_reward.stop_gradient = True

            loss_probs = fluid.layers.elementwise_mul(dy_reward, loss_probs)
            loss = fluid.layers.reduce_sum(loss_probs)

            sgd = SGDOptimizer(
                learning_rate=1e-3, parameter_list=policy.parameters())

            dy_param_init_value = {}

            dy_out = loss.numpy()

            for param in policy.parameters():
                dy_param_init_value[param.name] = param.numpy()

            loss.backward()
            sgd.minimize(loss)
            policy.clear_gradients()

            dy_param_value = {}
            for param in policy.parameters():
                dy_param_value[param.name] = param.numpy()

            return dy_out, dy_param_init_value, dy_param_value

        with fluid.dygraph.guard():
            dy_out, dy_param_init_value, dy_param_value = run_dygraph()

        with fluid.dygraph.guard():
            with _test_eager_guard():
                eager_out, eager_param_init_value, eager_param_value = run_dygraph(
                )

        with new_program_scope():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))

            policy = Policy(input_size=4)

            st_sgd = SGDOptimizer(learning_rate=1e-3)

            st_state = fluid.layers.data(
                name='st_state', shape=[4], dtype='float32')
            st_reward = fluid.layers.data(
                name='st_reward', shape=[1], dtype='float32')
            st_mask = fluid.layers.data(
                name='st_mask', shape=[2], dtype='float32')

            st_loss_probs = policy(st_state)

            st_loss_probs = fluid.layers.log(st_loss_probs)
            st_loss_probs = fluid.layers.elementwise_mul(st_loss_probs, st_mask)
            st_loss_probs = fluid.layers.reduce_sum(st_loss_probs, dim=-1)

            st_loss_probs = fluid.layers.elementwise_mul(st_reward,
                                                         st_loss_probs)
            st_loss = fluid.layers.reduce_sum(st_loss_probs)

            st_sgd.minimize(st_loss)

            # initialize params and fetch them
            static_param_init_value = {}
            static_param_name_list = []
            for param in policy.parameters():
                static_param_name_list.append(param.name)

            out = exe.run(fluid.default_startup_program(),
                          fetch_list=static_param_name_list)

            for i in range(len(static_param_name_list)):
                static_param_init_value[static_param_name_list[i]] = out[i]

            fetch_list = [st_loss.name]
            fetch_list.extend(static_param_name_list)

            out = exe.run(
                fluid.default_main_program(),
                feed={"st_state": state,
                      "st_reward": reward,
                      "st_mask": mask},
                fetch_list=fetch_list)

            static_param_value = {}
            static_out = out[0]
            for i in range(1, len(out)):
                static_param_value[static_param_name_list[i - 1]] = out[i]

        #self.assertTrue(np.allclose(dy_x_data.all(), static_x_data.all()))

        for key, value in six.iteritems(static_param_init_value):
            self.assertTrue(np.equal(value, dy_param_init_value[key]).all())

        self.assertTrue(np.equal(static_out, dy_out).all())

        for key, value in six.iteritems(static_param_value):
            self.assertTrue(np.equal(value, dy_param_value[key]).all())

        # check eager
        for key, value in six.iteritems(static_param_init_value):
            self.assertTrue(np.equal(value, eager_param_init_value[key]).all())

        self.assertTrue(np.equal(static_out, eager_out).all())

        for key, value in six.iteritems(static_param_value):
            self.assertTrue(np.equal(value, eager_param_value[key]).all())


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
