# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
import math
import unittest

import gym
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.fluid.dygraph import Layer, to_variable
from paddle.jit import ProgramTranslator
from paddle.jit.api import declarative

SEED = 2020
program_translator = ProgramTranslator()


class Policy(Layer):
    def __init__(self):
        super().__init__()

        self.affine1 = paddle.nn.Linear(4, 128)
        self.affine2 = paddle.nn.Linear(128, 2)
        self.dropout_ratio = 0.6

        self.saved_log_probs = []
        self.rewards = []

    @declarative
    def forward(self, x):
        x = paddle.reshape(x, shape=[1, 4])
        x = self.affine1(x)
        x = paddle.nn.functional.dropout(x, self.dropout_ratio)
        x = F.relu(x)
        action_scores = self.affine2(x)

        log_prob = paddle.nn.functional.softmax(action_scores, axis=1)

        return log_prob


class Args:
    gamma = 0.99
    log_interval = 1
    train_step = 10


def train(args, place, to_static):
    program_translator.enable(to_static)

    env = gym.make('CartPole-v0')
    env.seed(SEED)

    with fluid.dygraph.guard(place):
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        local_random = np.random.RandomState(SEED)

        policy = Policy()

        eps = np.finfo(np.float32).eps.item()
        optimizer = fluid.optimizer.AdamaxOptimizer(
            learning_rate=1e-2, parameter_list=policy.parameters()
        )

        def get_mean_and_std(values=[]):
            n = 0.0
            s = 0.0
            for val in values:
                s += val
                n += 1
            mean = s / n

            std = 0.0
            for val in values:
                std += (val - mean) * (val - mean)
            std /= n
            std = math.sqrt(std)

            return mean, std

        def sample_action(probs):
            sample = local_random.random_sample()
            idx = 0

            while idx < len(probs) and sample > probs[idx]:
                sample -= probs[idx]
                idx += 1
            mask = [0.0] * len(probs)
            mask[idx] = 1.0

            return idx, np.array([mask]).astype("float32")

        def choose_best_action(probs):
            idx = 0 if probs[0] > probs[1] else 1
            mask = [1.0, 0.0] if idx == 0 else [0.0, 1.0]

            return idx, np.array([mask]).astype("float32")

        def select_action(state):
            state = to_variable(state)
            state.stop_gradient = True
            loss_probs = policy(state)

            probs = loss_probs.numpy()

            action, _mask = sample_action(probs[0])
            mask = to_variable(_mask)
            mask.stop_gradient = True

            loss_probs = paddle.log(loss_probs)
            loss_probs = paddle.multiply(loss_probs, mask)
            loss_probs = paddle.sum(loss_probs, axis=-1)

            policy.saved_log_probs.append(loss_probs)
            return action, loss_probs

        def finish_episode():
            R = 0
            policy_loss = []
            returns = []
            for r in policy.rewards[::-1]:
                R = r + args.gamma * R
                returns.insert(0, R)

            mean, std = get_mean_and_std(returns)

            returns = np.array(returns).astype("float32")
            returns = (returns - mean) / (std + eps)

            # calculate policy loss of each step.
            for log_prob, R in zip(policy.saved_log_probs, returns):
                log_prob_numpy = log_prob.numpy()

                R_numpy = np.ones_like(log_prob_numpy).astype("float32")
                _R = -1 * R * R_numpy
                _R = to_variable(_R)
                _R.stop_gradient = True
                cur_loss = paddle.multiply(_R, log_prob)
                policy_loss.append(cur_loss)

            policy_loss = fluid.layers.concat(policy_loss)
            policy_loss = paddle.sum(policy_loss)

            policy_loss.backward()
            optimizer.minimize(policy_loss)
            policy.clear_gradients()

            del policy.rewards[:]
            del policy.saved_log_probs[:]

            return returns

        loss_data = []
        running_reward = 10
        for i_episode in itertools.count(1):
            state, ep_reward = env.reset(), 0
            # The default loop number is 10000 is models, we changed it to 1000 for smaller test
            for t in range(1, 1000):
                state = np.array(state).astype("float32")
                action, loss = select_action(state)
                state, reward, done, _ = env.step(action)

                # log loss_probs
                loss_data.append(loss.numpy()[0])

                policy.rewards.append(reward)
                ep_reward += reward

                if done:
                    break

            # sum loss and apply optimization
            returns = finish_episode()

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            if i_episode % args.log_interval == 0:
                print(
                    'Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\t loss_probs: {}'.format(
                        i_episode, ep_reward, running_reward, loss.numpy()[0]
                    )
                )

            if i_episode > args.train_step:
                break

        return np.array(loss_data)


class TestDeclarative(unittest.TestCase):
    def setUp(self):
        self.place = (
            fluid.CUDAPlace(0)
            if fluid.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )
        self.args = Args()

    def test_train(self):
        st_out = train(self.args, self.place, to_static=True)
        dy_out = train(self.args, self.place, to_static=False)
        np.testing.assert_allclose(st_out, dy_out, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
