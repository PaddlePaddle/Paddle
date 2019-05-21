#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
"""The controller used to search hyperparameters or neural architecture"""

__all__ = ['EvolutionaryController', 'SAController']


class EvolutionaryController(object):
    """Controller for Neural Architecture Search.
    """

    def __init__(self, *args, **kwargs):
        pass

    def update(self, tokens, reward):
        """Update the status of controller according current tokens and reward.
        """
        raise NotImplementedError('Abstract method.')

    def reset(self, range_table, constrain_func=None):
        """Reset the controller.
        """
        raise NotImplementedError('Abstract method.')

    def next_tokens(self):
        """Generate new tokens.
        """
        raise NotImplementedError('Abstract method.')


class SAController(EvolutionaryController):
    """Simulated annealing controller."""

    def __init__(self,
                 range_table,
                 reduce_rate=0.85,
                 init_temperature=1024,
                 max_iter_number=300):
        """Initialize.
        Args:
            range_table(list): variable range table.
            reduce_rate(float): reduce rate.
            init_temperature(float): init temperature.
            max_iter_number(int): max iteration number.
        """
        super(SAController, self).__init__()
        self._range_table = range_table
        self._reduce_rate = reduce_rate
        self._init_temperature = init_temperature
        self._max_iter_number = max_iter_number
        self._reward = -1
        self._tokens = None
        self._max_reward = -1
        self._best_tokens = None
        self._iter = 0

    def reset(self, range_table, constrain_func=None):
        self._range_table = range_table
        self._constrain_func = constrain_func
        self._iter = 0

    def update(self, tokens, reward):
        self._iter += 1
        temperature = self._init_temperature * self._reduce_rate**self._iter
        if (reward > self._reward) or (np.random.random() <= math.exp(
            (reward - self._reward) / temperature)):
            self._reward = reward
            self._tokens = tokens
        if reward > self._max_reward:
            self._max_reward = reward
            self._best_tokens = tokens

    def next_tokens(self, tokens):
        new_tokens = tokens[:]
        index = int(len(self._range_table) * np.random.random())
        new_tokens[index] = self._range_table[index] * np.random.random()
        if _constrain_func is None:
            return new_tokens
        for _ in range(self._max_iter_number):
            if not self._constrain_func(new_tokens):
                index = int(len(self._range_table) * np.random.random())
                new_tokens[index] = self._range_table[index] * np.random.random(
                )
            else:
                break
        return new_tokens
