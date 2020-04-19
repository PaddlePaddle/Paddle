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
"""The search space used to search neural architecture"""

__all__ = ['SearchSpace']


class SearchSpace(object):
    """Controller for Neural Architecture Search.
    """

    def __init__(self, *args, **kwargs):
        pass

    def init_tokens(self):
        """Get init tokens in search space.
        """
        raise NotImplementedError('Abstract method.')

    def range_table(self):
        """Get range table of current search space.
        """
        raise NotImplementedError('Abstract method.')

    def create_net(self, tokens):
        """Create networks for training and evaluation according to tokens.
        Args:
            tokens(list<int>): The tokens which represent a network.
        Return:
            (tuple): startup_program, train_program, evaluation_program, train_metrics, test_metrics
        """
        raise NotImplementedError('Abstract method.')

    def get_model_latency(self, program):
        """Get model latency according to program.
        Args:
            program(Program): The program to get latency.
        Return:
            (float): model latency.
        """
        raise NotImplementedError('Abstract method.')
