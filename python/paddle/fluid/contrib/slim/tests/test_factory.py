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

from paddle.fluid.contrib.slim.core import ConfigFactory
import unittest


class TestFactory(unittest.TestCase):
    def test_parse_pruning(self):
        factory = ConfigFactory('./configs/filter_pruning.yaml')

        pruner_1 = factory.instance('pruner_1')
        self.assertEquals(pruner_1.pruning_axis['*'], 0)
        self.assertEquals(pruner_1.criterions['*'], 'l1_norm')

        strategy = factory.instance('sensitive_pruning_strategy')
        pruner_1 = strategy.pruner
        self.assertEquals(pruner_1.criterions['*'], 'l1_norm')

        self.assertEquals(strategy.start_epoch, 0)
        self.assertEquals(strategy.sensitivities_file,
                          'mobilenet_acc_top1_sensitive.data')


if __name__ == '__main__':
    unittest.main()
