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

from paddle.fluid.contrib.slim import ConfigFactory
import unittest


class TestFactory(unittest.TestCase):
    def test_parse(self):
        factory = ConfigFactory('./configs/config.yaml')

        pruner = factory.instance('pruner_1')
        self.assertEquals(pruner.ratios['conv1_1.w'], 0.3)

        pruner = factory.instance('pruner_2')
        self.assertEquals(pruner.ratios['*'], 0.7)

        strategy = factory.instance('strategy_1')
        pruner = strategy.pruner
        self.assertEquals(pruner.ratios['*'], 0.7)

        compress_pass = factory.get_compress_pass()
        self.assertEquals(compress_pass.epoch, 100)

        strategy = compress_pass.strategies[0]
        self.assertEquals(strategy.delta_rate, 0.2)


if __name__ == '__main__':
    unittest.main()
