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

import unittest
import paddle
import paddle.fluid as fluid
import os
from fleet_meta_optimizer_base import TestFleetMetaOptimizer

paddle.enable_static()


class TestFleetRecomputeMetaOptimizer(TestFleetMetaOptimizer):
    def test_recompute_optimizer(self):
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'recompute')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        outs = [
            op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul'
        ]

        self.assertIn('subprog', ''.join(outs))

    def test_recompute_lars_optimizer(self):
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'recompute')
        self.set_strategy(strategy, 'lars')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        outs = [
            op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul'
        ]

        self.assertIn('lars_momentum', ops)
        self.assertIn('subprog', ''.join(outs))

    def test_recompute_lamb_optimizer(self):
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'recompute')
        self.set_strategy(strategy, 'lamb')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog, 'adam')

        ops = [op.type for op in avg_cost.block.ops]
        outs = [
            op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul'
        ]

        self.assertIn('lamb', ops)
        self.assertIn('subprog', ''.join(outs))


if __name__ == "__main__":
    unittest.main()
