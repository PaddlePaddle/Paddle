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
from paddle import fluid
import os
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
from fleet_meta_optimizer_base import TestFleetMetaOptimizer

paddle.enable_static()


class TestFleetDGCOptimizer(TestFleetMetaOptimizer):
    def test_dgc_optimizer(self):
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'dgc')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('dgc', ops)
        self.assertIn('dgc_momentum', ops)

    def test_dgc_not_apply_with_adam(self):
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'dgc')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog, 'adam')

        ops = [op.type for op in avg_cost.block.ops]
        self.assertNotIn('dgc', ops)
        self.assertNotIn('dgc_momentum', ops)

    def test_dgc_not_apply_with_one_worker(self):
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"

        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'dgc')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        self.assertNotIn('dgc', ops)
        self.assertNotIn('dgc_momentum', ops)

    def test_dgc_recompute_optimizer(self):
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'dgc')
        self.set_strategy(strategy, 'recompute')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        outs = [
            op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul'
        ]
        self.assertIn('dgc', ops)
        self.assertIn('dgc_momentum', ops)

        self.assertIn('subprog', ''.join(outs))


if __name__ == "__main__":
    unittest.main()
