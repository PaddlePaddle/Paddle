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
from paddle.incubate.distributed.fleet.collective import (
    CollectiveOptimizer,
    DistributedStrategy,
)


class CollectiveOptimizerTest(unittest.TestCase):
    def test_ds_as_None(self):
        optimizer = paddle.optimizer.Adam()
        dist_optimizer = CollectiveOptimizer(optimizer, strategy=None)

    def test_recompute_checkpoints(self):
        optimizer = paddle.optimizer.Adam()
        dist_strategy = DistributedStrategy()
        dist_strategy.forward_recompute = True
        dist_strategy.recompute_checkpoints = "NoneListTest"
        self.assertRaises(
            ValueError, CollectiveOptimizer, optimizer, dist_strategy
        )
        dist_strategy.recompute_checkpoints = []
        dist_optimizer = CollectiveOptimizer(optimizer, dist_strategy)
        self.assertRaises(ValueError, dist_optimizer.minimize, None)

    def test_recompute_strategy(self):
        optimizer = paddle.optimizer.Adam()
        optimizer = paddle.incubate.optimizer.RecomputeOptimizer(optimizer)
        dist_strategy = DistributedStrategy()
        dist_strategy.forward_recompute = True
        dist_strategy.recompute_checkpoints = ["Test"]
        dist_optimizer = CollectiveOptimizer(optimizer, strategy=dist_strategy)
        self.assertRaises(ValueError, dist_optimizer.minimize, None)

    def test_amp_strategy(self):
        optimizer = paddle.optimizer.Adam()
        optimizer = paddle.static.amp.decorate(
            optimizer, init_loss_scaling=1.0, use_dynamic_loss_scaling=True
        )
        dist_strategy = DistributedStrategy()
        dist_strategy.use_amp = True
        dist_optimizer = CollectiveOptimizer(optimizer, strategy=dist_strategy)
        self.assertRaises(ValueError, dist_optimizer.minimize, None)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
