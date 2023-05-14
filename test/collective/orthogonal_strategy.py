# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.distributed as dist
from paddle.distributed.fleet.base.orthogonal_strategy import OrthogonalStrategy
from paddle.distributed.fleet.base.strategy_group import (
    DPGroup,
    MPGroup,
    PPGroup,
    ShardingGroup,
)


class TestOrthogonalStrategyAPI(unittest.TestCase):
    def setUp(self):
        self._num_of_ranks = 2
        dist.init_parallel_env()
        self._global_rank = dist.get_rank()
        self._strategy = OrthogonalStrategy(
            [
                ("dp", 2, DPGroup),
                ("mp", 1, MPGroup),
                ("sharding", 1, ShardingGroup),
                ("pp", 1, PPGroup),
            ],
            fused_strategy_dict={"checkness": ["mp", "sharding", "pp"]},
        )

    def test_orthogonal_strategy(self):
        dp_group = self._strategy.strategy_group("dp")
        self.assertEqual(dp_group.world_size, self._num_of_ranks)
        self.assertEqual(dp_group.group.nranks, self._num_of_ranks)
        self.assertEqual(
            self._strategy.rank_in_strategy("dp"), self._global_rank
        )

        fused_group = self._strategy.fused_strategy_group("checkness")
        self.assertEqual(fused_group.world_size, 1)
        self.assertEqual(fused_group.group.nranks, 1)


class TestOrthogonalStrategyCustomAPI(unittest.TestCase):
    def setUp(self):
        self._num_of_ranks = 2
        dist.init_parallel_env()
        self._global_rank = dist.get_rank()
        self._strategy = OrthogonalStrategy(
            [
                ("dp", 1, DPGroup),
                ("mp", 2, MPGroup),
                ("sharding", 1, ShardingGroup),
                ("pp", 1, PPGroup),
            ],
            fused_strategy_dict={"checkness": ["mp", "sharding", "pp"]},
            strategy_rank_list=[1, 0],
        )

        self._strategy.strategy_group("mp").add_random_seed("local_seed", 123)
        self._strategy.strategy_group("mp").add_random_seed("global_seed", 321)

    def test_orthogonal_strategy(self):
        mp_group = self._strategy.strategy_group("mp")
        self.assertEqual(mp_group.world_size, self._num_of_ranks)
        self.assertEqual(mp_group.group.nranks, self._num_of_ranks)
        self.assertEqual(
            self._strategy.rank_in_strategy("mp"), self._global_rank
        )

        fused_group = self._strategy.fused_strategy_group("checkness")
        self.assertEqual(fused_group.world_size, 2)
        self.assertEqual(fused_group.group.nranks, 2)

        with mp_group.random_states_tracker.rng_state("local_seed"):
            a = paddle.randint(0, 100, [10]).numpy()[0]


if __name__ == '__main__':
    unittest.main()
