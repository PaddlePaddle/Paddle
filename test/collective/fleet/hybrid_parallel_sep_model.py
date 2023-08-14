# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

import numpy as np

import paddle
from paddle.distributed import fleet


class TestDistMPTraining(unittest.TestCase):
    def setUp(self):
        random.seed(2023)
        np.random.seed(2023)
        paddle.seed(2023)

        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": 1,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sep_degree": 2,
        }
        fleet.init(is_collective=True, strategy=self.strategy)

    def test_basic_hcg(self):
        hcg = fleet.get_hybrid_communicate_group()
        assert hcg.get_sep_parallel_rank() >= 0
        assert hcg.get_sep_parallel_world_size() == 2
        assert hcg.get_sep_parallel_group_src_rank() == 0
        assert hcg.get_sep_parallel_group() is not None
        assert hcg.get_dp_sep_parallel_group() is not None
        assert hcg.get_pp_mp_parallel_group() is not None


if __name__ == "__main__":
    unittest.main()
