# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import random
import numpy as np

import unittest
import paddle
import paddle.nn as nn
import paddle.distributed.fleet as fleet
import paddle.distributed.auto_parallel as auto
from paddle.distributed.passes import new_pass, PassManager
from auto_parallel_pass_test_base import AutoPallelPassTestBase
sys.path.append("..")
import auto_parallel_gpt_model as modeling
from auto_parallel_gpt_model import GPTModel, GPTForPretraining, GPTPretrainingCriterion


class TestShardingPass(AutoPallelPassTestBase):
    def init(self):
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        self.rtol = 1e-5
        self.atol = 1e-8

        rank = paddle.distributed.get_rank()
        paddle.seed(rank + 2021)
        random.seed(rank + 2021)
        np.random.seed(rank + 2021)

    def apply_passes(self):
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        dist_strategy.sharding = True
        dist_strategy.sharding_configs = {
            "sharding_degree": 2,
            "stage": 2,
        }
        fleet.init(is_collective=True, strategy=dist_strategy)

    def apply_no_passes(self):
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.pipeline = False
        dist_strategy.recompute = False
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)

    def test_bs_8(self):
        self.check_main(
            gpus=[0, 1], batch_size=8, sequence_len=512, vocab_size=1000)

    def get_model(self, place, batch_size, sequence_len, vocab_size):
        return self.get_gpt_model('dp', place, batch_size, sequence_len,
                                  vocab_size)


if __name__ == "__main__":
    unittest.main()
