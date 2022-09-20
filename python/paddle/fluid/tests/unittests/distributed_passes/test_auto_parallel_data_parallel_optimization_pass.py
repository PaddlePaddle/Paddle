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
from paddle.distributed.fleet import auto
from paddle.distributed.auto_parallel.dist_context import get_default_distributed_context
from paddle.distributed.passes import new_pass, PassManager, PassContext
from auto_parallel_pass_test_base import AutoPallelPassTestBase

sys.path.append("..")
import auto_parallel_gpt_model as modeling
from auto_parallel_gpt_model import GPTModel, GPTForPretraining, GPTPretrainingCriterion


class TestDataParallelPassWithScale1(AutoPallelPassTestBase):

    def init(self):
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        self.rtol = 1e-5
        self.atol = 1e-8
        # NOTE a hack to compare pass apply or not, since there is no
        # setting of this pass in dist_strategy
        self._apply_pass = False

        rank = paddle.distributed.get_rank()
        paddle.seed(rank + 2021)
        random.seed(rank + 2021)
        np.random.seed(rank + 2021)

    def apply_passes(self):
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)
        self._apply_pass = True

    def apply_no_passes(self):
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)
        self._apply_pass = False

    def test_bs_8(self):
        self.check_main(gpus=[0, 1],
                        batch_size=8,
                        sequence_len=512,
                        vocab_size=1000)

    # test scaling with fillconstant
    def get_model(self, place, batch_size, sequence_len, vocab_size):

        dist_main_prog, dist_startup_prog, data_holder, [
            loss
        ], gen_data = self.get_gpt_model('dp', place, batch_size, sequence_len,
                                         vocab_size)
        if self._apply_pass:
            config = {}
            config["dist_context"] = get_default_distributed_context()
            config["global_rank"] = paddle.distributed.get_rank()
            dp_pass = new_pass("auto_parallel_data_parallel_optimization",
                               config)
            dp_pass.apply([dist_main_prog], [dist_startup_prog], PassContext())

        return dist_main_prog, dist_startup_prog, data_holder, [loss], gen_data


class TestDataParallelPassWithScale2(TestDataParallelPassWithScale1):

    # test scaling with optimizer rescale_grad
    def get_model(self, place, batch_size, sequence_len, vocab_size):

        dist_main_prog, dist_startup_prog, data_holder, [
            loss
        ], gen_data = self.get_gpt_model('dp',
                                         place,
                                         batch_size,
                                         sequence_len,
                                         vocab_size,
                                         optimizer='LarsMomentum')
        if self._apply_pass:
            config = {}
            config["dist_context"] = get_default_distributed_context()
            config["global_rank"] = paddle.distributed.get_rank()
            dp_pass = new_pass("auto_parallel_data_parallel_optimization",
                               config)
            dp_pass.apply([dist_main_prog], [dist_startup_prog], PassContext())

        return dist_main_prog, dist_startup_prog, data_holder, [loss], gen_data


if __name__ == "__main__":
    unittest.main()
