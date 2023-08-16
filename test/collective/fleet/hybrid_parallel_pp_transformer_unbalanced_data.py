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

import numpy as np
from hybrid_parallel_pp_transformer import (
    ModelPipe,
    TestDistPPTraining,
    batch_size,
    length,
    micro_batch_size,
    set_random_seed,
    vocab_size,
)

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet


class TestDistPPTrainingUnbalancedData(TestDistPPTraining):
    def test_pp_model(self):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        topology = hcg.topology()
        set_random_seed(1024, dp_id, rank_id)

        model = ModelPipe(topology)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=scheduler, parameters=model.parameters()
        )

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

        for step_id in range(5):
            x = []
            for _ in range(batch_size // micro_batch_size):
                size = micro_batch_size
                x_data = np.random.randint(0, vocab_size, size=[size, length])
                x.append(paddle.to_tensor(x_data))
            e_loss = model.eval_batch([x, x], True)
            loss = model.train_batch([x, x], optimizer, scheduler)

            # TODO(shenliang03) add utest for loss
            if pp_id != 0:
                np.testing.assert_allclose(loss.numpy(), e_loss.numpy())


if __name__ == "__main__":
    unittest.main()
