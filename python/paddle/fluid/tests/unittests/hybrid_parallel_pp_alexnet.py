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

import unittest
import paddle
import numpy as np
import random
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from hybrid_parallel_pp_layer import AlexNetPipeDesc, AlexNet


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


batch_size = 4
micro_batch_size = 2


class TestDistPPTraning(unittest.TestCase):

    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size
        }
        fleet.init(is_collective=True, strategy=strategy)

    def build_optimizer(self, model):
        scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[2],
                                                       values=[0.001, 0.002],
                                                       verbose=True)
        optimizer = paddle.optimizer.SGD(learning_rate=scheduler,
                                         parameters=model.parameters())
        return scheduler, optimizer

    def test_pp_model(self):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        set_random_seed(1024, dp_id, rank_id)

        #construct model a
        model_a = AlexNet(10)
        scheduler_a, optimizer_a = self.build_optimizer(model_a)

        param_len = len(model_a.parameters())

        parameters = []
        for param in model_a.parameters():
            parameters.append(param.numpy())

        # construct model b
        model_b = AlexNetPipeDesc(num_stages=self.pipeline_parallel_size)
        scheduler_b, optimizer_b = self.build_optimizer(model_b)
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b)

        for idx, param in enumerate(model_b.parameters()):
            param.set_value(parameters[idx + pp_id * (param_len // 2)])

        # construct reader
        train_reader = paddle.batch(paddle.dataset.mnist.train(),
                                    batch_size=batch_size,
                                    drop_last=True)

        for step_id, data in enumerate(train_reader()):
            x_data = np.array([x[0] for x in data]).astype('float32').reshape(
                batch_size, 1, 28, 28)
            y_data = np.array([x[1] for x in data
                               ]).astype('int64').reshape(batch_size, 1)
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            img.stop_gradient = True
            label.stop_gradient = True

            if step_id >= 5:
                return True

            loss_a = model_a(img, label)
            loss_a.backward()
            optimizer_a.step()
            optimizer_a.clear_grad()
            scheduler_a.step()

            loss_b = model_b.train_batch([img, label], optimizer_b, scheduler_b)

            print("loss: ", loss_a.numpy(), loss_b.numpy())
            np.testing.assert_allclose(loss_a.numpy(),
                                       loss_b.numpy(),
                                       rtol=5e-5)


if __name__ == "__main__":
    unittest.main()
