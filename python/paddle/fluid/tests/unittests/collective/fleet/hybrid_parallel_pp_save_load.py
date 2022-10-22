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
import os
import shutil
import tempfile
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from hybrid_parallel_pp_transformer import ModelPipe, set_random_seed

batch_size = 8
length = 8
micro_batch_size = 2
vocab_size = 128


class TestDistPPSaveLoadTraning(unittest.TestCase):

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

    def test_pp_model(self):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        topology = hcg.topology()
        set_random_seed(1024, dp_id, rank_id)

        model = ModelPipe(topology)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[2],
                                                       values=[0.001, 0.002],
                                                       verbose=True)
        optimizer = paddle.optimizer.SGD(learning_rate=scheduler,
                                         parameters=model.parameters())

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
        output_dir = tempfile.mkdtemp()

        # warmup step
        for step_id in range(2):
            x_data = np.random.randint(0, vocab_size, size=[batch_size, length])
            x = paddle.to_tensor(x_data)
            x.stop_gradient = True
            loss = model.train_batch([x, x], optimizer, scheduler)

        model._layers.save_state_dict(output_dir)
        paddle.save(optimizer.state_dict(),
                    os.path.join(output_dir, "model_state.pdopt"))

        # construct data
        test_steps = 5
        np_data = np.random.randint(0,
                                    vocab_size,
                                    size=[test_steps, batch_size, length])

        origin_loss = []
        for step_id in range(5):
            x_data = np_data[step_id, :]
            x = paddle.to_tensor(x_data)
            x.stop_gradient = True
            loss = model.train_batch([x, x], optimizer, scheduler)
            origin_loss.append(loss.numpy())

        # test step
        model._layers.set_state_dir(output_dir)
        opt_dict = paddle.load(os.path.join(output_dir, "model_state.pdopt"))
        optimizer.set_state_dict(opt_dict)

        for step_id in range(5):
            x_data = np_data[step_id, :]
            x = paddle.to_tensor(x_data)
            x.stop_gradient = True
            loss = model.train_batch([x, x], optimizer, scheduler)
            print("origin loss: ", origin_loss[step_id], "current loss: ",
                  loss.numpy())
            np.testing.assert_allclose(loss.numpy(), origin_loss[step_id])

        # finally, remove the model/optimizer path
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    unittest.main()
