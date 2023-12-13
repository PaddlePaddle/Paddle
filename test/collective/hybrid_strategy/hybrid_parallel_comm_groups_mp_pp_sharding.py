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

import numpy as np

import paddle
from paddle.distributed import fleet


class TestHybridParallelNewGroup:
    def __init__(self):
        paddle.distributed.init_parallel_env()
        topo = fleet.CommunicateTopology(
            ["data", "pipe", "sharding", "sep", "model"], [1, 2, 2, 1, 2]
        )
        self.hcg = fleet.HybridCommunicateGroup(topo)

        d1 = np.array([1, 2, 3])
        d2 = np.array([2, 3, 4])
        self.tensor1 = paddle.to_tensor(d1)
        self.tensor2 = paddle.to_tensor(d2)

    def test_all(self):
        global_rank = self.hcg.get_global_rank()

        dp_rank = self.hcg.get_data_parallel_rank()
        dp_world_size = self.hcg.get_data_parallel_world_size()
        np.testing.assert_array_equal(dp_rank, 0)
        np.testing.assert_array_equal(dp_world_size, 1)

        mp_rank = self.hcg.get_model_parallel_rank()
        mp_world_size = self.hcg.get_model_parallel_world_size()
        np.testing.assert_array_equal(mp_rank, global_rank % 2)
        np.testing.assert_array_equal(mp_world_size, 2)

        pp_rank = self.hcg.get_stage_id()
        pp_world_size = self.hcg.get_pipe_parallel_world_size()
        np.testing.assert_array_equal(pp_rank, global_rank // 4)
        np.testing.assert_array_equal(pp_world_size, 2)

        sharding_rank = self.hcg.get_sharding_parallel_rank()
        sharding_world_size = self.hcg.get_sharding_parallel_world_size()
        np.testing.assert_array_equal(sharding_world_size, 2)
        np.testing.assert_array_equal(sharding_rank, (global_rank // 2) % 2)

        paddle.distributed.barrier()
        print("test barrier api in default group ok")


if __name__ == "__main__":
    gpt = TestHybridParallelNewGroup()
    gpt.test_all()
