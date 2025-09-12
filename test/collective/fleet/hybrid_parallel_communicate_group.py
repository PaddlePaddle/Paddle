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

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.base import topology as tp


class TestNewGroupAPI:
    def __init__(self):
        paddle.distributed.init_parallel_env()
        topo = fleet.CommunicateTopology(
            ["data", "sep", "model", "sharding", "pipe"], [2, 1, 1, 1, 1]
        )
        self.hcg = fleet.HybridCommunicateGroup(topo)

        d1 = np.array([1, 2, 3])
        d2 = np.array([2, 3, 4])
        self.tensor1 = paddle.to_tensor(d1)
        self.tensor2 = paddle.to_tensor(d2)

    def test_all(self):
        topo = self.hcg.topology()
        global_rank = self.hcg.get_data_parallel_rank()

        dp_rank = self.hcg.get_data_parallel_rank()
        dp_gp = self.hcg.get_data_parallel_group()
        dp_world_size = self.hcg.get_data_parallel_world_size()
        dp_src_rank = self.hcg.get_data_parallel_group_src_rank()
        np.testing.assert_array_equal(dp_world_size, 2)
        np.testing.assert_array_equal(dp_src_rank, 0)

        mp_rank = self.hcg.get_model_parallel_rank()
        mp_gp = self.hcg.get_model_parallel_group()
        mp_world_size = self.hcg.get_model_parallel_world_size()
        mp_src_rank = self.hcg.get_model_parallel_group_src_rank()
        np.testing.assert_array_equal(mp_world_size, 1)

        tmp = np.array([0, 0, 0])
        result = paddle.to_tensor(tmp)
        paddle.distributed.scatter(
            result,
            [self.tensor2, self.tensor1],
            src=dp_src_rank,
            group=dp_gp,
            sync_op=True,
        )
        if dp_rank == 0:
            np.testing.assert_array_equal(result, self.tensor2)
        elif dp_rank == 1:
            np.testing.assert_array_equal(result, self.tensor1)
        print("test scatter api ok")

        paddle.distributed.broadcast(result, src=1, group=dp_gp, sync_op=True)
        np.testing.assert_array_equal(result, self.tensor1)
        print("test broadcast api ok")

        paddle.distributed.reduce(
            result, dst=dp_src_rank, group=dp_gp, sync_op=True
        )
        if dp_rank == 0:
            np.testing.assert_array_equal(
                result, paddle.add(self.tensor1, self.tensor1)
            )
        elif dp_rank == 1:
            np.testing.assert_array_equal(result, self.tensor1)
        print("test reduce api ok")

        paddle.distributed.all_reduce(result, sync_op=True)
        np.testing.assert_array_equal(
            result,
            paddle.add(paddle.add(self.tensor1, self.tensor1), self.tensor1),
        )
        print("test all_reduce api ok")

        paddle.distributed.wait(result, dp_gp, use_calc_stream=True)
        paddle.distributed.wait(result, dp_gp, use_calc_stream=False)
        print("test wait api ok")

        result = []
        paddle.distributed.all_gather(
            result, self.tensor1, group=dp_gp, sync_op=True
        )
        np.testing.assert_array_equal(result[0], self.tensor1)
        np.testing.assert_array_equal(result[1], self.tensor1)
        print("test all_gather api ok")

        paddle.distributed.barrier(group=dp_gp)
        print("test barrier api ok")


class TestHybridEPGroup:
    def __init__(self):
        paddle.distributed.init_parallel_env()
        group_names = [
            "moe_sharding",
            "sharding",
            "pipe",
            "sep",
            "data",
            "expert",
            "model",
        ]
        dims = [1, 1, 1, 1, 1, 2, 2]

        self.hcg = tp.EPHybridCommunicateGroup(group_names, dims)

    def test_all(self):
        global_rank = paddle.distributed.get_rank()

        dp_rank = self.hcg.get_data_parallel_rank()
        assert dp_rank == 0
        assert self.hcg.get_expert_parallel_world_size() == 2
        assert self.hcg.get_moe_sharding_parallel_world_size() == 1
        assert self.hcg.get_model_parallel_world_size() == 2
        assert self.hcg.get_expert_parallel_rank() == global_rank
        assert self.hcg.get_moe_sharding_parallel_rank() == 0
        assert self.hcg.get_expert_parallel_group_src_rank() == 0
        assert (
            self.hcg.get_moe_sharding_parallel_group_src_rank() == global_rank
        )

        moe_sharding_group = self.hcg.get_moe_sharding_parallel_group()
        ep_group = self.hcg.get_expert_parallel_group()
        mp_group = self.hcg.get_model_parallel_group()
        assert moe_sharding_group.ranks == [global_rank]
        assert ep_group.ranks == [0, 1]
        assert mp_group.ranks == [0, 1]


if __name__ == "__main__":
    gpt = TestNewGroupAPI()
    gpt.test_all()
    ep_test = TestHybridEPGroup()
    ep_test.test_all()
