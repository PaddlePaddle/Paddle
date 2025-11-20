# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


import paddle
from paddle.distributed.fleet.base import topology as tp


class TestHybridCPGroup:
    def __init__(self):
        paddle.distributed.init_parallel_env()
        group_names = [
            "moe_sharding",
            "sharding",
            "pipe",
            "sep",
            "data",
            "expert",
            "context",
            "model",
        ]
        dims = [1, 4, 1, 1, 1, 4, 4, 1]

        self.hcg = tp.EPHybridCommunicateGroup(group_names, dims)

    def test_all(self):
        global_rank = paddle.distributed.get_rank()

        dp_rank = self.hcg.get_data_parallel_rank()
        assert dp_rank == 0
        assert self.hcg.get_expert_parallel_world_size() == 4
        assert self.hcg.get_moe_sharding_parallel_world_size() == 1
        assert self.hcg.get_model_parallel_world_size() == 1
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
        assert ep_group.ranks == [0, 1, 2, 3]
        assert mp_group.ranks == [global_rank]

        assert self.hcg.get_context_parallel_rank() == global_rank
        assert self.hcg.get_context_parallel_world_size() == 4
        cp_group = self.hcg.get_context_parallel_group()
        assert cp_group.ranks == [0, 1, 2, 3]
        assert self.hcg.get_context_parallel_group_src_rank() == 0
        cp_sharding_group = self.hcg.get_cp_sharding_parallel_group()
        assert cp_sharding_group.ranks == [global_rank]
        assert self.hcg.get_cp_sharding_parallel_group_src_rank() == global_rank
        cp_mp_group = self.hcg.get_cp_mp_parallel_group()
        assert cp_mp_group.ranks == [0, 1, 2, 3]
        assert self.hcg.get_cp_mp_parallel_group_src_rank() == 0
        assert self.hcg.get_sharding_parallel_world_size() == 4
        assert (
            self.hcg.get_sharding_parallel_world_size(
                with_context_parallel=True
            )
            == 1
        )
        assert self.hcg.get_sharding_parallel_rank() == global_rank
        assert (
            self.hcg.get_sharding_parallel_rank(with_context_parallel=True) == 0
        )


if __name__ == "__main__":
    cp_test = TestHybridCPGroup()
    cp_test.test_all()
