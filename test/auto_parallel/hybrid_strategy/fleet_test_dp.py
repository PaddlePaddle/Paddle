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

import paddle.distributed as dist
from paddle.distributed import fleet


def test_dp_parallel():
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.hybrid_configs = {
        "dp_degree": 2,
        "mp_degree": 1,
        "pp_degree": 1,
    }
    fleet.init(is_collective=True, strategy=dist_strategy)

    mesh = dist.ProcessMesh([0, 1], dim_names=["dp"])

    hcg = fleet.get_hybrid_communicate_group()

    group = mesh.get_group(dim_name="dp")
    hcg_group = hcg.get_data_parallel_group()

    group_ranks = group.ranks
    hcg_group_ranks = hcg_group.ranks
    assert set(group_ranks) == set(hcg_group_ranks)
    group_id = group.id
    hcg_group_id = hcg_group.id
    assert group_id == hcg_group_id


if __name__ == "__main__":
    test_dp_parallel()
