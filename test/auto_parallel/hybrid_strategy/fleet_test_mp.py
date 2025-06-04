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


def test_mp_parallel():
    # 初始化fleet策略
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 2,
        "pp_degree": 1,
    }
    fleet.init(is_collective=True, strategy=dist_strategy)

    # 创建ProcessMesh
    mesh = dist.ProcessMesh([0, 1], dim_names=["mp"])

    hcg = fleet.get_hybrid_communicate_group()

    # 获取并验证通信组
    group = mesh.get_group(dim_name="mp")
    hcg_group = hcg.get_model_parallel_group()

    # 比较通信组的进程列表
    group_ranks = group.ranks
    hcg_group_ranks = hcg_group.ranks
    assert set(group_ranks) == set(hcg_group_ranks)


if __name__ == "__main__":
    test_mp_parallel()
