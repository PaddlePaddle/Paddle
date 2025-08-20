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

import os

import paddle.distributed as dist
from paddle.distributed import fleet


class TestProcessMeshGroupConsistency:
    def __init__(self):
        # Get configuration from environment variables
        self.dp = int(os.getenv("dp", "1"))
        self.mp = int(os.getenv("mp", "1"))
        self.pp = int(os.getenv("pp", "1"))
        self.sep = int(os.getenv("sep", "1"))
        self.sharding = int(os.getenv("sharding", "1"))

        # Determine which parallel type to test
        self.parallel_type = os.getenv("parallel_type", "dp")

    def init_dist_env(self):
        """Initialize distributed environment"""
        # Configure distributed strategy
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.hybrid_configs = {
            "dp_degree": self.dp,
            "mp_degree": self.mp,
            "pp_degree": self.pp,
            "sep_degree": self.sep,
            "sharding_degree": self.sharding,
        }

        # Add corresponding configuration based on parallel type
        if self.sep > 1:
            dist_strategy.hybrid_configs["sep_degree"] = self.sep
        if self.sharding > 1:
            dist_strategy.hybrid_configs["sharding_degree"] = self.sharding

        fleet.init(is_collective=True, strategy=dist_strategy)

    def test_process_mesh_group_consistency(self):
        """Test consistency between ProcessMesh created groups and HCG created groups"""

        # Create corresponding ProcessMesh and get corresponding HCG group based on parallel type
        if self.parallel_type == "dp":
            mesh = dist.ProcessMesh([0, 1], dim_names=["dp"])
            hcg = fleet.get_hybrid_communicate_group()
            group = mesh.get_group(dim_name="dp")
            hcg_group = hcg.get_data_parallel_group()

        elif self.parallel_type == "mp":
            mesh = dist.ProcessMesh([0, 1], dim_names=["mp"])
            hcg = fleet.get_hybrid_communicate_group()
            group = mesh.get_group(dim_name="mp")
            hcg_group = hcg.get_model_parallel_group()

        elif self.parallel_type == "pp":
            mesh = dist.ProcessMesh([0, 1], dim_names=["pp"])
            hcg = fleet.get_hybrid_communicate_group()
            group = mesh.get_group(dim_name="pp")
            hcg_group = hcg.get_pipe_parallel_group()

        elif self.parallel_type == "sep":
            mesh = dist.ProcessMesh([0, 1], dim_names=["sep"])
            hcg = fleet.get_hybrid_communicate_group()
            group = mesh.get_group(dim_name="sep")
            hcg_group = hcg.get_sep_parallel_group()

        elif self.parallel_type == "sharding":
            mesh = dist.ProcessMesh([0, 1], dim_names=["sharding"])
            hcg = fleet.get_hybrid_communicate_group()
            group = mesh.get_group(dim_name="sharding")
            hcg_group = hcg.get_sharding_parallel_group()

        else:
            raise ValueError(f"Unsupported parallel type: {self.parallel_type}")

        # Verify that group ranks are consistent
        group_ranks = group.ranks
        hcg_group_ranks = hcg_group.ranks
        assert set(group_ranks) == set(hcg_group_ranks)

        # Verify that group IDs are consistent
        group_id = group.id
        hcg_group_id = hcg_group.id
        assert group_id == hcg_group_id

    def run_test_cases(self):
        """Run test cases"""
        self.init_dist_env()
        self.test_process_mesh_group_consistency()


if __name__ == "__main__":
    TestProcessMeshGroupConsistency().run_test_cases()
