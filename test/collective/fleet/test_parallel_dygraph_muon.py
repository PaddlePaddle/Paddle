# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from legacy_test.test_parallel_dygraph_dataparallel import (
    TestMultipleAccelerators,
)


class TestMuonParallel(TestMultipleAccelerators):
    def test_muon_sharding_optimizer(self):
        """MuonSharding test: iterate ns_coeff_type combinations.

        Test logic is in hybrid_parallel_sharding_muon_model.py,
        iterating 4 ns_coeff_types. fp32 matmul is auto-selected on V100.
        """
        self.run_mnist_2accelerators(
            'hybrid_parallel_sharding_muon_model.py',
            need_envs={"MULTI_PRECISION": "1"},
        )

    def test_muon_sharding_fused_gradient(self):
        """MuonSharding test with FLAGS_shard_fused_gradient=1.

        Covers muon_sharding_optimizer.py L627-635 (comm_buffer_2d reduce)
        and L665-667 (comm_buffer_2d scale_grads).
        """
        self.run_mnist_2accelerators(
            'hybrid_parallel_sharding_muon_model.py',
            need_envs={
                "FLAGS_shard_fused_gradient": "1",
                "MULTI_PRECISION": "1",
            },
        )

    def test_muon_sharding_fuse_optimizer_states(self):
        """MuonSharding test with enable_fuse_optimizer_states=True.

        Covers muon_sharding_optimizer.py L125 (use_fusion_storage).
        """
        self.run_mnist_2accelerators(
            'hybrid_parallel_sharding_muon_model.py',
            need_envs={
                "ENABLE_FUSE_OPTIMIZER_STATES": "1",
                "MULTI_PRECISION": "1",
            },
        )

    def test_muon_sharding_release_grads_fused(self):
        """MuonSharding test with fused gradient + release_gradients.

        Covers muon_sharding_optimizer.py L633-635 (sd_release_grads path
        in fused gradient reduce: copy_grad_to_buffer when grad_storage is None).
        """
        self.run_mnist_2accelerators(
            'hybrid_parallel_sharding_muon_model.py',
            need_envs={
                "FLAGS_shard_fused_gradient": "1",
                "RELEASE_GRADIENTS": "1",
                "MULTI_PRECISION": "1",
            },
        )

    def test_muon_sharding_multi_precision(self):
        """MuonSharding test with multi_precision=True.

        Covers muon.py L575 (master_weight.scale_ with weight_decay),
        L582-583 (master_weight.subtract_ + assign back to param).
        """
        self.run_mnist_2accelerators(
            'hybrid_parallel_sharding_muon_model.py',
            need_envs={"MULTI_PRECISION": "1"},
        )


if __name__ == "__main__":
    unittest.main()
