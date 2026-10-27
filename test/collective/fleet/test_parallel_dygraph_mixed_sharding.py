# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


class TestMixedShardingParallel(TestMultipleAccelerators):
    def test_mixed_sharding_optimizer(self):
        """MixedShardingOptimizer (Sharding Stage1 V3) on 2 cards.

        Test logic is in hybrid_parallel_sharding_mixed_model.py: a
        MixedInnerOptimizer whose two subs declare opposite ``keep_whole``,
        so one model exercises the V1 (whole tensor on one owner rank) and
        V2 (element-wise reduce-scatter shards) partitions at once, with a
        non-AdamW state layout.
        """
        self.run_mnist_2accelerators(
            'hybrid_parallel_sharding_mixed_model.py',
        )

    def test_mixed_sharding_comm_overlap(self):
        """Same model with sharding_configs.comm_overlap enabled.

        Covers the backward reduce-overlap hook registration path of
        MixedShardingOptimizer (_register_reduce_overlap_hook).
        """
        self.run_mnist_2accelerators(
            'hybrid_parallel_sharding_mixed_model.py',
            need_envs={"COMM_OVERLAP": "1"},
        )

    def test_mixed_sharding_fuse_optimizer_states(self):
        """Same model with enable_fuse_optimizer_states=True.

        Covers MixedShardingOptimizer calling the inner optimizer's
        use_fusion_storage().
        """
        self.run_mnist_2accelerators(
            'hybrid_parallel_sharding_mixed_model.py',
            need_envs={"ENABLE_FUSE_OPTIMIZER_STATES": "1"},
        )


if __name__ == "__main__":
    unittest.main()
