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

"""
Launcher for VMM V2 distributed pool routing test.

Spawns 2 GPU processes running a real AMP O2 + Stage2/Stage3 distributed
training flow with VMM V2 allocator enabled, verifying that every tensor
category (params, master weights, optimizer states, gradients) routes to
the correct VMM V2 memory pool.
"""

import unittest

from legacy_test.test_parallel_dygraph_dataparallel import (
    TestMultipleAccelerators,
)


class TestVMMV2ShardingPoolRouting(TestMultipleAccelerators):
    def test_vmm_v2_sharding_stage2(self):
        self.run_mnist_2accelerators(
            'dygraph_vmm_v2_sharding_pool_routing.py',
            need_envs={
                "FLAGS_use_vmm_auto_growth_best_fit_allocator_v2": "1",
                "VMM_V2_TEST_STAGE": "2",
            },
        )

    def test_vmm_v2_sharding_stage3(self):
        self.run_mnist_2accelerators(
            'dygraph_vmm_v2_sharding_pool_routing.py',
            need_envs={
                "FLAGS_use_vmm_auto_growth_best_fit_allocator_v2": "1",
                "VMM_V2_TEST_STAGE": "3",
            },
        )


if __name__ == "__main__":
    unittest.main()
