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
#
# Launcher: runs hybrid_parallel_sharding_muon_no_hook_model.py on 2 GPUs to
# verify that the MuonShardingOptimizer "no-hook shared color" sync-only path
# under comm_overlap is bit-for-bit identical to the non-overlap baseline.

import unittest

from legacy_test.test_parallel_dygraph_dataparallel import (
    TestMultipleAccelerators,
)


class TestMuonNoHookOverlap(TestMultipleAccelerators):
    def test_muon_no_hook_overlap_matches_baseline(self):
        self.run_mnist_2accelerators(
            "hybrid_parallel_sharding_muon_no_hook_model.py",
        )


if __name__ == "__main__":
    unittest.main()
