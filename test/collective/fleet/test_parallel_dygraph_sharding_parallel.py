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

import os
import unittest

from legacy_test.test_parallel_dygraph_dataparallel import (
    TestMultipleAccelerators,
)


class TestHybridParallel(TestMultipleAccelerators):
    # check sharding logic as well as the accuracy with single mode
    def test_hybrid_parallel_sharding_logic(self):
        # test shard v2
        os.environ["FLAGS_shard_split_param"] = "1"
        self.run_mnist_2accelerators('hybrid_parallel_sharding_model.py')

        # test shard grad reduce
        os.environ["FLAGS_shard_split_param"] = "0"
        self.run_mnist_2accelerators('hybrid_parallel_sharding_model.py')

    def test_hybrid_parallel_sharding_tensor_fusion(self):
        os.environ["FLAGS_shard_split_param"] = "0"
        self.run_mnist_2accelerators(
            'hybrid_parallel_sharding_model_with_fusion.py'
        )

    def test_hybrid_parallel_sharding_tensor_fusion_amp(self):
        os.environ["FLAGS_shard_split_param"] = "0"
        self.run_mnist_2accelerators(
            'hybrid_parallel_sharding_model_with_fusion_amp.py'
        )

    def test_hybrid_parallel_sharding_state_dict(self):
        os.environ["FLAGS_shard_split_param"] = "0"
        self.run_mnist_2accelerators('hybrid_parallel_sharding_state_dict.py')

    def test_group_param_tensor_fusion(self):
        self.run_mnist_2accelerators(
            'hybrid_parallel_tensor_fusion_with_group.py'
        )

    def test_group_shard_with_color(self):
        # test shard v2
        os.environ["FLAGS_shard_split_param"] = "1"
        os.environ["FLAGS_shard_param_with_color"] = "1"
        self.run_mnist_2accelerators('hybrid_parallel_sharding_model.py')
        # reset
        os.environ["FLAGS_shard_param_with_color"] = "0"

    def test_hybrid_parallel_sharding_with_fuse_optimizer_states(self):
        # test shard v2
        os.environ["FLAGS_shard_split_param"] = "1"
        self.run_mnist_2accelerators(
            'hybrid_parallel_sharding_model_with_fuse_optimizer_states_enabled.py'
        )
        # reset
        os.environ["FLAGS_shard_split_param"] = "0"


if __name__ == "__main__":
    unittest.main()
