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


class TestHybridParallelShardingV2ChunkOffload(TestMultipleAccelerators):
    # check sharding logic as well as the accuracy with single mode
    def test_hybrid_parallel_sharding_v2_chunk_offload(self):
        # test sharding v2 chunk offload
        os.environ["FLAGS_shard_split_param"] = "1"
        self.run_mnist_2accelerators(
            'hybrid_parallel_sharding_model_chunkoffload.py'
        )


if __name__ == "__main__":
    unittest.main()
