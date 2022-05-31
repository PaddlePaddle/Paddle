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

from __future__ import print_function

import unittest
import paddle.fluid as fluid

from test_parallel_dygraph_dataparallel import TestMultipleGpus


class TestDygraphShardingStage2(TestMultipleGpus):

    # check sharding logic as well as the accuracy with single mode
    def test_dygraph_sharding_stage2(self):
        self.run_mnist_2gpu('dygraph_group_sharded_stage2.py')
        self.run_mnist_2gpu('dygraph_sharding_stage2.py')

    def test_dygraph_sharding_stage2_offload(self):
        self.run_mnist_2gpu('dygraph_group_sharded_stage2_offload.py')
        self.run_mnist_2gpu('dygraph_sharding_stage2_offload.py')


if __name__ == "__main__":
    unittest.main()
