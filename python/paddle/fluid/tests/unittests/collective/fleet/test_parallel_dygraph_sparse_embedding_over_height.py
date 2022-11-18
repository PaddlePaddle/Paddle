# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import unittest

import paddle.fluid as fluid
from test_dist_base import TestDistBase
from spawn_runner_base import TestDistSpawnRunner
from parallel_dygraph_sparse_embedding_over_height import (
    TestSparseEmbeddingOverHeight,
)

flag_name = os.path.splitext(__file__)[0]


class TestParallelDygraphSparseEmdeddingOverHeight(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_sparse_embedding(self):
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place(
                os.path.abspath(
                    "../../parallel_dygraph_sparse_embedding_over_height.py"
                ),
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name,
            )


class TestParallelDygraphSparseEmdeddingOverHeightSpawn(TestDistSpawnRunner):
    def test_sparse_embedding_with_spawn(self):
        if fluid.core.is_compiled_with_cuda() and sys.version_info >= (3, 4):
            self.check_dist_result_with_spawn(
                test_class=TestSparseEmbeddingOverHeight, delta=1e-5
            )


if __name__ == "__main__":
    unittest.main()
