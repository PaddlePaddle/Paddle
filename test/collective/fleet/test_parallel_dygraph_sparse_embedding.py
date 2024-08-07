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

sys.path.append("../../legacy_test")

from parallel_dygraph_sparse_embedding import TestSparseEmbedding
from spawn_runner_base import TestDistSpawnRunner
from test_dist_base import TestDistBase

from paddle import base

flag_name = os.path.splitext(__file__)[0]


class TestParallelDygraphSparseEmbedding(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_sparse_embedding(self):
        if base.core.is_compiled_with_cuda():
            self.check_with_place(
                os.path.abspath(
                    "../../legacy_test/parallel_dygraph_sparse_embedding.py"
                ),
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name,
            )


class TestParallelDygraphSparseEmbeddingFP64(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_sparse_embedding_fp64(self):
        if base.core.is_compiled_with_cuda():
            self.check_with_place(
                os.path.abspath(
                    "../../legacy_test/parallel_dygraph_sparse_embedding_fp64.py"
                ),
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name,
            )


class TestParallelDygraphSparseEmbeddingSpawn(TestDistSpawnRunner):
    def test_sparse_embedding_with_spawn(self):
        if base.core.is_compiled_with_cuda():
            self.check_dist_result_with_spawn(
                test_class=TestSparseEmbedding, delta=1e-5
            )


if __name__ == "__main__":
    unittest.main()
