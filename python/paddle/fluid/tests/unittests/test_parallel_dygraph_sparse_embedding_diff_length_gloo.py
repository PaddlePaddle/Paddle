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

import os
import sys
import unittest

import paddle.fluid as fluid
from test_dist_base import TestDistBase
from spawn_runner_base import TestDistSpawnRunner
from parallel_dygraph_sparse_embedding import TestSparseEmbedding
from parallel_dygraph_sparse_embedding_fp64 import TestSparseEmbeddingFP64

flag_name = os.path.splitext(__file__)[0]


class TestParallelDygraphSparseEmdedding_GLOO(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._gloo_mode = True
        self._dygraph = True
        self._diff_batch = True

    def test_sparse_embedding(self):
        self.check_with_place(
            "parallel_dygraph_sparse_embedding.py",
            delta=1e-5,
            check_error_log=True,
            log_name=flag_name)


if __name__ == "__main__":
    unittest.main()
