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


class TestHybridPipeParallel(TestMultipleAccelerators):
    def test_hybrid_parallel_pp_layer(self):
        self.run_mnist_2accelerators(
            os.path.abspath('../../legacy_test/hybrid_parallel_pp_layer.py'),
            need_envs={
                "PADDLE_P2P_SYNC_SEND": "1",
            },
        )

    def test_hybrid_parallel_pp_tuple_inputs(self):
        self.run_mnist_2accelerators(
            'hybrid_parallel_pp_embedding.py',
            need_envs={
                "PADDLE_P2P_SYNC_SEND": "1",
            },
        )

    def test_hybrid_parallel_shared_weight(self):
        self.run_mnist_2accelerators(
            'hybrid_parallel_shared_weight.py',
            need_envs={
                "PADDLE_P2P_SYNC_SEND": "1",
            },
        )

    def test_pipeline_parallel_amp(self):
        self.run_mnist_2accelerators(
            'hybrid_parallel_pp_amp.py',
            need_envs={
                "PADDLE_P2P_SYNC_SEND": "1",
            },
        )

    def test_hybrid_parallel_pp_return_micro_batch_loss(self):
        self.run_mnist_2accelerators(
            'hybrid_parallel_pp_return_micro_batch_loss.py',
            need_envs={
                "PADDLE_P2P_SYNC_SEND": "1",
            },
        )


if __name__ == "__main__":
    unittest.main()
