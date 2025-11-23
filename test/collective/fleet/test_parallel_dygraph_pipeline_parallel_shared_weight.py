# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestHybridPipeParallelSharedWeight(TestMultipleAccelerators):
    def test_hybrid_parallel_shared_weight_with_multi_attrs_and_subset_all_on_one_stage(
        self,
    ):
        self.run_mnist_2accelerators(
            'hybrid_parallel_shared_weight_with_multi_attrs_and_subset_all_on_one_stage.py'
        )

    def test_hybrid_parallel_shared_weight_with_multi_attrs_and_subset_pivot_not_at_first(
        self,
    ):
        self.run_mnist_2accelerators(
            'hybrid_parallel_shared_weight_with_multi_attrs_and_subset_pivot_not_at_first.py'
        )

    def test_hybrid_parallel_shared_weight_with_multi_attrs_and_subset(self):
        self.run_mnist_2accelerators(
            'hybrid_parallel_shared_weight_with_multi_attrs_and_subset.py'
        )

    def test_hybrid_parallel_shared_weight_with_multi_attrs(self):
        self.run_mnist_2accelerators(
            'hybrid_parallel_shared_weight_with_multi_attrs.py'
        )

    def test_hybrid_parallel_shared_weight(self):
        self.run_mnist_2accelerators('hybrid_parallel_shared_weight.py')


if __name__ == "__main__":
    unittest.main()
