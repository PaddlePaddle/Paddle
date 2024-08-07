# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class TestHybridPipeParallelWithVirtualStage(TestMultipleAccelerators):
    def test_hybrid_parallel_pp_layer_with_virtual_stage(self):
        # self.run_mnist_2accelerators('hybrid_parallel_pp_layer_with_virtual_stage.py')
        pass

    def test_hybrid_parallel_pp_transformer_with_virtual_stage(self):
        # self.run_mnist_2accelerators(
        #    'hybrid_parallel_pp_transformer_with_virtual_stage.py'
        # )
        pass

    def test_hybrid_parallel_save_load_with_virtual_stage(self):
        # self.run_mnist_2accelerators(
        #    'hybrid_parallel_pp_save_load_with_virtual_stage.py'
        # )
        pass


if __name__ == "__main__":
    unittest.main()
