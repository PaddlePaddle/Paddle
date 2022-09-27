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

import unittest
import paddle.fluid as fluid

from test_parallel_dygraph_dataparallel import TestMultipleGpus


class TestPipelineParallel(TestMultipleGpus):

    def test_pipeline_parallel(self):
        self.run_mnist_2gpu('hybrid_parallel_pp_alexnet.py')


class TestModelParallelWithRecompute(TestMultipleGpus):

    def test_model_parallel_with_recompute(self):
        self.run_mnist_2gpu("dygraph_recompute_hybrid.py")


if __name__ == "__main__":
    unittest.main()
