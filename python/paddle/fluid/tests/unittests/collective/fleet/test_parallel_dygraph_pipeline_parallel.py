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
import os

from test_parallel_dygraph_dataparallel import TestMultipleGpus


class TestHybridPipeParallel(TestMultipleGpus):
    def test_hybrid_parallel_pp_layer(self):
        self.run_mnist_2gpu(
            os.path.abspath('../../hybrid_parallel_pp_layer.py')
        )
        self.run_mnist_2gpu(
            os.path.abspath('../../hybrid_parallel_pp_layer.py'),
            eager_mode=False,
        )

    def test_hybrid_parallel_pp_tuple_inputs(self):
        self.run_mnist_2gpu('hybrid_parallel_pp_embedding.py')
        self.run_mnist_2gpu('hybrid_parallel_pp_embedding.py', eager_mode=False)

    def test_hybrid_parallel_shared_weight(self):
        self.run_mnist_2gpu('hybrid_parallel_shared_weight.py')
        self.run_mnist_2gpu(
            'hybrid_parallel_shared_weight.py', eager_mode=False
        )

    def test_pipeline_parallel_amp(self):
        self.run_mnist_2gpu('hybrid_parallel_pp_amp.py')
        self.run_mnist_2gpu('hybrid_parallel_pp_amp.py', eager_mode=False)

    def test_pipeline_parallel_fp16(self):
        self.run_mnist_2gpu('hybrid_parallel_pp_fp16.py')
        self.run_mnist_2gpu('hybrid_parallel_pp_fp16.py', eager_mode=False)

    def test_hybrid_parallel_transformer(self):
        self.run_mnist_2gpu('hybrid_parallel_pp_transformer.py')
        self.run_mnist_2gpu(
            'hybrid_parallel_pp_transformer.py', eager_mode=False
        )

    def test_hybrid_parallel_save_load(self):
        self.run_mnist_2gpu('hybrid_parallel_pp_save_load.py')
        self.run_mnist_2gpu('hybrid_parallel_pp_save_load.py', eager_mode=False)

    def test_hybrid_parallel_recompute(self):
        self.run_mnist_2gpu('hybrid_parallel_pp_recompute.py')
        self.run_mnist_2gpu('hybrid_parallel_pp_recompute.py', eager_mode=False)

    def test_hybrid_parallel_pp_clip_grad(self):
        self.run_mnist_2gpu('hybrid_parallel_pp_clip_grad.py')
        self.run_mnist_2gpu('hybrid_parallel_pp_clip_grad.py', eager_mode=False)

    def test_hybrid_parallel_transformer_unbalanced_data(self):
        self.run_mnist_2gpu('hybrid_parallel_pp_transformer_unbalanced_data.py')
        self.run_mnist_2gpu(
            'hybrid_parallel_pp_transformer_unbalanced_data.py',
            eager_mode=False,
        )


if __name__ == "__main__":
    os.environ["FLAGS_enable_eager_mode"] = "1"
    unittest.main()
