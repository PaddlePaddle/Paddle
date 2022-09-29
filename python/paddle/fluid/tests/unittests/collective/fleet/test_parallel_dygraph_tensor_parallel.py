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

from test_parallel_dygraph_dataparallel import TestMultipleGpus


class TestHybridParallel(TestMultipleGpus):

    def test_hybrid_parallel_mp_random(self):
        self.run_mnist_2gpu('hybrid_parallel_mp_random.py')
        self.run_mnist_2gpu('hybrid_parallel_mp_random.py', eager_mode=False)

    def test_hybrid_parallel_mp_model(self):
        self.run_mnist_2gpu('hybrid_parallel_mp_model.py')
        self.run_mnist_2gpu('hybrid_parallel_mp_model.py', eager_mode=False)

    def test_hybrid_parallel_mp_amp(self):
        self.run_mnist_2gpu('hybrid_parallel_mp_amp.py')
        self.run_mnist_2gpu('hybrid_parallel_mp_amp.py', eager_mode=False)

    def test_hybrid_parallel_mp_fp16(self):
        self.run_mnist_2gpu('hybrid_parallel_mp_fp16.py')
        self.run_mnist_2gpu('hybrid_parallel_mp_fp16.py', eager_mode=False)

    def test_hybrid_parallel_mp_clip_grad(self):
        self.run_mnist_2gpu('hybrid_parallel_mp_clip_grad.py')
        self.run_mnist_2gpu('hybrid_parallel_mp_clip_grad.py', eager_mode=False)


if __name__ == "__main__":
    os.environ["FLAGS_enable_eager_mode"] = "1"
    unittest.main()
