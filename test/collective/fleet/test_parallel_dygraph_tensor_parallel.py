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

from legacy_test.test_parallel_dygraph_dataparallel import (
    TestMultipleAccelerators,
)

from paddle.framework import core


class TestHybridParallel(TestMultipleAccelerators):
    def test_hybrid_parallel_mp_random(self):
        self.run_mnist_2accelerators('hybrid_parallel_mp_random.py')

    def test_hybrid_parallel_mp_model_with_sequence_parallel(self):
        self.run_mnist_2accelerators(
            'hybrid_parallel_mp_model_with_sequence_parallel.py'
        )

    def test_hybrid_parallel_mp_amp(self):
        self.run_mnist_2accelerators('hybrid_parallel_mp_amp.py')

    def test_hybrid_parallel_mp_fp16(self):
        self.run_mnist_2accelerators('hybrid_parallel_mp_fp16.py')

    def test_hybrid_parallel_mp_bf16(self):
        # XPU will use its own fast_paddle lib for bf16 training, therefore skip ordinary ut here.
        if not core.is_compiled_with_xpu():
            self.run_mnist_2accelerators('hybrid_parallel_mp_bf16.py')

    def test_hybrid_parallel_mp_clip_grad(self):
        self.run_mnist_2accelerators('hybrid_parallel_mp_clip_grad.py')

    def test_hybrid_parallel_mp_broadcast_obj(self):
        self.run_mnist_2accelerators('hybrid_parallel_mp_broadcast_obj.py')


if __name__ == "__main__":
    unittest.main()
