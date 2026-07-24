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

import paddle
from paddle.base import core
from paddle.compat import seed as compat_seed


class TestCompatSeed(unittest.TestCase):
    def test_seed(self):
        paddle.seed(42)
        seed_cpu_random = core.default_cpu_generator().random()
        if paddle.is_compiled_with_cuda():
            seed_gpu_random = core.default_cuda_generator(0).random()
        if paddle.is_compiled_with_xpu():
            seed_xpu_random = core.default_xpu_generator(0).random()
        paddle.seed(42)
        compat_seed()
        compat_seed_cpu_random = core.default_cpu_generator().random()

        if paddle.is_compiled_with_cuda():
            compat_seed_gpu_random = core.default_cuda_generator(0).random()
            assert seed_gpu_random != compat_seed_gpu_random, (
                "GPU Random Seed Not Change!"
            )
        if paddle.is_compiled_with_xpu():
            compat_seed_xpu_random = core.default_xpu_generator(0).random()
            assert seed_xpu_random != compat_seed_xpu_random, (
                "XPU Random Seed Not Change!"
            )

        assert seed_cpu_random != compat_seed_cpu_random, (
            "CPU Random Seed Not Change!"
        )


if __name__ == '__main__':
    unittest.main()
