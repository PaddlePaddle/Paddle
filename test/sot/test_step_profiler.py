# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import unittest

import paddle
from paddle.jit import sot
from paddle.jit.sot.utils import strict_mode_guard
from paddle.pir_utils import DygraphPirGuard


class SimpleModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(3, 2, 3)
        self.conv2 = paddle.nn.Conv2D(2, 3, 3)

    def inner_fn(self, x):
        sot.psdb.fallback()
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        self.inner_fn(x)

        return x


class TestStepProfilerSmokeTest(unittest.TestCase):
    # Temporarily disable this test
    # @sot_step_profiler_guard(True)
    @strict_mode_guard(False)
    def test_step_profiler_smoke(self):
        with DygraphPirGuard():
            model = SimpleModel()
            model = paddle.jit.to_static(model, full_graph=False)
            x = paddle.randn([1, 3, 32, 32])

            model(x)


if __name__ == "__main__":
    unittest.main()
