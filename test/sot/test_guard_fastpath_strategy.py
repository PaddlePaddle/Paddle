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

from __future__ import annotations

import unittest

from test_case_base import (
    TestCaseBase,
)

import paddle
from paddle.jit.sot.utils import ENV_SOT_UNSAFE_CACHE_FASTPATH
from paddle.utils.environments import (
    EnvironmentVariableGuard,
)


def add(x, y):
    return x + y


class TestGuardOutputs(TestCaseBase):
    def test_guard_inputs(self):
        # NOTE: When UNSAFE CACHE FASTPATH is enabled, if the same cache entry is hit consecutively
        # for 32 times (this threshold is configurable), the cache is considered stable and
        # subsequent guard checks will be skipped to improve performance.
        # The related logic is implemented in the OpcodeExecutorCache class.
        with EnvironmentVariableGuard(ENV_SOT_UNSAFE_CACHE_FASTPATH, True):
            self.assertTrue(ENV_SOT_UNSAFE_CACHE_FASTPATH.get())
            for _ in range(50):
                self.assert_results(add, 1, paddle.ones([4]))

        with EnvironmentVariableGuard(ENV_SOT_UNSAFE_CACHE_FASTPATH, False):
            self.assertFalse(ENV_SOT_UNSAFE_CACHE_FASTPATH.get())
            for _ in range(1000):
                self.assert_results(add, 1, 2)


if __name__ == '__main__':
    unittest.main()
