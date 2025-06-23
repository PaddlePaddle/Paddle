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
from paddle.jit.sot.opcode_translator.executor.executor_cache import (
    OpcodeExecutorCache,
)
from paddle.jit.sot.utils import ENV_SOT_UNSAFE_CACHE_FASTPATH
from paddle.utils.environments import (
    EnvironmentVariableGuard,
)


def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


class TestUnsafeCacheFastPath(TestCaseBase):
    def test_guard(self):
        # NOTE: When UNSAFE CACHE FASTPATH is enabled, if the same cache entry is hit consecutively
        # for 32 times (this threshold is configurable), the cache is considered stable and
        # subsequent guard checks will be skipped to improve performance.
        # The related logic is implemented in the OpcodeExecutorCache class.
        with EnvironmentVariableGuard(ENV_SOT_UNSAFE_CACHE_FASTPATH, True):

            self.assertTrue(ENV_SOT_UNSAFE_CACHE_FASTPATH.get())

            self.assertFalse(
                OpcodeExecutorCache().is_fastpath_threshold_reached(
                    add.__code__
                )
            )
            for _ in range(33):
                self.assert_results(add, 1, paddle.ones([4]))
            self.assertTrue(
                OpcodeExecutorCache().is_fastpath_threshold_reached(
                    add.__code__
                )
            )
            self.assertFalse(
                OpcodeExecutorCache().is_fastpath_threshold_reached(
                    subtract.__code__
                )
            )

        with EnvironmentVariableGuard(ENV_SOT_UNSAFE_CACHE_FASTPATH, False):
            self.assertFalse(ENV_SOT_UNSAFE_CACHE_FASTPATH.get())
            self.assertFalse(
                OpcodeExecutorCache().is_fastpath_threshold_reached(
                    subtract.__code__
                )
            )
            for _ in range(35):
                self.assert_results(add, 1, 2)
            self.assertFalse(
                OpcodeExecutorCache().is_fastpath_threshold_reached(
                    subtract.__code__
                )
            )


if __name__ == '__main__':
    unittest.main()
