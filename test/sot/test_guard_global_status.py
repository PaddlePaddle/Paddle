# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
    test_instruction_translator_cache_context,
)

import paddle


class TestGuardIsGradEnabled(TestCaseBase):
    def test_switch_is_grad_enabled(self):
        def fn(x):
            return x + 1

        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            with paddle.no_grad():
                self.assert_results(fn, 1)
            self.assertEqual(ctx.translate_count, 1)
            with paddle.enable_grad():
                self.assert_results(fn, 1)
            self.assertEqual(ctx.translate_count, 2)


if __name__ == "__main__":
    unittest.main()
