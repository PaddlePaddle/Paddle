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

import unittest

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle


def if_breakgraph(x, y):
    z = x + y
    if x:
        z = z + 1
    else:
        z = z - 1
    return z


class TestResumeCache(TestCaseBase):
    def test_resume_cache_in_if_breakgraph(self):
        x = paddle.to_tensor(0)
        with test_instruction_translator_cache_context() as cache:
            self.assertEqual(cache.translate_count, 0)
            self.assert_results(if_breakgraph, x, 1)
            self.assertEqual(cache.translate_count, 2)
            self.assert_results(if_breakgraph, x, 2)
            self.assertEqual(cache.translate_count, 3)


if __name__ == "__main__":
    unittest.main()
