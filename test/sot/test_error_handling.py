# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from test_case_base import TestCaseBase

from paddle.jit import sot
from paddle.jit.sot.utils import strict_mode_guard


def fn_with_try_except():
    sot.psdb.breakgraph()
    sot.psdb.fallback()
    try:
        raise ValueError("ValueError")
    except ValueError:
        print("catch ValueError")
        return True


class TestErrorHandling(TestCaseBase):
    @strict_mode_guard(False)
    def test_fn_with_try_except(self):
        self.assert_results(fn_with_try_except)


if __name__ == "__main__":
    unittest.main()
