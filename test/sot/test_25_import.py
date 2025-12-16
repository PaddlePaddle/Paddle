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

from test_case_base import TestCaseBase

from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils.exceptions import FallbackError


@check_no_breakgraph
def import_math_model():
    import math

    return math.sqrt(4)


def import_relative():
    from . import test_case_base

    return test_case_base


@check_no_breakgraph
def import_paddle_model(x: int):
    import paddle

    return paddle.zeros([2, 3]) + x


class TestImportModel(TestCaseBase):
    def test_import_model(self):
        self.assert_results(import_math_model)
        self.assert_results(import_paddle_model, 1)

    def test_relative_import_error(self):
        self.assert_exceptions(
            FallbackError,
            "relative import with no known parent package",
            import_relative,
        )


if __name__ == "__main__":
    unittest.main()
