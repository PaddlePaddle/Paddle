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


@check_no_breakgraph
def import_os_model():
    import os

    return os.name


@check_no_breakgraph
def import_sys_version():
    from sys import version

    return version[:5]  # Return first 5 chars to check import


@check_no_breakgraph
def import_paddle_nn():
    import paddle.nn

    return paddle.nn.Layer.__name__


@check_no_breakgraph
def import_paddle_nn_from():
    from paddle import nn

    return nn.Layer.__name__


@check_no_breakgraph
def import_paddle_nn_functional():
    import paddle.nn.functional as F

    return F.relu.__name__


@check_no_breakgraph
def import_paddle_nn_linear():
    from paddle.nn import Linear

    return Linear.__name__


@check_no_breakgraph
def import_collections():
    import collections

    return collections.defaultdict.__name__


@check_no_breakgraph
def import_collections_defaultdict():
    from collections import defaultdict

    return defaultdict.__name__


@check_no_breakgraph
def import_multiple_modules():
    import os
    import sys

    return os.name + sys.version[:3]


@check_no_breakgraph
def import_paddle_nn_linear_as():
    from paddle.nn import Linear as Lin

    return Lin.__name__


class TestImportModel(TestCaseBase):
    def test_import_model(self):
        self.assert_results(import_math_model)
        self.assert_results(import_paddle_model, 1)
        self.assert_results(import_os_model)
        self.assert_results(import_sys_version)
        self.assert_results(import_paddle_nn)
        self.assert_results(import_paddle_nn_from)
        self.assert_results(import_paddle_nn_functional)
        self.assert_results(import_paddle_nn_linear)
        self.assert_results(import_collections)
        self.assert_results(import_collections_defaultdict)
        self.assert_results(import_multiple_modules)
        self.assert_results(import_paddle_nn_linear_as)

    def test_relative_import_error(self):
        self.assert_exceptions(
            FallbackError,
            "relative import with no known parent package",
            import_relative,
        )


if __name__ == "__main__":
    unittest.main()
