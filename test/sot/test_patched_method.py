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

import types
import unittest

from test_case_base import (
    TestCaseBase,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


class ScaleOp:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def apply(self, x):
        return x * self.weight + self.bias


class Module(types.ModuleType):
    def __init__(self):
        super().__init__("module")
        _scale = ScaleOp(2, 3)
        self.scale = _scale.apply


mod = Module()


@check_no_breakgraph
def call_patched_fn(x):
    return mod.scale(x)


class TestCallPatchedFn(TestCaseBase):
    def test_call_patched_fn(self):
        x = paddle.randn([16, 72])
        self.assert_results(call_patched_fn, x)


if __name__ == "__main__":
    unittest.main()
