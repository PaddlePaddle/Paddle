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

import paddle
from paddle.jit.sot.utils.envs import min_graph_size_guard, strict_mode_guard

# 8 will trigger the warmup in RESUME instruction and cause a segmentation fault
# RUN_N_TIMES should be larger than 8
RUN_N_TIMES = 20


def listcomp_fn():
    print(1)
    x = [i for i in range(10)]  # noqa: C416
    return x


def genexpr_fn():
    print(1)
    x = (i for i in range(10))
    return x


class TestListComp(TestCaseBase):
    @strict_mode_guard(False)
    @min_graph_size_guard(10)
    def test_listcomp(self):
        for _ in range(RUN_N_TIMES):
            paddle.jit.to_static(listcomp_fn)()


class TestGenExpr(TestCaseBase):
    @strict_mode_guard(False)
    @min_graph_size_guard(10)
    def test_genexpr(self):
        for _ in range(RUN_N_TIMES):
            paddle.jit.to_static(genexpr_fn)()


if __name__ == "__main__":
    unittest.main()
