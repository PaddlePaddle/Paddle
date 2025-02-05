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
from paddle.jit.sot.opcode_translator.executor.dispatcher import Dispatcher
from paddle.jit.sot.utils.envs import min_graph_size_guard

# 8 will trigger the warmup in RESUME instruction and cause a segmentation fault
# RUN_N_TIMES should be larger than 8
RUN_N_TIMES = 20

builtin_fn = str.split
# Remove builtin_fn from Dispatcher to ensure that trigger a BreakGraph Error
if builtin_fn in Dispatcher.handlers:
    del Dispatcher.handlers[builtin_fn]


def builtin_fn_with_breakgraph():
    str.split("1,2,3,4,5", ",")


class TestSpecialization(TestCaseBase):
    @min_graph_size_guard(10)
    def test_specialization(self):
        for _ in range(RUN_N_TIMES):
            paddle.jit.to_static(builtin_fn_with_breakgraph)()


if __name__ == "__main__":
    unittest.main()
