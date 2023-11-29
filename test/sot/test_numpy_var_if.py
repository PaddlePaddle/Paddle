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

import numpy as np
from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph, check_no_fallback
from paddle.jit.sot.utils import ENV_MIN_GRAPH_SIZE

ENV_MIN_GRAPH_SIZE.set(-1)


@check_no_breakgraph
@check_no_fallback
def forward(x, y):
    if x == 0:
        return y + 2
    else:
        return y * 2


@check_no_breakgraph
@check_no_fallback
def forward2(x, y):
    if x == x:  # numpy == numpy
        return y + 2
    else:
        return y * 2


class TestJumpWithNumpy(TestCaseBase):
    def test_jump(self):
        self.assert_results(forward, np.array([1]), paddle.to_tensor(2))
        self.assert_results(forward, np.array([0]), paddle.to_tensor(2))
        self.assert_results(forward2, np.array([0]), paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()
