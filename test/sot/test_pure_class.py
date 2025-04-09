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

import paddle.distributed as dist
from paddle.jit.sot.psdb import check_no_breakgraph, check_no_fallback


@check_no_breakgraph
@check_no_fallback
def forward():
    mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
    placement = [dist.Shard(0), dist.Replicate(), dist.Partial()]


class TestPureClass(TestCaseBase):
    def test_class(self):
        self.assert_results(forward)


if __name__ == "__main__":
    unittest.main()
