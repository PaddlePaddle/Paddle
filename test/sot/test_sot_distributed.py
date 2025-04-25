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

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
import paddle.distributed as dist
from paddle.jit.sot.psdb import check_no_breakgraph


@check_no_breakgraph
def fn(x, y):
    return x + y


@unittest.skipIf(
    not paddle.is_compiled_with_distribute(),
    reason='Not compiled with distribute.',
)
class TestGuardForDistInfo(TestCaseBase):
    def test_fn(self):
        x = paddle.ones([2, 2])
        x.stop_gradient = False
        y = paddle.zeros([2, 2])
        y.stop_gradient = False
        mesh1 = dist.ProcessMesh([0, 1], dim_names=['x'])
        mesh2 = dist.ProcessMesh([0, 1], dim_names=['y'])
        mesh3 = dist.ProcessMesh([0, 2], dim_names=['x'])
        dist_x1 = dist.shard_tensor(
            x, mesh1, [dist.Replicate()], stop_gradient=False
        )
        dist_y1 = dist.shard_tensor(
            y, mesh1, [dist.Replicate()], stop_gradient=False
        )
        dist_x2 = dist.shard_tensor(x, mesh2, [dist.Replicate()])
        dist_y2 = dist.shard_tensor(y, mesh2, [dist.Replicate()])
        dist_x3 = dist.shard_tensor(x, mesh3, [dist.Replicate()])
        dist_y3 = dist.shard_tensor(y, mesh3, [dist.Replicate()])
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(fn, dist_x1, dist_y1)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(fn, dist_x2, dist_y2)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(fn, dist_x3, dist_y3)
            self.assertEqual(ctx.translate_count, 2)


if __name__ == "__main__":
    unittest.main()
