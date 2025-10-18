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
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import paddle
import paddle.distributed as dist

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle._typing import TensorOrTensors


class TileTestCase:
    def __init__(
        self,
        x_shape: list[int],
        x_placements: list[dist.Placement],
        repeat_times: TensorOrTensors | Sequence[int],
        out_shape: list[int],
        out_placements: list[dist.Placement],
    ):
        self.x_shape = x_shape
        self.x_placements = x_placements
        self.repeat_times = repeat_times
        self.out_shape = out_shape
        self.out_placements = out_placements


class TestTileCoShard:
    def setUp(self):
        self.mesh = dist.ProcessMesh(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=['x', 'y', 'z']
        )
        self.test_cases_forward = [
            TileTestCase(
                [8, 16, 24],
                [
                    dist.Shard(0),
                    dist.Shard(2, shard_order=0),
                    dist.Shard(2, shard_order=1),
                ],
                [2, 2, 1, 1],
                [2, 16, 16, 24],
                [
                    dist.Replicate(),
                    dist.Shard(3, shard_order=0),
                    dist.Shard(3, shard_order=1),
                ],
            ),
            TileTestCase(
                [8, 16, 24],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(2),
                ],
                [1, 2],
                [8, 16, 48],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
            ),
            TileTestCase(
                [8, 16, 24],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(2),
                ],
                [],
                [8, 16, 24],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(2),
                ],
            ),
        ]

    def run_test_case_forward(self, test_case: TileTestCase):
        paddle.seed(2025)
        x = paddle.rand(test_case.x_shape, "float32")
        x_placements = test_case.x_placements
        input = dist.shard_tensor(x, self.mesh, x_placements)
        out = paddle.tile(input, test_case.repeat_times)
        case_info = f"input_shape: {test_case.x_shape}, input_placements: {x_placements}, axis: {test_case.repeat_times}"
        # Verify output shape
        np.testing.assert_equal(
            out.shape,
            test_case.out_shape,
            err_msg=f"Output shape mismatch when {case_info}. Expected: {test_case.out_shape}, Actual: {out.shape}",
        )

        # Verify placements
        assert out.placements
        for actual, expected in zip(out.placements, test_case.out_placements):
            np.testing.assert_equal(
                actual,
                expected,
                err_msg=f"Output placements mismatch when {case_info}. Expected: {test_case.out_placements}, Actual: {out.placements}",
            )

    def run_all_tests(self):
        self.setUp()
        for test_case in self.test_cases_forward:
            self.run_test_case_forward(test_case)


if __name__ == '__main__':
    TestTileCoShard().run_all_tests()
