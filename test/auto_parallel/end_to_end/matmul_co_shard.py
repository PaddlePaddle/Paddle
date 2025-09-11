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

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed import Partial, Replicate, Shard


class MatmulTestCase:
    def __init__(
        self,
        x_shape: list[int],
        x_placements: list[dist.Placement],
        y_shape: list[int],
        y_placements: list[dist.Placement],
        trans_x: bool,
        trans_y: bool,
        output_shape: list[int],
        output_placements: list[dist.Placement],
    ):
        self.x_shape = x_shape
        self.x_placements = x_placements
        self.y_shape = y_shape
        self.y_placements = y_placements
        self.trans_x = trans_x
        self.trans_y = trans_y
        self.output_shape = output_shape
        self.output_placements = output_placements


class TestMatmulCoShard:
    def setUp(self):
        self.mesh = dist.ProcessMesh(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=['x', 'y', 'z']
        )
        self.test_cases_forward = [
            # test flatten
            MatmulTestCase(
                [64, 32],
                [Shard(0, shard_order=0), Shard(0, shard_order=1), Replicate()],
                [32, 48],
                [Replicate(), Replicate(), Shard(1)],
                False,
                False,
                [64, 48],
                [Shard(0, shard_order=0), Shard(0, shard_order=1), Shard(1)],
            ),
            MatmulTestCase(
                [64, 32],
                [Replicate(), Replicate(), Replicate()],
                [32, 48],
                [Shard(0, shard_order=0), Shard(0, shard_order=1), Shard(1)],
                False,
                False,
                [64, 48],
                [Partial(), Partial(), Shard(1)],
            ),
            MatmulTestCase(
                [64, 32],
                [Shard(0, shard_order=1), Shard(0, shard_order=1), Shard(1)],
                [32, 48],
                [Replicate(), Replicate(), Replicate()],
                False,
                False,
                [64, 48],
                [Shard(0, shard_order=0), Shard(0, shard_order=1), Partial()],
            ),
            MatmulTestCase(
                [64, 32],
                [Shard(0, shard_order=1), Shard(0, shard_order=1), Shard(1)],
                [32, 48],
                [Shard(0), Replicate(), Replicate()],
                False,
                False,
                [64, 48],
                [Shard(0, shard_order=0), Shard(0, shard_order=1), Partial()],
            ),
            MatmulTestCase(
                [512, 48, 64, 32],
                [Shard(0, shard_order=1), Shard(0, shard_order=1), Shard(1)],
                [1, 32, 48],
                [Replicate(), Replicate(), Replicate()],
                False,
                False,
                [512, 48, 64, 48],
                [Shard(0, shard_order=0), Shard(0, shard_order=1), Shard(1)],
            ),
            MatmulTestCase(
                [512, 48, 32, 64],
                [Shard(0), Shard(2, shard_order=0), Shard(2, shard_order=1)],
                [1, 32, 48],
                [Replicate(), Replicate(), Shard(2)],
                True,
                False,
                [512, 48, 64, 48],
                [Shard(0), Partial(), Shard(3)],
            ),
            MatmulTestCase(
                [512, 48, 64, 32],
                [Shard(0), Shard(2, shard_order=0), Shard(2, shard_order=1)],
                [1, 48, 32],
                [Shard(1), Replicate(), Replicate()],
                False,
                True,
                [512, 48, 64, 48],
                [Shard(0), Shard(2, shard_order=0), Shard(2, shard_order=1)],
            ),
            MatmulTestCase(
                [512, 48, 32, 64],
                [Shard(2, shard_order=0), Shard(2, shard_order=1), Shard(3)],
                [1, 48, 32],
                [Shard(1, shard_order=0), Shard(1, shard_order=1), Shard(2)],
                True,
                True,
                [512, 48, 64, 48],
                [Shard(3, shard_order=0), Shard(3, shard_order=1), Shard(2)],
            ),
        ]

    def run_test_case_forward(self, test_case: MatmulTestCase):
        x = paddle.rand(test_case.x_shape, "float32")
        x_placements = test_case.x_placements
        x = dist.shard_tensor(x, self.mesh, x_placements)

        y = paddle.rand(test_case.y_shape, "float32")
        y_placements = test_case.y_placements
        y = dist.shard_tensor(y, self.mesh, y_placements)

        out = paddle.matmul(x, y, test_case.trans_x, test_case.trans_y)
        case_info = f"x_shape: {test_case.x_shape}, x_placements: {x_placements}, y_shape: {test_case.y_shape}, y_placements: {test_case.y_placements}, trans_x: {test_case.trans_x}, trans_y: {test_case.trans_y}"
        # Verify output shape
        np.testing.assert_equal(
            out.shape,
            test_case.output_shape,
            err_msg=f"Output shape mismatch when {case_info}. Expected: {test_case.output_shape}, Actual: {out.shape}",
        )

        # Verify placements
        assert out.placements
        for actual, expected in zip(
            out.placements, test_case.output_placements
        ):
            np.testing.assert_equal(
                actual,
                expected,
                err_msg=f"Output placements mismatch when {case_info}. Expected: {test_case.output_placements}, Actual: {out.placements}",
            )

    def run_all_tests(self):
        self.setUp()
        for test_case in self.test_cases_forward:
            self.run_test_case_forward(test_case)


if __name__ == '__main__':
    TestMatmulCoShard().run_all_tests()
