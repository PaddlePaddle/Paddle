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


class BinaryElementwiseTestCase:
    """
    A data class to hold parameters for a binary elementwise operation test case.
    """

    def __init__(
        self,
        shape: list[int],
        placements_x: Sequence[dist.Placement],
        placements_y: Sequence[dist.Placement],
        expected_placements_x: Sequence[dist.Placement],
        expected_placements_y: Sequence[dist.Placement],
        expected_placements_out: Sequence[dist.Placement],
    ):
        self.shape = shape
        self.placements_x = placements_x
        self.placements_y = placements_y
        self.expected_placements_x = expected_placements_x
        self.expected_placements_y = expected_placements_y
        self.expected_placements_out = expected_placements_out


class TestAddCoShard:
    """
    Unit tests for co_shard SPMD rule on binary elementwise operations.
    """

    def setUp(self):
        """
        Initializes the process mesh and defines the test cases.
        """
        self.mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
        # The expected placements for inputs and output are the same after co-sharding.
        expected_placements = [
            dist.Shard(dim=0, shard_order=0),
            dist.Shard(dim=0, shard_order=1),
        ]
        self.test_cases = [
            # Test Case 1: [[0],[1]],[[0,1],[]] -> [[0,1],[]], [[0,1],[]], [[0,1],[]]
            BinaryElementwiseTestCase(
                shape=[64, 64],
                placements_x=[dist.Shard(0), dist.Shard(1)],
                placements_y=[
                    dist.Shard(dim=0, shard_order=0),
                    dist.Shard(dim=0, shard_order=1),
                ],
                expected_placements_x=expected_placements,
                expected_placements_y=expected_placements,
                expected_placements_out=expected_placements,
            ),
            # Test Case 2: [[0],[]], [[1],[]] -> [[0,1],[]], [[0,1],[]], [[0,1],[]]
            BinaryElementwiseTestCase(
                shape=[64, 64],
                placements_x=[dist.Shard(0), dist.Replicate()],
                placements_y=[dist.Shard(1), dist.Replicate()],
                expected_placements_x=expected_placements,
                expected_placements_y=expected_placements,
                expected_placements_out=expected_placements,
            ),
        ]

    def run_test_case(self, test_case: BinaryElementwiseTestCase):
        # Prepare inputs
        a = paddle.randn(test_case.shape, dtype='float32')
        b = paddle.randn(test_case.shape, dtype='float32')

        # Shard tensors
        x = dist.shard_tensor(a, self.mesh, test_case.placements_x)
        y = dist.shard_tensor(b, self.mesh, test_case.placements_y)

        # Perform operation
        out = paddle.add(x, y)

        case_info = f"placements_x: {test_case.placements_x}, placements_y: {test_case.placements_y}"

        # Verify placements of inputs (post-operation due to potential resharding)
        assert x.placements and y.placements
        for actual, expected in zip(
            x.placements, test_case.expected_placements_x
        ):
            np.testing.assert_equal(
                actual,
                expected,
                err_msg=f"Input 'x' placements mismatch when {case_info}. Expected: {test_case.expected_placements_x}, Actual: {x.placements}",
            )
        for actual, expected in zip(
            y.placements, test_case.expected_placements_y
        ):
            np.testing.assert_equal(
                actual,
                expected,
                err_msg=f"Input 'y' placements mismatch when {case_info}. Expected: {test_case.expected_placements_y}, Actual: {y.placements}",
            )

        # Verify output shape
        np.testing.assert_equal(
            out.shape,
            test_case.shape,
            err_msg=f"Output shape mismatch when {case_info}. Expected: {test_case.shape}, Actual: {out.shape}",
        )

        # Verify output placements
        assert out.placements
        for actual, expected in zip(
            out.placements, test_case.expected_placements_out
        ):
            np.testing.assert_equal(
                actual,
                expected,
                err_msg=f"Output placements mismatch when {case_info}. Expected: {test_case.expected_placements_out}, Actual: {out.placements}",
            )

    def run_all_tests(self):
        self.setUp()
        for test_case in self.test_cases:
            self.run_test_case(test_case)


if __name__ == '__main__':
    TestAddCoShard().run_all_tests()
