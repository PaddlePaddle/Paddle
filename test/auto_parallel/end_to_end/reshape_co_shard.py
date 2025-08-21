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

from typing import TYPE_CHECKING, Any

import numpy as np

import paddle
import paddle.distributed as dist

if TYPE_CHECKING:
    from collections.abc import Callable


class ReshapeTestCase:
    def __init__(
        self,
        input_shape: list[int],
        input_placements: list[dist.Placement],
        target_shape: list[int],
        output_placements: list[dist.Placement],
        slice_funtor: Callable[[int], Any] | None = None,
    ):
        self.input_shape = input_shape
        self.input_placements = input_placements
        self.target_shape = target_shape
        self.output_placements = output_placements
        self.slice_funtor = slice_funtor


class TestReshapeCoShard:
    def setUp(self):
        self.mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
        self.test_cases = [
            # test flatten
            ReshapeTestCase(
                [4, 6, 8],
                [dist.Shard(0), dist.Shard(1)],
                [192],
                [dist.Shard(0, shard_order=0), dist.Shard(0, shard_order=1)],
                lambda idx: (idx,),
            ),
            ReshapeTestCase(
                [4, 6, 8],
                [dist.Shard(1), dist.Shard(2)],
                [192],
                [dist.Replicate(), dist.Replicate()],
                lambda idx: slice(None),
            ),
            ReshapeTestCase(
                [4, 6, 8],
                [dist.Shard(0, shard_order=0), dist.Shard(0, shard_order=1)],
                [192],
                [dist.Shard(0, shard_order=0), dist.Shard(0, shard_order=1)],
                lambda idx: (idx,),
            ),
            ReshapeTestCase(
                [2, 12, 8],
                [dist.Shard(0), dist.Shard(1)],
                [192],
                [dist.Shard(0), dist.Replicate()],
                lambda idx: (idx // 2,),
            ),
            # test split
            ReshapeTestCase(
                [192],
                [dist.Shard(0, shard_order=0), dist.Shard(0, shard_order=1)],
                [4, 6, 8],
                [dist.Shard(0, shard_order=0), dist.Shard(0, shard_order=1)],
                lambda idx: (idx,),
            ),
            ReshapeTestCase(
                [192],
                [dist.Shard(0, shard_order=0), dist.Shard(0, shard_order=1)],
                [6, 4, 8],
                [dist.Replicate(), dist.Replicate()],
                lambda idx: slice(None),
            ),
            # test combination
            ReshapeTestCase(
                [4, 6, 8],
                [dist.Shard(0), dist.Shard(1)],
                [2, 12, 8],
                [dist.Shard(0), dist.Replicate()],
                lambda idx: (idx // 2,),
            ),
            ReshapeTestCase(
                [4, 6, 8],
                [dist.Shard(0, shard_order=0), dist.Shard(0, shard_order=1)],
                [2, 12, 8],
                [dist.Replicate(), dist.Replicate()],
                lambda idx: slice(None),
            ),
            ReshapeTestCase(
                [4, 6, 8],
                [dist.Shard(0), dist.Shard(1)],
                [12, 2, 8],
                [dist.Shard(0, shard_order=0), dist.Shard(0, shard_order=1)],
                lambda idx: slice(idx % 4 * 3, idx % 4 * 3 + 3),
            ),
            ReshapeTestCase(
                [4, 6, 8],
                [dist.Shard(0, shard_order=0), dist.Shard(0, shard_order=1)],
                [12, 2, 8],
                [dist.Shard(0, shard_order=0), dist.Shard(0, shard_order=1)],
                lambda idx: slice(idx % 4 * 3, idx % 4 * 3 + 3),
            ),
            ReshapeTestCase(
                [4, 6, 8],
                [dist.Shard(0), dist.Shard(1)],
                [8, 6, 4],
                [dist.Shard(0, shard_order=0), dist.Shard(0, shard_order=1)],
                lambda idx: slice(idx % 4 * 2, idx % 4 * 2 + 2),
            ),
            ReshapeTestCase(
                [4, 6, 8],
                [dist.Shard(1), dist.Shard(2)],
                [8, 6, 4],
                [dist.Replicate(), dist.Replicate()],
                lambda idx: slice(None),
            ),
            ReshapeTestCase(
                [4, 6, 8],
                [dist.Shard(0), dist.Shard(2)],
                [8, 6, 4],
                [dist.Shard(0), dist.Replicate()],
                lambda idx: (idx // 2, idx // 2 + 4),
            ),
            ReshapeTestCase(
                [4, 6, 8],
                [dist.Shard(0, shard_order=0), dist.Shard(0, shard_order=1)],
                [8, 6, 4],
                [dist.Shard(0, shard_order=0), dist.Shard(0, shard_order=1)],
                lambda idx: slice(idx % 4 * 2, idx % 4 * 2 + 2),
            ),
            ReshapeTestCase(
                [4, 6, 8],
                [dist.Shard(2, shard_order=0), dist.Shard(2, shard_order=1)],
                [24, 2, 4],
                [dist.Replicate(), dist.Replicate()],
                lambda idx: slice(None),
            ),
            ReshapeTestCase(
                [4, 6, 8],
                [dist.Shard(2, shard_order=0), dist.Shard(1, shard_order=1)],
                [24, 4, 2],
                [dist.Shard(2, shard_order=0), dist.Shard(1, shard_order=1)],
                lambda idx: (slice(None), idx % 4, slice(None)),
            ),
        ]

    def run_test_case(self, test_case: ReshapeTestCase):
        a = paddle.rand(test_case.input_shape, "float32")
        input_placements = test_case.input_placements
        input = dist.shard_tensor(a, self.mesh, input_placements)
        out = paddle.reshape(input, test_case.target_shape)
        case_info = f"input_shape: {test_case.input_shape}, input_placements: {input_placements}, target_shape: {test_case.target_shape}"
        # Verify output shape
        np.testing.assert_equal(
            out.shape,
            test_case.target_shape,
            err_msg=f"Output shape mismatch when {case_info}. Expected: {test_case.target_shape}, Actual: {out.shape}",
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
        # Verify local_value if given
        if test_case.slice_funtor:
            idx = dist.get_rank()
            np.testing.assert_equal(
                out._local_value().numpy().flatten(),
                a[test_case.slice_funtor(idx)].numpy().flatten(),
                err_msg=f"Local values mismatch when {case_info}.",
            )

    def run_all_tests(self):
        self.setUp()
        for test_case in self.test_cases:
            self.run_test_case(test_case)


if __name__ == '__main__':
    TestReshapeCoShard().run_all_tests()
