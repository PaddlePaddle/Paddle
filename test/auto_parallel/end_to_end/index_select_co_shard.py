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


class IndexSelectTestCase:
    def __init__(
        self,
        x_shape: list[int],
        x_placements: list[dist.Placement],
        index_shape: list[int],
        index_placements: list[dist.Placement],
        axis: int,
        out_shape: list[int],
        out_placements: list[dist.Placement],
    ):
        self.x_shape = x_shape
        self.x_placements = x_placements
        self.index_shape = index_shape
        self.index_placements = index_placements
        self.axis = axis
        self.out_shape = out_shape
        self.out_placements = out_placements


class IndexSelectGradTestCase:
    def __init__(
        self,
        x_shape: list[int],
        x_placements: list[dist.Placement],
        index_shape: list[int],
        index_placements: list[dist.Placement],
        axis: int,
        out_grad_shape: list[int],
        out_grad_placements: list[dist.Placement],
        x_grad_placements: list[dist.Placement],
    ):
        self.x_shape = x_shape
        self.x_placements = x_placements
        self.index_shape = index_shape
        self.index_placements = index_placements
        self.axis = axis
        self.out_grad_shape = out_grad_shape
        self.out_grad_placements = out_grad_placements
        self.x_grad_placements = x_grad_placements


class TestIndexSelectCoShard:
    def setUp(self):
        self.mesh = dist.ProcessMesh(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=['x', 'y', 'z']
        )
        self.test_cases_forward = [
            IndexSelectTestCase(
                [8, 16, 32],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                [8],
                [dist.Replicate(), dist.Replicate(), dist.Replicate()],
                1,
                [8, 8, 32],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
            ),
            IndexSelectTestCase(
                [8, 16, 32],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                [8],
                [dist.Replicate(), dist.Replicate(), dist.Shard(0)],
                1,
                [8, 8, 32],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
            ),
            IndexSelectTestCase(
                [8, 16, 32],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                [8],
                [dist.Shard(0), dist.Replicate(), dist.Replicate()],
                1,
                [8, 8, 32],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
            ),
            IndexSelectTestCase(
                [8, 16, 32],
                [dist.Replicate(), dist.Replicate(), dist.Shard(0)],
                [8],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
                1,
                [8, 8, 32],
                [
                    dist.Shard(1, shard_order=0),
                    dist.Shard(1, shard_order=1),
                    dist.Shard(0),
                ],
            ),
            IndexSelectTestCase(
                [8, 16, 32],
                [dist.Shard(0), dist.Replicate(), dist.Replicate()],
                [8],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
                1,
                [8, 8, 32],
                [dist.Shard(0), dist.Shard(1), dist.Replicate()],
            ),
        ]
        self.test_cases_backward = [
            IndexSelectGradTestCase(
                [8, 16, 32],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                [8],
                [dist.Replicate(), dist.Replicate(), dist.Replicate()],
                1,
                [8, 8, 32],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
            ),
            IndexSelectGradTestCase(
                [8, 16, 32],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                [8],
                [dist.Replicate(), dist.Replicate(), dist.Shard(0)],
                1,
                [8, 8, 32],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Partial(),
                ],
            ),
            IndexSelectGradTestCase(
                [8, 16, 32],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                [8],
                [dist.Shard(0), dist.Replicate(), dist.Replicate()],
                1,
                [8, 8, 32],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
            ),
            IndexSelectGradTestCase(
                [8, 16, 32],
                [dist.Replicate(), dist.Replicate(), dist.Shard(0)],
                [8],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
                1,
                [8, 8, 32],
                [
                    dist.Shard(1, shard_order=0),
                    dist.Shard(1, shard_order=1),
                    dist.Shard(0),
                ],
                [dist.Partial(), dist.Partial(), dist.Shard(0)],
            ),
            IndexSelectGradTestCase(
                [8, 16, 32],
                [dist.Shard(0), dist.Replicate(), dist.Replicate()],
                [8],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
                1,
                [8, 8, 32],
                [dist.Shard(0), dist.Shard(1), dist.Replicate()],
                [dist.Shard(0), dist.Partial(), dist.Replicate()],
            ),
        ]

    def run_test_case_forward(self, test_case: IndexSelectTestCase):
        x = paddle.rand(test_case.x_shape, "float32")
        x_placements = test_case.x_placements
        x = dist.shard_tensor(x, self.mesh, x_placements)
        index = paddle.randint(
            0,
            test_case.x_shape[test_case.axis],
            test_case.index_shape,
            dtype="int32",
        )
        index_placements = test_case.index_placements
        index = dist.shard_tensor(index, self.mesh, index_placements)

        out = paddle.index_select(x, index, test_case.axis)
        case_info = f"x_shape: {test_case.x_shape}, x_placements: {x_placements}, index_shape: {test_case.index_shape}, index_placements: {index_placements}, axis: {test_case.axis}"
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

    def run_test_case_backward(self, test_case: IndexSelectGradTestCase):
        x = paddle.rand(test_case.x_shape, "float32")
        x.stop_gradient = False
        x_placements = test_case.x_placements
        x = dist.shard_tensor(x, self.mesh, x_placements)

        index = paddle.randint(
            0,
            test_case.x_shape[test_case.axis],
            test_case.index_shape,
            dtype="int32",
        )
        index_placements = test_case.index_placements
        index = dist.shard_tensor(index, self.mesh, index_placements)

        out = paddle.index_select(x, index, test_case.axis)

        out_grad = paddle.ones(out.shape, "float32")
        out_grad = dist.shard_tensor(
            out_grad, self.mesh, test_case.out_grad_placements
        )

        (x_grad,) = paddle.grad([out], x, [out_grad])

        case_info = f"x_shape: {test_case.x_shape}, x_placements: {test_case.x_placements}, index_shape: {test_case.index_shape}, index_placements: {test_case.index_placements}, axis: {test_case.axis}, out_grad_shape: {test_case.out_grad_shape}, out_grad_placements: {test_case.out_grad_placements}"
        # Verify output shape
        np.testing.assert_equal(
            x_grad.shape,
            test_case.x_shape,
            err_msg=f"Output shape mismatch when {case_info}. Expected: {test_case.x_shape}, Actual: {x_grad.shape}",
        )

        # Verify placements
        assert x_grad.placements
        for actual, expected in zip(
            x_grad.placements, test_case.x_grad_placements
        ):
            np.testing.assert_equal(
                actual,
                expected,
                err_msg=f"Output placements mismatch when {case_info}. Expected: {test_case.x_grad_placements}, Actual: {x_grad.placements}",
            )

    def run_all_tests(self):
        self.setUp()
        for test_case in self.test_cases_forward:
            self.run_test_case_forward(test_case)


if __name__ == '__main__':
    TestIndexSelectCoShard().run_all_tests()
