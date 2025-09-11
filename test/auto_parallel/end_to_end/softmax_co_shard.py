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


class SoftmaxTestCase:
    def __init__(
        self,
        input_shape: list[int],
        input_placements: list[dist.Placement],
        axis: int,
        output_shape: list[int],
        output_placements: list[dist.Placement],
        slice_funtor: Callable[[int], Any] | None = None,
    ):
        self.input_shape = input_shape
        self.input_placements = input_placements
        self.axis = axis
        self.output_shape = output_shape
        self.output_placements = output_placements
        self.slice_funtor = slice_funtor


class SoftmaxGradTestCase:
    def __init__(
        self,
        input_shape: list[int],
        axis: int,
        output_shape: list[int],
        output_placements: list[dist.Placement],
        out_grad_placements: list[dist.Placement],
        x_grad_placements: list[dist.Placement],
    ):
        self.input_shape = input_shape
        self.axis = axis
        self.output_shape = output_shape
        self.output_placements = output_placements
        self.out_grad_placements = out_grad_placements
        self.x_grad_placements = x_grad_placements


class TestSoftmaxCoShard:
    def setUp(self):
        self.mesh = dist.ProcessMesh(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=['x', 'y', 'z']
        )
        self.test_cases_forward = [
            SoftmaxTestCase(
                [32, 48, 128],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                0,
                [32, 48, 128],
                [dist.Replicate(), dist.Replicate(), dist.Shard(1)],
            ),
            SoftmaxTestCase(
                [32, 48, 128],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                -3,
                [32, 48, 128],
                [dist.Replicate(), dist.Replicate(), dist.Shard(1)],
            ),
            SoftmaxTestCase(
                [32, 48, 128],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                1,
                [32, 48, 128],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
            ),
        ]
        self.test_cases_backward = [
            SoftmaxGradTestCase(
                [32, 48, 128],
                0,
                [32, 48, 128],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                [dist.Replicate(), dist.Replicate(), dist.Shard(1)],
            ),
            SoftmaxGradTestCase(
                [32, 48, 128],
                0,
                [32, 48, 128],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                [
                    dist.Shard(0),
                    dist.Shard(1, shard_order=0),
                    dist.Shard(1, shard_order=1),
                ],
                [
                    dist.Replicate(),
                    dist.Shard(1, shard_order=0),
                    dist.Shard(1, shard_order=1),
                ],
            ),
            SoftmaxGradTestCase(
                [32, 48, 128],
                1,
                [32, 48, 128],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(1),
                ],
                [
                    dist.Shard(1, shard_order=0),
                    dist.Shard(1, shard_order=1),
                    dist.Shard(0),
                ],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(0, shard_order=2),
                ],
            ),
            SoftmaxGradTestCase(
                [32, 48, 128],
                1,
                [32, 48, 128],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
                [dist.Replicate(), dist.Replicate(), dist.Shard(2)],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Shard(2),
                ],
            ),
            SoftmaxGradTestCase(
                [32, 48, 128],
                -1,
                [32, 48, 128],
                [
                    dist.Shard(0),
                    dist.Shard(1),
                    dist.Replicate(),
                ],
                [
                    dist.Shard(1, shard_order=0),
                    dist.Shard(1, shard_order=1),
                    dist.Replicate(),
                ],
                [
                    dist.Shard(1, shard_order=0),
                    dist.Shard(1, shard_order=1),
                    dist.Replicate(),
                ],
            ),
            SoftmaxGradTestCase(
                [32, 48, 128],
                -1,
                [32, 48, 128],
                [dist.Shard(0), dist.Shard(1), dist.Replicate()],
                [
                    dist.Shard(0, shard_order=0),
                    dist.Shard(0, shard_order=1),
                    dist.Replicate(),
                ],
                [dist.Shard(0), dist.Shard(1), dist.Replicate()],
            ),
            SoftmaxGradTestCase(
                [32, 48, 128],
                -1,
                [32, 48, 128],
                [
                    dist.Shard(0),
                    dist.Shard(1, shard_order=0),
                    dist.Shard(1, shard_order=1),
                ],
                [
                    dist.Shard(1, shard_order=0),
                    dist.Shard(1, shard_order=1),
                    dist.Replicate(),
                ],
                [
                    dist.Shard(1, shard_order=0),
                    dist.Shard(1, shard_order=1),
                    dist.Shard(1, shard_order=2),
                ],
            ),
        ]

    def run_test_case_forward(self, test_case: SoftmaxTestCase):
        paddle.seed(2025)
        a = paddle.rand(test_case.input_shape, "float32")
        input_placements = test_case.input_placements
        input = dist.shard_tensor(a, self.mesh, input_placements)
        out = paddle.nn.functional.softmax(input, test_case.axis)
        case_info = f"input_shape: {test_case.input_shape}, input_placements: {input_placements}, axis: {test_case.axis}"
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
        # Verify local_value if given
        if test_case.slice_funtor:
            idx = dist.get_rank()
            np.testing.assert_equal(
                out._local_value().numpy().flatten(),
                a[test_case.slice_funtor(idx)].numpy().flatten(),
                err_msg=f"Local values mismatch when {case_info}.",
            )

    def run_test_case_backward(self, test_case: SoftmaxGradTestCase):
        a = paddle.rand(test_case.input_shape, "float32")
        a.stop_gradient = False
        input_placements = [dist.Replicate() for _ in range(self.mesh.ndim)]
        input = dist.shard_tensor(a, self.mesh, input_placements)
        out = paddle.nn.functional.softmax(input, test_case.axis)
        out = dist.reshard(out, self.mesh, test_case.output_placements)

        out_grad = paddle.ones(out.shape, "float32")
        out_grad = dist.shard_tensor(
            out_grad, self.mesh, test_case.out_grad_placements
        )

        (x_grad,) = paddle.grad([out], input, [out_grad])

        case_info = f"input_shape: {test_case.input_shape}, axis: {test_case.axis}, out_placements: {test_case.output_placements}, out_grad_placements: {test_case.out_grad_placements}"
        # Verify output shape
        np.testing.assert_equal(
            x_grad.shape,
            test_case.input_shape,
            err_msg=f"Output shape mismatch when {case_info}. Expected: {test_case.input_shape}, Actual: {x_grad.shape}",
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
    TestSoftmaxCoShard().run_all_tests()
