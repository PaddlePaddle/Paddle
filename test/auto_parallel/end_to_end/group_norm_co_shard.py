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


class GroupNormTestCase:
    def __init__(
        self,
        input_shape: list[int],
        num_groups: int,
        input_placements: list[dist.Placement],
        output_placements: list[dist.Placement],
        weight_placements: list[dist.Placement],
        bias_placements: list[dist.Placement],
        slice_funtor: Callable[[int], Any] | None = None,
    ):
        self.input_shape = input_shape
        self.num_groups = num_groups
        self.input_placements = input_placements
        self.output_placements = output_placements
        self.weight_placements = weight_placements
        self.bias_placements = bias_placements
        self.slice_funtor = slice_funtor


class GroupNormGradTestCase:
    def __init__(
        self,
        input_shape: list[int],
        num_groups: int,
        output_placements: list[dist.Placement],
        out_grad_placements: list[dist.Placement],
        x_grad_placements: list[dist.Placement],
        weight_grad_placements: list[dist.Placement],
        bias_grad_placements: list[dist.Placement],
    ):
        self.input_shape = input_shape
        self.num_groups = num_groups
        self.output_placements = output_placements
        self.out_grad_placements = out_grad_placements
        self.x_grad_placements = x_grad_placements
        self.weight_grad_placements = weight_grad_placements
        self.bias_grad_placements = bias_grad_placements


class TestGroupNormCoShard:
    def setUp(self):
        self.mesh = dist.ProcessMesh(
            [[0, 1], [2, 3]], dim_names=['x', 'y']
        )
        self.test_cases_forward = [
            GroupNormTestCase(
                [32, 48, 128, 256],
                4,
                [dist.Shard(dim=0,shard_order = 0), dist.Shard(dim=0,shard_order = 1)],
                [dist.Shard(dim=0,shard_order = 0), dist.Shard(dim=0,shard_order = 1)],
                [dist.Replicate()],
                [dist.Replicate()],
            ),
            GroupNormTestCase(
                [32, 48, 128, 256],
                4,
                [dist.Shard(0), dist.Shard(1)],
                [dist.Shard(0), dist.Shard(1)],
                [dist.Replicate()],
                [dist.Replicate()],
            ),
        ]
        self.test_cases_backward = [
            GroupNormGradTestCase(
                [32, 48, 128, 256],
                4,
                [dist.Shard(dim=0,shard_order = 0), dist.Shard(dim=0,shard_order = 1)],
                [dist.Shard(dim=0,shard_order = 0), dist.Shard(dim=0,shard_order = 1)],
                [dist.Shard(dim=0,shard_order = 0), dist.Shard(dim=0,shard_order = 1)],
                [dist.Replicate()],
                [dist.Replicate()],
            ),
            GroupNormGradTestCase(
                [32, 48, 128, 256],
                4,
                [dist.Shard(0), dist.Shard(1)],
                [dist.Shard(0), dist.Shard(1)],
                [dist.Shard(0), dist.Shard(1)],
                [dist.Replicate()],
                [dist.Replicate()],
            ),
        ]

    def run_test_case_forward(self, test_case: GroupNormTestCase):
        a = paddle.rand(test_case.input_shape, "float32")
        input_placements = test_case.input_placements
        input = dist.shard_tensor(a, self.mesh, input_placements)
        gn = paddle.nn.GroupNorm(
            test_case.num_groups, test_case.input_shape[1]
        )
        weight = dist.shard_tensor(
            gn.weight, self.mesh, test_case.weight_placements
        )
        bias = dist.shard_tensor(gn.bias, self.mesh, test_case.bias_placements)
        gn.weight = weight
        gn.bias = bias
        out = gn(input)
        case_info = f"input_shape: {test_case.input_shape}, input_placements: {input_placements}, num_groups: {test_case.num_groups}"
        # Verify output shape
        np.testing.assert_equal(
            out.shape,
            test_case.input_shape,
            err_msg=f"Output shape mismatch when {case_info}. Expected: {test_case.input_shape}, Actual: {out.shape}",
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

    def run_test_case_backward(self, test_case: GroupNormGradTestCase):
        a = paddle.rand(test_case.input_shape, "float32")
        a.stop_gradient = False
        input_placements = [
            dist.Replicate() for _ in range(len(test_case.input_shape))
        ]
        input = dist.shard_tensor(a, self.mesh, input_placements)
        gn = paddle.nn.GroupNorm(
            test_case.num_groups, test_case.input_shape[1]
        )
        out = gn(input)
        out = dist.reshard(out, self.mesh, test_case.output_placements)

        out_grad = paddle.ones(out.shape, "float32")
        out_grad = dist.shard_tensor(
            out_grad, self.mesh, test_case.out_grad_placements
        )

        (x_grad, weight_grad, bias_grad) = paddle.grad(
            [out], [input, gn.weight, gn.bias], [out_grad]
        )

        case_info = f"input_shape: {test_case.input_shape}, num_groups: {test_case.num_groups}, out_placements: {test_case.output_placements}, out_grad_placements: {test_case.out_grad_placements}"
        # Verify output shape
        np.testing.assert_equal(
            x_grad.shape,
            test_case.input_shape,
            err_msg=f"x_grad shape mismatch when {case_info}. Expected: {test_case.input_shape}, Actual: {x_grad.shape}",
        )
        np.testing.assert_equal(
            weight_grad.shape,
            [test_case.input_shape[1]],
            err_msg=f"weight_grad shape mismatch when {case_info}. Expected: {[test_case.input_shape[1]]}, Actual: {weight_grad.shape}",
        )
        np.testing.assert_equal(
            bias_grad.shape,
            [test_case.input_shape[1]],
            err_msg=f"bias_grad shape mismatch when {case_info}. Expected: {[test_case.input_shape[1]]}, Actual: {bias_grad.shape}",
        )

        # Verify placements
        assert x_grad.placements
        for actual, expected in zip(
            x_grad.placements, test_case.x_grad_placements
        ):
            np.testing.assert_equal(
                actual,
                expected,
                err_msg=f"x_grad placements mismatch when {case_info}. Expected: {test_case.x_grad_placements}, Actual: {x_grad.placements}",
            )
        assert weight_grad.placements
        for actual, expected in zip(
            weight_grad.placements, test_case.weight_grad_placements
        ):
            np.testing.assert_equal(
                actual,
                expected,
                err_msg=f"weight_grad placements mismatch when {case_info}. Expected: {test_case.weight_grad_placements}, Actual: {weight_grad.placements}",
            )
        assert bias_grad.placements
        for actual, expected in zip(
            bias_grad.placements, test_case.bias_grad_placements
        ):
            np.testing.assert_equal(
                actual,
                expected,
                err_msg=f"bias_grad placements mismatch when {case_info}. Expected: {test_case.bias_grad_placements}, Actual: {bias_grad.placements}",
            )

    def run_all_tests(self):
        self.setUp()
        for test_case in self.test_cases_forward:
            self.run_test_case_forward(test_case)
        for test_case in self.test_cases_backward:
            self.run_test_case_backward(test_case)


if __name__ == '__main__':
    TestGroupNormCoShard().run_all_tests()