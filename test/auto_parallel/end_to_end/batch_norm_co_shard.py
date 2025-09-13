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


class BatchNormTestCase:
    def __init__(
        self,
        input_shape: list[int],
        input_placements: list[dist.Placement],
        output_placements: list[dist.Placement],
        weight_placements: list[dist.Placement],
        bias_placements: list[dist.Placement],
        running_mean_placements: list[dist.Placement],
        running_var_placements: list[dist.Placement],
        slice_funtor: Callable[[int], Any] | None = None,
    ):
        self.input_shape = input_shape
        self.input_placements = input_placements
        self.output_placements = output_placements
        self.weight_placements = weight_placements
        self.bias_placements = bias_placements
        self.running_mean_placements = running_mean_placements
        self.running_var_placements = running_var_placements
        self.slice_funtor = slice_funtor


class BatchNormGradTestCase:
    def __init__(
        self,
        input_shape: list[int],
        output_placements: list[dist.Placement],
        out_grad_placements: list[dist.Placement],
        x_grad_placements: list[dist.Placement],
        weight_grad_placements: list[dist.Placement],
        bias_grad_placements: list[dist.Placement],
    ):
        self.input_shape = input_shape
        self.output_placements = output_placements
        self.out_grad_placements = out_grad_placements
        self.x_grad_placements = x_grad_placements
        self.weight_grad_placements = weight_grad_placements
        self.bias_grad_placements = bias_grad_placements


class TestBatchNormCoShard:
    def setUp(self):
        self.mesh = dist.ProcessMesh(
            [[0, 1], [2, 3]], dim_names=['x', 'y']
        )
        self.test_cases_forward = [
            # Shard on batch dimension
            BatchNormTestCase(
                [32, 48, 128, 256],
                [dist.Shard(dim=1,shard_order = 0), dist.Shard(dim=1,shard_order = 1)],
                [dist.Shard(dim=1,shard_order = 0), dist.Shard(dim=1,shard_order = 1)],
                [dist.Replicate()],
                [dist.Replicate()],
                [dist.Shard(dim=1,shard_order = 0), dist.Shard(dim=1,shard_order = 1)],
                [dist.Shard(dim=1,shard_order = 0), dist.Shard(dim=1,shard_order = 1)]
            ),
            # Shard on channel dimension
            BatchNormTestCase(
                [32, 48, 128, 256],
                [dist.Shard(0), dist.Shard(1)],
                [dist.Shard(0), dist.Shard(1)],
                [dist.Replicate()],
                [dist.Replicate()],
                [dist.Shard(1)],
                [dist.Shard(1)],
            ),
        ]
        self.test_cases_backward = [
            # Shard on batch dimension
            BatchNormGradTestCase(
                [32, 48, 128, 256],
                [dist.Shard(dim=1,shard_order = 0), dist.Shard(dim=1,shard_order = 1)],
                [dist.Shard(dim=1,shard_order = 0), dist.Shard(dim=1,shard_order = 1)],
                [dist.Shard(dim=1,shard_order = 0), dist.Shard(dim=1,shard_order = 1)],
                [dist.Replicate()],
                [dist.Replicate()],
            ),
            # Shard on channel dimension
            BatchNormGradTestCase(
                [32, 48, 128, 256],
                [dist.Shard(0), dist.Shard(1)],
                [dist.Shard(0), dist.Shard(1)],
                [dist.Shard(0), dist.Shard(1)],
                [dist.Replicate()],
                [dist.Replicate()],
            ),
        ]

    def run_test_case_forward(self, test_case: BatchNormTestCase):
        a = paddle.rand(test_case.input_shape, "float32")
        input_placements = test_case.input_placements
        input = dist.shard_tensor(a, self.mesh, input_placements)
        bn = paddle.nn.BatchNorm2D(test_case.input_shape[1])

        weight = dist.shard_tensor(
            bn.weight, self.mesh, test_case.weight_placements
        )
        bias = dist.shard_tensor(bn.bias, self.mesh, test_case.bias_placements)
        running_mean = dist.shard_tensor(
            bn._mean, self.mesh, test_case.running_mean_placements
        )
        running_var = dist.shard_tensor(
            bn._variance, self.mesh, test_case.running_var_placements
        )
        bn.weight = weight
        bn.bias = bias
        bn._mean = running_mean
        bn._variance = running_var

        out = bn(input)
        case_info = f"input_shape: {test_case.input_shape}, input_placements: {input_placements}"

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

    def run_test_case_backward(self, test_case: BatchNormGradTestCase):
        a = paddle.rand(test_case.input_shape, "float32")
        a.stop_gradient = False
        input_placements = [
            dist.Replicate() for _ in range(len(test_case.input_shape))
        ]
        input = dist.shard_tensor(a, self.mesh, input_placements)
        bn = paddle.nn.BatchNorm2D(test_case.input_shape[1])
        out = bn(input)
        out = dist.reshard(out, self.mesh, test_case.output_placements)

        out_grad = paddle.ones(out.shape, "float32")
        out_grad = dist.shard_tensor(
            out_grad, self.mesh, test_case.out_grad_placements
        )

        (x_grad, weight_grad, bias_grad) = paddle.grad(
            [out], [input, bn.weight, bn.bias], [out_grad]
        )

        case_info = f"input_shape: {test_case.input_shape}, out_placements: {test_case.output_placements}, out_grad_placements: {test_case.out_grad_placements}"

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
    TestBatchNormCoShard().run_all_tests()