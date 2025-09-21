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


class LayerNormTestCase:
    """
    A data class to hold parameters for a LayerNorm operation test case.
    """

    def __init__(
        self,
        input_shape: list[int],
        normalized_shape: list[int] | int,
        input_placements: Sequence[dist.Placement],
        expected_input_placements: Sequence[dist.Placement],
        expected_output_placements: Sequence[dist.Placement],
    ):
        self.input_shape = input_shape
        self.normalized_shape = normalized_shape
        self.input_placements = input_placements
        self.expected_input_placements = expected_input_placements
        self.expected_output_placements = expected_output_placements


class TestLayerNormCoShard:
    """
    Unit tests for co_shard SPMD rule on LayerNorm operation.
    """

    def setUp(self):
        """
        Initializes the process mesh and defines the test cases.
        """
        self.mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])

        # After resharding, the input and output are expected to be sharded on dimension 0
        # across the flattened mesh.
        expected_placements = [
            dist.Shard(dim=0, shard_order=0),
            dist.Shard(dim=0, shard_order=1),
        ]

        self.test_cases = [
            LayerNormTestCase(
                input_shape=[64, 32, 128, 128],
                normalized_shape=[32, 128, 128],
                input_placements=[
                    dist.Shard(dim=0, shard_order=0),
                    dist.Shard(dim=0, shard_order=1),
                ],
                expected_input_placements=expected_placements,
                expected_output_placements=expected_placements,
            )
        ]

    def run_test_case(self, test_case: LayerNormTestCase):
        # Prepare inputs and model
        x = paddle.rand(test_case.input_shape, dtype="float32")
        layer_norm = paddle.nn.LayerNorm(test_case.normalized_shape)

        # Shard tensor
        input_tensor = dist.shard_tensor(
            x, self.mesh, test_case.input_placements
        )

        # Perform operation
        out = layer_norm(input_tensor)

        case_info = f"input_placements: {test_case.input_placements}"

        # Verify placements of the input tensor (post-operation, due to resharding)
        assert input_tensor.placements
        for actual, expected in zip(
            input_tensor.placements, test_case.expected_input_placements
        ):
            np.testing.assert_equal(
                actual,
                expected,
                err_msg=f"Input placements mismatch when {case_info}. Expected: {test_case.expected_input_placements}, Actual: {input_tensor.placements}",
            )

        # Verify output shape
        np.testing.assert_equal(
            out.shape,
            test_case.input_shape,
            err_msg=f"Output shape mismatch when {case_info}. Expected: {test_case.input_shape}, Actual: {out.shape}",
        )

        # Verify output placements
        assert out.placements
        for actual, expected in zip(
            out.placements, test_case.expected_output_placements
        ):
            np.testing.assert_equal(
                actual,
                expected,
                err_msg=f"Output placements mismatch when {case_info}. Expected: {test_case.expected_output_placements}, Actual: {out.placements}",
            )

    def run_all_tests(self):
        self.setUp()
        for test_case in self.test_cases:
            self.run_test_case(test_case)


if __name__ == '__main__':
    TestLayerNormCoShard().run_all_tests()
