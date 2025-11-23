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

import numpy as np

import paddle
from paddle.incubate.nn.functional import swiglu


class TestFusedWeightedSwigluBwd(unittest.TestCase):
    """Test cases for fused weighted swiglu backward function."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        paddle.seed(42)
        self.default_seq_len = 4096
        self.default_topk = 8
        self.default_moe_intermediate_size = 2048

    def generate_test_data(
        self, seq_len: int, topk: int, moe_intermediate_size: int
    ) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Generate test data for swiglu backward test.

        Args:
            seq_len: Sequence length
            topk: Top-k value
            moe_intermediate_size: MOE intermediate size

        Returns:
            Tuple of (o1, unzipped_probs, do2_s)
        """
        o1 = paddle.rand(
            [topk, seq_len, moe_intermediate_size * 2], dtype="bfloat16"
        )
        unzipped_probs = paddle.rand([topk, seq_len, 1], dtype="float32")
        do2_s = paddle.rand(
            [topk, seq_len, moe_intermediate_size], dtype="bfloat16"
        )

        return o1, unzipped_probs, do2_s

    def compute_gold_reference(
        self,
        o1: paddle.Tensor,
        unzipped_probs: paddle.Tensor,
        do2_s: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Compute gold reference results using separate operations.

        Args:
            o1: Input tensor 1
            unzipped_probs: Probability tensor
            do2_s: Gradient tensor

        Returns:
            Tuple of (do1, probs_grad, o2_s)
        """
        # Forward pass
        o2 = swiglu(o1)
        o2_s = o2 * unzipped_probs

        # Backward pass
        do2 = do2_s.cast(paddle.float32) * unzipped_probs
        do2 = do2.cast(paddle.bfloat16)
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)

        # Compute probability gradients
        probs_grad = (
            do2_s.cast(paddle.float32) * (o2.cast(paddle.float32))
        ).sum(axis=-1)

        return do1, probs_grad, o2_s

    def compute_fused_result(
        self,
        o1: paddle.Tensor,
        unzipped_probs: paddle.Tensor,
        do2_s: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Compute results using fused implementation.

        Args:
            o1: Input tensor 1
            unzipped_probs: Probability tensor
            do2_s: Gradient tensor

        Returns:
            Tuple of (do1, probs_grad, o2_s)
        """
        return paddle.incubate.nn.functional.fused_swiglu_weighted_bwd(
            o1, do2_s, unzipped_probs
        )

    def assert_tensors_close(
        self,
        tensor1: paddle.Tensor,
        tensor2: paddle.Tensor,
        rtol: float = 1e-2,
        atol: float = 1e-2,
        tensor_name: str = "tensor",
    ):
        """
        Assert that two tensors are close within tolerance.

        Args:
            tensor1: First tensor
            tensor2: Second tensor
            rtol: Relative tolerance
            atol: Absolute tolerance
            tensor_name: Name of tensor for error messages
        """
        # Convert to float32 for comparison
        t1_np = tensor1.astype("float32").numpy()
        t2_np = tensor2.astype("float32").numpy()

        # Check for NaN values
        nan_count_1 = np.sum(np.isnan(t1_np))
        nan_count_2 = np.sum(np.isnan(t2_np))

        self.assertEqual(
            nan_count_1, 0, f"{tensor_name}_1 contains {nan_count_1} NaN values"
        )
        self.assertEqual(
            nan_count_2, 0, f"{tensor_name}_2 contains {nan_count_2} NaN values"
        )

        # Check shapes match
        self.assertEqual(
            t1_np.shape,
            t2_np.shape,
            f"{tensor_name} shapes don't match: {t1_np.shape} vs {t2_np.shape}",
        )

        # Compare values
        try:
            np.testing.assert_allclose(t1_np, t2_np, rtol=rtol, atol=atol)
        except AssertionError as e:
            self.fail(f"{tensor_name} comparison failed: {e!s}")

    def test_default_configuration(self):
        """Test with default configuration parameters."""
        # Generate test data
        o1, unzipped_probs, do2_s = self.generate_test_data(
            self.default_seq_len,
            self.default_topk,
            self.default_moe_intermediate_size,
        )

        # Compute results
        do1_gold, pg_gold, o2_s_gold = self.compute_gold_reference(
            o1, unzipped_probs, do2_s
        )
        do1_fused, pg_fused, o2_s_fused = self.compute_fused_result(
            o1, unzipped_probs, do2_s
        )

        # Flatten probability gradients for comparison
        pg_gold_flat = pg_gold.flatten()
        pg_fused_flat = pg_fused.flatten()

        # Compare results
        self.assert_tensors_close(
            pg_gold_flat, pg_fused_flat, tensor_name="probs_grad"
        )
        self.assert_tensors_close(o2_s_gold, o2_s_fused, tensor_name="o2_s")
        self.assert_tensors_close(do1_gold, do1_fused, tensor_name="do1")

    def test_various_configurations(self):
        """Test with various configuration parameters."""
        test_cases = [
            (1024, 4, 512),
            (2048, 8, 1024),
            (512, 16, 256),
        ]

        for seq_len, topk, moe_intermediate_size in test_cases:
            with self.subTest(
                seq_len=seq_len,
                topk=topk,
                moe_intermediate_size=moe_intermediate_size,
            ):
                self._test_single_configuration(
                    seq_len, topk, moe_intermediate_size
                )

    def _test_single_configuration(
        self, seq_len: int, topk: int, moe_intermediate_size: int
    ):
        """
        Test a single configuration.

        Args:
            seq_len: Sequence length
            topk: Top-k value
            moe_intermediate_size: MOE intermediate size
        """
        # Generate test data
        o1, unzipped_probs, do2_s = self.generate_test_data(
            seq_len, topk, moe_intermediate_size
        )

        # Compute results
        do1_gold, pg_gold, o2_s_gold = self.compute_gold_reference(
            o1, unzipped_probs, do2_s
        )
        do1_fused, pg_fused, o2_s_fused = self.compute_fused_result(
            o1, unzipped_probs, do2_s
        )

        # Flatten probability gradients for comparison
        pg_gold_flat = pg_gold.flatten()
        pg_fused_flat = pg_fused.flatten()

        # Compare results
        self.assert_tensors_close(
            pg_gold_flat, pg_fused_flat, tensor_name="probs_grad"
        )
        self.assert_tensors_close(o2_s_gold, o2_s_fused, tensor_name="o2_s")
        self.assert_tensors_close(do1_gold, do1_fused, tensor_name="do1")

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        seq_len, topk, moe_intermediate_size = 1024, 4, 512
        o1, unzipped_probs, do2_s = self.generate_test_data(
            seq_len, topk, moe_intermediate_size
        )

        do1, pg, o2_s = self.compute_fused_result(o1, unzipped_probs, do2_s)

        # Check shapes
        expected_do1_shape = [topk, seq_len, moe_intermediate_size * 2]
        expected_pg_shape = [topk, seq_len, 1]
        expected_o2_s_shape = [topk, seq_len, moe_intermediate_size]

        self.assertEqual(
            list(do1.shape), expected_do1_shape, "do1 shape mismatch"
        )
        self.assertEqual(
            list(pg.shape), expected_pg_shape, "probs_grad shape mismatch"
        )
        self.assertEqual(
            list(o2_s.shape), expected_o2_s_shape, "o2_s shape mismatch"
        )

    def test_output_dtypes(self):
        """Test that output dtypes are correct."""
        seq_len, topk, moe_intermediate_size = 1024, 4, 512
        o1, unzipped_probs, do2_s = self.generate_test_data(
            seq_len, topk, moe_intermediate_size
        )

        do1, pg, o2_s = self.compute_fused_result(o1, unzipped_probs, do2_s)

        # Check dtypes
        self.assertEqual(
            do1.dtype, paddle.bfloat16, "do1 dtype should be bfloat16"
        )
        self.assertEqual(
            pg.dtype, paddle.float32, "probs_grad dtype should be float32"
        )
        self.assertEqual(
            o2_s.dtype, paddle.bfloat16, "o2_s dtype should be bfloat16"
        )

    def test_edge_cases(self):
        """Test edge cases."""
        # Test with minimum sizes
        seq_len, topk, moe_intermediate_size = 128, 1, 128
        o1, unzipped_probs, do2_s = self.generate_test_data(
            seq_len, topk, moe_intermediate_size
        )

        # Should not raise any errors
        do1_gold, pg_gold, o2_s_gold = self.compute_gold_reference(
            o1, unzipped_probs, do2_s
        )
        do1_fused, pg_fused, o2_s_fused = self.compute_fused_result(
            o1, unzipped_probs, do2_s
        )

        # Check for NaN values
        self.assertFalse(
            paddle.any(paddle.isnan(do1_fused.cast(paddle.float32)))
        )
        self.assertFalse(paddle.any(paddle.isnan(pg_fused)))
        self.assertFalse(
            paddle.any(paddle.isnan(o2_s_fused.cast(paddle.float32)))
        )

    def test_zero_inputs(self):
        """Test with zero inputs."""
        seq_len, topk, moe_intermediate_size = 512, 2, 256

        # Create zero tensors
        o1 = paddle.zeros(
            [topk, seq_len, moe_intermediate_size * 2], dtype="bfloat16"
        )
        unzipped_probs = paddle.ones(
            [topk, seq_len, 1], dtype="float32"
        )  # Use ones to avoid division by zero
        do2_s = paddle.zeros(
            [topk, seq_len, moe_intermediate_size], dtype="bfloat16"
        )

        # Should handle zeros gracefully
        do1, pg, o2_s = self.compute_fused_result(o1, unzipped_probs, do2_s)

        # Check for NaN values
        self.assertFalse(paddle.any(paddle.isnan(do1.cast(paddle.float32))))
        self.assertFalse(paddle.any(paddle.isnan(pg)))
        self.assertFalse(paddle.any(paddle.isnan(o2_s.cast(paddle.float32))))

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        seq_len, topk, moe_intermediate_size = 1024, 4, 512

        # First run
        paddle.seed(12345)
        o1_1, unzipped_probs_1, do2_s_1 = self.generate_test_data(
            seq_len, topk, moe_intermediate_size
        )
        do1_1, pg_1, o2_s_1 = self.compute_fused_result(
            o1_1, unzipped_probs_1, do2_s_1
        )

        # Second run with same seed
        paddle.seed(12345)
        o1_2, unzipped_probs_2, do2_s_2 = self.generate_test_data(
            seq_len, topk, moe_intermediate_size
        )
        do1_2, pg_2, o2_s_2 = self.compute_fused_result(
            o1_2, unzipped_probs_2, do2_s_2
        )

        # Results should be identical
        self.assert_tensors_close(
            do1_1, do1_2, rtol=0, atol=0, tensor_name="do1_reproducibility"
        )
        self.assert_tensors_close(
            pg_1, pg_2, rtol=0, atol=0, tensor_name="pg_reproducibility"
        )
        self.assert_tensors_close(
            o2_s_1, o2_s_2, rtol=0, atol=0, tensor_name="o2_s_reproducibility"
        )


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
