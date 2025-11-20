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


class TestActQuantDequant(unittest.TestCase):
    """Test cases for activation quantization and dequantization functions."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        paddle.set_default_dtype('float32')

    def act_quant(
        self, x: paddle.Tensor, block_size: int = 128
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """
        Quantize activation tensor to float8_e4m3fn format.

        Args:
            x: Input tensor to quantize
            block_size: Block size for quantization (default: 128)

        Returns:
            Tuple of (quantized_tensor, scale_factors)
        """
        self.assertTrue(x.is_contiguous(), "Input tensor must be contiguous")
        self.assertEqual(
            x.shape[-1] % block_size,
            0,
            f"Last dimension size must be divisible by block_size (block_size={block_size})",
        )

        # Convert to float32 for computation
        x_float = x.astype('float32')

        # Reshape to process blocks
        original_shape = x.shape
        # Reshape to (..., num_blocks, block_size)
        new_shape = [
            *original_shape[:-1],
            original_shape[-1] // block_size,
            block_size,
        ]
        x_reshaped = x_float.reshape(new_shape)

        # Compute scaling factors: max(abs(x)) / 448.0 for each block
        abs_x = paddle.abs(x_reshaped)
        max_vals = paddle.max(
            abs_x, axis=-1, keepdim=False
        )  # (..., num_blocks)
        s = max_vals / 448.0

        # Expand scaling factors to match x_reshaped shape for division
        s_expanded = s.unsqueeze(-1)  # (..., num_blocks, 1)

        # Quantize: x / s
        y_reshaped = x_reshaped / s_expanded

        # Reshape back to original shape
        y_float = y_reshaped.reshape(original_shape)

        # Convert to target dtype
        y = y_float.astype('float8_e4m3fn')

        return y, s

    def dequant_ref(
        self, x: paddle.Tensor, s: paddle.Tensor, block_size: int = 128
    ) -> paddle.Tensor:
        """
        Reference implementation for dequantizing activation tensor.

        Args:
            x: Quantized tensor
            s: Scale factors
            block_size: Block size used in quantization (default: 128)

        Returns:
            Dequantized tensor
        """
        self.assertTrue(
            x.is_contiguous() and s.is_contiguous(),
            "Input tensors must be contiguous",
        )
        self.assertEqual(x.dim(), 2, "Input tensor x must have 2 dimensions")
        self.assertEqual(s.dim(), 2, "Input tensor s must have 2 dimensions")

        M, N = x.shape

        # Convert to float32 for computation
        x_float = x.astype('float32')

        # Check if s needs to be expanded to match x shape
        if s.shape[1] == N // block_size:
            # s has shape (M, N//block_size), need to expand to (M, N)
            # Reshape s to (M, N//block_size, 1) then repeat along last dimension
            s_expanded = s.unsqueeze(-1)  # (M, N//block_size, 1)
            s_expanded = paddle.tile(
                s_expanded, [1, 1, block_size]
            )  # (M, N//block_size, block_size)
            s_expanded = s_expanded.reshape([M, N])  # (M, N)
        else:
            # s already has shape (M, N)
            s_expanded = s

        # Dequantize: x * s
        y = x_float * s_expanded

        # Convert to default dtype
        y = y.astype(paddle.get_default_dtype())

        return y

    def test_act_quant_basic_functionality(self):
        """Test basic functionality of act_quant function."""
        # Test with simple case
        x = paddle.randn([4, 256]).astype("bfloat16")
        x = paddle.clip(x, min=-10, max=10)

        x_fp8, scale = self.act_quant(x, block_size=128)

        # Check output shapes
        self.assertEqual(x_fp8.shape, x.shape)
        self.assertEqual(scale.shape, [4, 2])  # 256 // 128 = 2

        # Check output dtypes
        self.assertEqual(x_fp8.dtype, paddle.float8_e4m3fn)
        self.assertEqual(scale.dtype, paddle.float32)

    def test_act_dequant_consistency_small(self):
        """Test consistency between reference and fused implementations with small tensors."""
        test_cases = [
            (512, 7168),
            (2048, 7168),
            (4096, 7168),
        ]

        for height, width in test_cases:
            with self.subTest(height=height, width=width):
                self._test_single_case(height, width)

    def test_act_dequant_consistency_various_sizes(self):
        """Test with various tensor sizes."""
        test_cases = [
            (128, 256),
            (256, 512),
            (1024, 2048),
        ]

        for height, width in test_cases:
            with self.subTest(height=height, width=width):
                self._test_single_case(height, width)

    def _test_single_case(
        self, height: int, width: int, rtol: float = 1e-2, atol: float = 1e-2
    ):
        """
        Test a single case with given dimensions.

        Args:
            height: Tensor height
            width: Tensor width
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
        """
        # Generate test data
        x = paddle.clip(
            paddle.randn([height, width]).astype("bfloat16"), min=-50, max=50
        )

        # Perform quantization
        x_fp8, scale = self.act_quant(x)

        # Get results from both implementations
        if hasattr(paddle.incubate.nn.functional, 'fused_act_dequant'):
            dequant_result_fused = (
                paddle.incubate.nn.functional.fused_act_dequant(x_fp8, scale)
            )
        else:
            # Skip fused test if not available
            self.skipTest(
                "fused_act_dequant not available in this Paddle version"
            )

        dequant_result_ref = self.dequant_ref(x_fp8, scale)

        # Convert to numpy for comparison
        fused_np = dequant_result_fused.astype("float32").numpy()
        ref_np = dequant_result_ref.astype("float32").numpy()

        # Check for NaN values
        nan_cnt_fused = np.sum(np.isnan(fused_np))
        nan_cnt_ref = np.sum(np.isnan(ref_np))

        self.assertEqual(
            nan_cnt_fused,
            0,
            f"Fused result contains {nan_cnt_fused} NaN values",
        )
        self.assertEqual(
            nan_cnt_ref,
            0,
            f"Reference result contains {nan_cnt_ref} NaN values",
        )

        # Compare results
        try:
            np.testing.assert_allclose(fused_np, ref_np, rtol=rtol, atol=atol)
        except AssertionError as e:
            self.fail(
                f"Results don't match for shape [{height}, {width}]: {e!s}"
            )

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Test non-divisible block size
        x = paddle.randn([4, 255]).astype(
            "bfloat16"
        )  # 255 is not divisible by 128

        with self.assertRaises(AssertionError):
            self.act_quant(x, block_size=128)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)
