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
import paddle.incubate.nn.functional as F


class TestFusedWeightedSwigluActQuant(unittest.TestCase):
    """Test cases for paddle.fused_weighted_swiglu_act_quant function"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        paddle.seed(42)
        np.random.seed(42)

    def dequantize_fp8_to_bf16(
        self, fp8_tensor: paddle.Tensor, scale: paddle.Tensor
    ) -> paddle.Tensor:
        """Helper function to dequantize fp8 tensor to bf16"""
        expanded_scale = paddle.repeat_interleave(scale, repeats=128, axis=-1)
        # Handle non-aligned cases by truncating
        expanded_scale = expanded_scale[:, : fp8_tensor.shape[-1]]
        return fp8_tensor.astype('float32') * expanded_scale

    def _test_single_case(self, height: int, width: int):
        """Test single case with given height and width"""
        # Generate test data
        x = paddle.clip(
            paddle.randn([height, width]).astype("bfloat16"), min=-50, max=50
        )
        prob = paddle.randn([height, 1]).astype("float32")

        # Compute golden result
        golden_res = F.swiglu(x) * prob

        # Compute fused result
        fused_res, fused_scales = (
            paddle.incubate.nn.functional.fused_weighted_swiglu_act_quant(
                x, prob, using_pow2_scaling=False
            )
        )

        # Dequantize fused result
        dequantized_res = self.dequantize_fp8_to_bf16(fused_res, fused_scales)

        # Convert to numpy for comparison
        golden_np = golden_res.astype("float32").numpy()
        fused_np = dequantized_res.numpy()

        # Check for NaN values
        nan_cnt_golden = np.sum(np.isnan(golden_np))
        nan_cnt_fused = np.sum(np.isnan(fused_np))

        # Assert no NaN values
        self.assertEqual(
            nan_cnt_golden,
            0,
            f"Golden result contains {nan_cnt_golden} NaN values",
        )
        self.assertEqual(
            nan_cnt_fused,
            0,
            f"Fused result contains {nan_cnt_fused} NaN values",
        )

        # Assert numerical closeness
        np.testing.assert_allclose(
            golden_np,
            fused_np,
            rtol=0.01,
            atol=1,
            err_msg=f"Results don't match for shape [{height}, {width}]",
        )

    def test_width_4096_height_8192(self):
        """Test case: width=4096, height=8192"""
        self._test_single_case(height=8192, width=4096)

    def test_width_4096_height_16384(self):
        """Test case: width=4096, height=16384"""
        self._test_single_case(height=16384, width=4096)

    def test_width_4096_height_32768(self):
        """Test case: width=4096, height=32768"""
        self._test_single_case(height=32768, width=4096)

    def test_width_7168_height_8192(self):
        """Test case: width=7168, height=8192"""
        self._test_single_case(height=8192, width=7168)

    def test_width_7168_height_16384(self):
        """Test case: width=7168, height=16384"""
        self._test_single_case(height=16384, width=7168)

    def test_width_7168_height_32768(self):
        """Test case: width=7168, height=32768"""
        self._test_single_case(height=32768, width=7168)

    def test_all_combinations(self):
        """Test all width and height combinations"""
        widths = [4096, 7168]
        heights = [8192, 16384, 32768]

        for width in widths:
            for height in heights:
                with self.subTest(width=width, height=height):
                    self._test_single_case(height, width)

    def test_edge_cases(self):
        """Test edge cases with smaller dimensions"""
        # Test with smaller dimensions
        small_cases = [(128, 256), (256, 512), (512, 1024)]

        for height, width in small_cases:
            with self.subTest(height=height, width=width):
                self._test_single_case(height, width)

    def test_input_validation(self):
        """Test input validation"""
        # Test with invalid inputs
        with self.assertRaises((ValueError, TypeError)):
            # Test with mismatched dimensions
            x = paddle.randn([100, 200]).astype("bfloat16")
            prob = paddle.randn([150, 1]).astype("float32")  # Wrong height
            paddle.incubate.nn.functional.fused_weighted_swiglu_act_quant(
                x, prob, using_pow2_scaling=False
            )


if __name__ == '__main__':
    # Set up test environment
    paddle.device.set_device('gpu')

    # Run tests
    unittest.main(verbosity=2)
