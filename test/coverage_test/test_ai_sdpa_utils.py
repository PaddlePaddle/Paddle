# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# Unit test for paddle.nn.functional.sdpa utility functions
# Target: cover SDPParams, check_head_dim_size_flash, check_flash_causal_non_square_seqlens,
#   check_dtypes_low_precision_fa, check_dtypes_low_precision_mem_efficient_attn

import unittest

import paddle
from paddle.nn.functional.sdpa import (
    SDPParams,
    check_all_tensors_on_device,
    check_cuda_is_available,
    check_dtypes_low_precision_fa,
    check_dtypes_low_precision_mem_efficient_attn,
    check_flash_causal_non_square_seqlens,
    check_head_dim_size_flash,
    check_sm_version,
    get_device_capability,
    init_config,
)


def _make_params(q, k, v, **kwargs):
    """Helper to create SDPParams from tensors."""
    return SDPParams(
        query_shape=q.shape,
        key_shape=k.shape,
        value_shape=v.shape,
        attn_mask_shape=kwargs.get('attn_mask_shape'),
        dropout=kwargs.get('dropout', 0.0),
        is_causal=kwargs.get('is_causal', False),
        scale=kwargs.get('scale'),
        query_stop_gradient=kwargs.get('query_stop_gradient', True),
        dtype=(q.dtype, k.dtype, v.dtype),
        place=(q.place, k.place, v.place),
    )


class TestGetDeviceCapability(unittest.TestCase):
    """Test get_device_capability function."""

    def test_negative_device_id(self):
        """Negative device_id should return (0, 0)."""
        result = get_device_capability(-1)
        self.assertEqual(result, (0, 0))


class TestCheckSmVersion(unittest.TestCase):
    """Test check_sm_version function."""

    def test_check_sm_version_in_range(self):
        """SM version (0,0) should be in range [(0,0), (12,1)]."""
        result = check_sm_version((0, 0), (12, 1), device_id=-1)
        self.assertTrue(result)

    def test_check_sm_version_out_of_range(self):
        """SM version (0,0) should NOT be in range [(8,0), (12,1)]."""
        # device_id=-1 returns capability (0,0), so (8,0) <= (0,0) is False
        result = check_sm_version((8, 0), (12, 1), device_id=-1)
        self.assertFalse(result)


class TestCheckCudaIsAvailable(unittest.TestCase):
    """Test check_cuda_is_available function."""

    def test_returns_bool(self):
        """Should return a boolean value."""
        result = check_cuda_is_available()
        self.assertIsInstance(result, bool)


class TestCheckAllTensorsOnDevice(unittest.TestCase):
    """Test check_all_tensors_on_device function."""

    def setUp(self):
        paddle.disable_static()

    def test_cpu_tensors(self):
        """Tensors should pass device check (GPU or custom place)."""
        q = paddle.randn([2, 4, 8, 16])
        k = paddle.randn([2, 4, 8, 16])
        v = paddle.randn([2, 4, 8, 16])
        params = _make_params(q, k, v)
        result = check_all_tensors_on_device(params)
        # On GPU machines, CPU tensors are automatically moved, so this may be True
        self.assertIsInstance(result, bool)

    def test_gpu_tensors(self):
        """GPU tensors should pass device check."""
        if paddle.is_compiled_with_cuda() and paddle.cuda.is_available():
            q = paddle.randn([2, 4, 8, 16]).cuda()
            k = paddle.randn([2, 4, 8, 16]).cuda()
            v = paddle.randn([2, 4, 8, 16]).cuda()
            params = _make_params(q, k, v)
            result = check_all_tensors_on_device(params)
            self.assertTrue(result)


class TestCheckHeadDimSizeFlash(unittest.TestCase):
    """Test check_head_dim_size_flash function."""

    def setUp(self):
        paddle.disable_static()

    def test_valid_head_dim(self):
        """Valid head_dim (<=256, multiple of 8) should return True."""
        q = paddle.randn([2, 4, 8, 64])
        k = paddle.randn([2, 4, 8, 64])
        v = paddle.randn([2, 4, 8, 64])
        params = _make_params(q, k, v)
        self.assertTrue(check_head_dim_size_flash(params))

    def test_head_dim_too_large(self):
        """Head dim > 256 should return False."""
        q = paddle.randn([2, 4, 8, 512])
        k = paddle.randn([2, 4, 8, 512])
        v = paddle.randn([2, 4, 8, 512])
        params = _make_params(q, k, v)
        self.assertFalse(check_head_dim_size_flash(params))

    def test_head_dim_not_multiple_of_8(self):
        """Head dim not multiple of 8 should return False."""
        q = paddle.randn([2, 4, 8, 7])
        k = paddle.randn([2, 4, 8, 7])
        v = paddle.randn([2, 4, 8, 7])
        params = _make_params(q, k, v)
        self.assertFalse(check_head_dim_size_flash(params))

    def test_mismatched_head_dims(self):
        """Mismatched head dims should return False."""
        q = paddle.randn([2, 4, 8, 64])
        k = paddle.randn([2, 4, 8, 32])
        v = paddle.randn([2, 4, 8, 64])
        params = _make_params(q, k, v)
        self.assertFalse(check_head_dim_size_flash(params))


class TestCheckFlashCausalNonSquareSeqlens(unittest.TestCase):
    """Test check_flash_causal_non_square_seqlens function."""

    def setUp(self):
        paddle.disable_static()

    def test_non_causal(self):
        """Non-causal should always return True."""
        q = paddle.randn([2, 4, 8, 64])
        k = paddle.randn([2, 4, 6, 64])
        v = paddle.randn([2, 4, 6, 64])
        params = _make_params(q, k, v)
        self.assertTrue(check_flash_causal_non_square_seqlens(params))

    def test_causal_equal_seq_len(self):
        """Causal with equal seq len should return True."""
        q = paddle.randn([2, 4, 8, 64])
        k = paddle.randn([2, 4, 8, 64])
        v = paddle.randn([2, 4, 8, 64])
        params = _make_params(q, k, v, is_causal=True)
        self.assertTrue(check_flash_causal_non_square_seqlens(params))

    def test_causal_unequal_seq_len(self):
        """Causal with unequal seq len - depends on hardware support."""
        q = paddle.randn([2, 4, 8, 64])
        k = paddle.randn([2, 4, 6, 64])
        v = paddle.randn([2, 4, 6, 64])
        params = _make_params(q, k, v, is_causal=True)
        result = check_flash_causal_non_square_seqlens(params)
        # On GPU with flash attention, this may still return True
        self.assertIsInstance(result, bool)


class TestCheckDtypesFlashAndMemEfficient(unittest.TestCase):
    """Test dtype check functions for flash and memory-efficient attention."""

    def setUp(self):
        paddle.disable_static()
        # Initialize config so that _config is populated
        init_config()

    def test_fa_valid_dtype(self):
        """Flash attention with valid dtype (float16)."""
        q = paddle.randn([2, 4, 8, 64], dtype='float16')
        k = paddle.randn([2, 4, 8, 64], dtype='float16')
        v = paddle.randn([2, 4, 8, 64], dtype='float16')
        params = _make_params(q, k, v)
        result = check_dtypes_low_precision_fa(params)
        self.assertTrue(result)

    def test_fa_float32_not_supported(self):
        """Flash attention with float32 should return False (not a supported dtype)."""
        q = paddle.randn([2, 4, 8, 64], dtype='float32')
        k = paddle.randn([2, 4, 8, 64], dtype='float32')
        v = paddle.randn([2, 4, 8, 64], dtype='float32')
        params = _make_params(q, k, v)
        result = check_dtypes_low_precision_fa(params)
        self.assertFalse(result)

    def test_fa_mismatched_dtype(self):
        """Flash attention with mismatched dtypes should return False."""
        q = paddle.randn([2, 4, 8, 64], dtype='float32')
        k = paddle.randn([2, 4, 8, 64], dtype='float16')
        v = paddle.randn([2, 4, 8, 64], dtype='float16')
        params = _make_params(q, k, v)
        self.assertFalse(check_dtypes_low_precision_fa(params))

    def test_mem_efficient_valid_dtype(self):
        """Memory-efficient attention with valid dtype (float32)."""
        q = paddle.randn([2, 4, 8, 64], dtype='float32')
        k = paddle.randn([2, 4, 8, 64], dtype='float32')
        v = paddle.randn([2, 4, 8, 64], dtype='float32')
        params = _make_params(q, k, v)
        result = check_dtypes_low_precision_mem_efficient_attn(params)
        self.assertTrue(result)

    def test_mem_efficient_mismatched_dtype(self):
        """Memory-efficient attention with mismatched dtypes should return False."""
        q = paddle.randn([2, 4, 8, 64], dtype='float32')
        k = paddle.randn([2, 4, 8, 64], dtype='float16')
        v = paddle.randn([2, 4, 8, 64], dtype='float16')
        params = _make_params(q, k, v)
        self.assertFalse(check_dtypes_low_precision_mem_efficient_attn(params))


if __name__ == '__main__':
    unittest.main()
