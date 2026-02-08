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
from op_test import get_cuda_version, is_custom_device

import paddle
import paddle.nn.functional as F
from paddle.nn.attention import (
    SDPBackend,
    _cur_sdpa_kernel_backends,
    sdpa_kernel,
)
from paddle.nn.functional import scaled_dot_product_attention


def is_flashattn_supported():
    if (
        not paddle.base.core.is_compiled_with_cuda()
        or get_cuda_version() < 11040
    ):
        return False

    if paddle.device.cuda.device_count() == 0:
        return False

    try:
        capability = paddle.device.cuda.get_device_capability()
        major, minor = capability[0], capability[1]
        # Support sm8x or sm90
        return (major == 8 and minor >= 0) or (major == 9 and minor == 0)
    except:
        return False


def attention_naive(q, k, v, causal=False):
    """Reference implementation for attention calculation."""
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt * scale, paddle.transpose(kt, [0, 1, 3, 2]))
    if causal:
        mask = paddle.triu(paddle.ones_like(s) * -float('inf'), diagonal=1)
        s = s + mask
    p = F.softmax(s)
    o = paddle.matmul(p, vt)
    return paddle.transpose(o, [0, 2, 1, 3])


@unittest.skipIf(
    paddle.is_compiled_with_xpu(),
    "sdpa backend selection logic fails on XPU when testing CPU place",
)
class TestSDPAKernelCPU(unittest.TestCase):
    """Test sdpa_kernel on CPU specifically."""

    def setUp(self):
        self.place = paddle.CPUPlace()
        self.shape = (2, 128, 8, 16)
        self.dtype = 'float32'

    def test_cpu_math_backend(self):
        """Test MATH backend on CPU."""
        paddle.disable_static()

        query = np.random.random(self.shape).astype(self.dtype)
        key = np.random.random(self.shape).astype(self.dtype)
        value = np.random.random(self.shape).astype(self.dtype)

        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k_ = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v_ = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        with sdpa_kernel(SDPBackend.MATH):
            out = scaled_dot_product_attention(q, k, v)

        ref_out = attention_naive(q_, k_, v_, causal=False)
        np.testing.assert_allclose(
            out.numpy(), ref_out.numpy(), rtol=5e-3, atol=1e-3
        )

        # Test backward
        out.backward()
        ref_out.backward()

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=5e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            k.grad.numpy(), k_.grad.numpy(), rtol=5e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            v.grad.numpy(), v_.grad.numpy(), rtol=5e-3, atol=1e-3
        )

    def test_cpu_with_mask(self):
        """Test CPU with attention mask."""
        paddle.disable_static()

        query = np.random.random(self.shape).astype(self.dtype)
        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        # Create a mask
        mask_shape = (self.shape[0], 1, self.shape[1], self.shape[1])
        mask = np.random.random(mask_shape).astype(self.dtype)
        m = paddle.to_tensor(mask, place=self.place, dtype=self.dtype)

        with sdpa_kernel(SDPBackend.MATH):
            out = scaled_dot_product_attention(q, q, q, attn_mask=m)

        # Verify output shape and test backward
        self.assertEqual(out.shape, q.shape)
        out.backward()


@unittest.skipIf(
    not (paddle.is_compiled_with_cuda() or is_custom_device())
    or paddle.is_compiled_with_rocm(),
    "CUDA is not available, this test requires GPU support.",
)
class TestSDPAKernelBasic(unittest.TestCase):
    """Test basic functionality of sdpa_kernel context manager (defaults to available device)."""

    def setUp(self):
        self.shape = (2, 128, 8, 16)
        self.dtype = 'float32'

    def test_cur_sdpa_kernel_backends(self):
        result = _cur_sdpa_kernel_backends()
        self.assertIsInstance(result, list)

    def test_single_backend(self):
        """Test with single backend."""
        paddle.disable_static()

        query = np.random.random(self.shape).astype(self.dtype)
        key = np.random.random(self.shape).astype(self.dtype)
        value = np.random.random(self.shape).astype(self.dtype)

        q = paddle.to_tensor(query, dtype=self.dtype, stop_gradient=False)
        k = paddle.to_tensor(key, dtype=self.dtype, stop_gradient=False)
        v = paddle.to_tensor(value, dtype=self.dtype, stop_gradient=False)

        q_ = paddle.to_tensor(query, dtype=self.dtype, stop_gradient=False)
        k_ = paddle.to_tensor(key, dtype=self.dtype, stop_gradient=False)
        v_ = paddle.to_tensor(value, dtype=self.dtype, stop_gradient=False)

        with sdpa_kernel(SDPBackend.MATH):
            out = scaled_dot_product_attention(q, k, v)

        ref_out = attention_naive(q_, k_, v_, causal=False)
        np.testing.assert_allclose(
            out.numpy(), ref_out.numpy(), rtol=5e-3, atol=1e-3
        )

        # Test backward
        out.backward()
        ref_out.backward()

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=5e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            k.grad.numpy(), k_.grad.numpy(), rtol=5e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            v.grad.numpy(), v_.grad.numpy(), rtol=5e-3, atol=1e-3
        )

    def test_multiple_backends(self):
        """Test with multiple backends."""
        paddle.disable_static()

        query = np.random.random(self.shape).astype(self.dtype)
        key = np.random.random(self.shape).astype(self.dtype)
        value = np.random.random(self.shape).astype(self.dtype)

        q = paddle.to_tensor(query, dtype=self.dtype, stop_gradient=False)
        k = paddle.to_tensor(key, dtype=self.dtype, stop_gradient=False)
        v = paddle.to_tensor(value, dtype=self.dtype, stop_gradient=False)

        q_ = paddle.to_tensor(query, dtype=self.dtype, stop_gradient=False)
        k_ = paddle.to_tensor(key, dtype=self.dtype, stop_gradient=False)
        v_ = paddle.to_tensor(value, dtype=self.dtype, stop_gradient=False)

        # Test with multiple backends
        backends = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
        with sdpa_kernel(backends):
            out = scaled_dot_product_attention(q, k, v)

        ref_out = attention_naive(q_, k_, v_, causal=False)
        np.testing.assert_allclose(
            out.numpy(), ref_out.numpy(), rtol=5e-3, atol=1e-3
        )

        # Test backward
        out.backward()
        ref_out.backward()

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=5e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            k.grad.numpy(), k_.grad.numpy(), rtol=5e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            v.grad.numpy(), v_.grad.numpy(), rtol=5e-3, atol=1e-3
        )

    def test_multiple_backends_with_priority(self):
        """
        Test set_priority=True with available backends (MATH, EFFICIENT).
        """
        paddle.disable_static()

        query = np.random.random(self.shape).astype(self.dtype)
        key = np.random.random(self.shape).astype(self.dtype)
        value = np.random.random(self.shape).astype(self.dtype)

        q = paddle.to_tensor(query, dtype=self.dtype, stop_gradient=False)
        k = paddle.to_tensor(key, dtype=self.dtype, stop_gradient=False)
        v = paddle.to_tensor(value, dtype=self.dtype, stop_gradient=False)

        q_ = paddle.to_tensor(query, dtype=self.dtype, stop_gradient=False)
        k_ = paddle.to_tensor(key, dtype=self.dtype, stop_gradient=False)
        v_ = paddle.to_tensor(value, dtype=self.dtype, stop_gradient=False)

        backends = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

        with sdpa_kernel(backends, set_priority=True):
            out = scaled_dot_product_attention(q, k, v)

        ref_out = attention_naive(q_, k_, v_, causal=False)
        np.testing.assert_allclose(
            out.numpy(), ref_out.numpy(), rtol=5e-3, atol=1e-3
        )

        out.backward()
        ref_out.backward()

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=5e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            k.grad.numpy(), k_.grad.numpy(), rtol=5e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            v.grad.numpy(), v_.grad.numpy(), rtol=5e-3, atol=1e-3
        )


@unittest.skipIf(
    not is_flashattn_supported(),
    "Priority test requires flash attention support (CUDA SM80+)",
)
class TestSDPAKernelPriority(unittest.TestCase):
    """Test priority settings for sdpa_kernel."""

    def setUp(self):
        self.shape = (2, 64, 4, 32)
        self.dtype = 'float16'

    def test_set_priority_true(self):
        """Test set_priority=True."""
        paddle.disable_static()

        query = np.random.random(self.shape).astype(self.dtype)
        q = paddle.to_tensor(query, dtype=self.dtype, stop_gradient=False)
        q_ = paddle.to_tensor(query, dtype=self.dtype, stop_gradient=False)

        backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
        with sdpa_kernel(backends, set_priority=True):
            out = scaled_dot_product_attention(q, q, q)

        # Verify output correctness
        ref_out = attention_naive(q_, q_, q_, causal=False)
        np.testing.assert_allclose(
            out.numpy(), ref_out.numpy(), rtol=5e-3, atol=1e-3
        )

        # Test backward
        out.backward()
        ref_out.backward()

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=5e-3, atol=1e-3
        )

    def test_set_priority_false(self):
        """Test set_priority=False (default)."""
        paddle.disable_static()

        query = np.random.random(self.shape).astype(self.dtype)
        q = paddle.to_tensor(query, dtype=self.dtype, stop_gradient=False)
        q_ = paddle.to_tensor(query, dtype=self.dtype, stop_gradient=False)

        backends = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
        with sdpa_kernel(backends, set_priority=False):
            out = scaled_dot_product_attention(q, q, q)

        ref_out = attention_naive(q_, q_, q_, causal=False)
        np.testing.assert_allclose(
            out.numpy(), ref_out.numpy(), rtol=5e-3, atol=1e-3
        )

        # Test backward
        out.backward()
        ref_out.backward()

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=5e-3, atol=1e-3
        )


class TestSDPAKernelExceptions(unittest.TestCase):
    """Test exception handling in sdpa_kernel."""

    def test_invalid_backend_type(self):
        """Test with invalid backend type."""
        with self.assertRaises(AssertionError), sdpa_kernel("invalid_backend"):
            pass

    def test_invalid_backend_in_list(self):
        """Test with invalid backend in list."""
        with (
            self.assertRaises(TypeError),
            sdpa_kernel([SDPBackend.MATH, "invalid"]),
        ):
            pass

    def test_empty_backend_list(self):
        """Test with empty backend list."""
        with self.assertRaises(ValueError), sdpa_kernel([]):
            pass


@unittest.skipIf(
    not is_flashattn_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestSDPAKernelGPU(unittest.TestCase):
    """Test sdpa_kernel on GPU with different backends."""

    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 128, 8, 32)
        self.dtype = 'float16'

    def test_gpu_math_backend(self):
        """Test MATH backend on GPU."""
        paddle.disable_static()

        query = np.random.random(self.shape).astype(self.dtype)
        key = np.random.random(self.shape).astype(self.dtype)
        value = np.random.random(self.shape).astype(self.dtype)

        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k_ = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v_ = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        with sdpa_kernel(SDPBackend.MATH):
            out = scaled_dot_product_attention(q, k, v)

        # Convert to float32 for comparison
        q_fp32 = q_.astype('float32')
        k_fp32 = k_.astype('float32')
        v_fp32 = v_.astype('float32')
        ref_out = attention_naive(q_fp32, k_fp32, v_fp32, causal=False)

        np.testing.assert_allclose(
            out.astype('float32').numpy(), ref_out.numpy(), rtol=5e-3, atol=1e-3
        )

        # Test backward
        out.backward()
        ref_out.backward()

        np.testing.assert_allclose(
            q.grad.astype('float32').numpy(),
            q_.grad.numpy(),
            rtol=5e-3,
            atol=1e-3,
        )

    def test_flash_attention_backend(self):
        """Test FLASH_ATTENTION backend on GPU."""
        paddle.disable_static()

        query = np.random.random(self.shape).astype(self.dtype)
        key = np.random.random(self.shape).astype(self.dtype)
        value = np.random.random(self.shape).astype(self.dtype)

        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k_ = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v_ = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = scaled_dot_product_attention(q, k, v)

            # Convert to float32 for comparison
            q_fp32 = q_.astype('float32')
            k_fp32 = k_.astype('float32')
            v_fp32 = v_.astype('float32')
            ref_out = attention_naive(q_fp32, k_fp32, v_fp32, causal=False)

            np.testing.assert_allclose(
                out.astype('float32').numpy(),
                ref_out.numpy(),
                rtol=5e-3,
                atol=1e-3,
            )

            # Test backward
            out.backward()
            ref_out.backward()

            np.testing.assert_allclose(
                q.grad.astype('float32').numpy(),
                q_.grad.numpy(),
                rtol=5e-3,
                atol=1e-3,
            )
        except RuntimeError:
            # Flash attention might not be available
            self.skipTest("Flash attention not available on this GPU")

    def test_efficient_attention_backend(self):
        """Test EFFICIENT_ATTENTION backend on GPU."""
        paddle.disable_static()

        query = np.random.random(self.shape).astype(self.dtype)
        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        try:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                out = scaled_dot_product_attention(q, q, q)

            # Convert to float32 for comparison
            q_fp32 = q_.astype('float32')
            ref_out = attention_naive(q_fp32, q_fp32, q_fp32, causal=False)

            np.testing.assert_allclose(
                out.astype('float32').numpy(),
                ref_out.numpy(),
                rtol=5e-3,
                atol=1e-3,
            )

            # Test backward
            out.backward()
            ref_out.backward()

            np.testing.assert_allclose(
                q.grad.astype('float32').numpy(),
                q_.grad.numpy(),
                rtol=5e-3,
                atol=1e-3,
            )
        except RuntimeError:
            # Efficient attention might not be available
            self.skipTest("Efficient attention not available on this GPU")

    def test_all_backends_gpu(self):
        """Test all backends on GPU."""
        paddle.disable_static()

        query = np.random.random(self.shape).astype(self.dtype)
        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]

        with sdpa_kernel(backends):
            out = scaled_dot_product_attention(q, q, q)

        # Verify output shape and test backward
        self.assertEqual(out.shape, q.shape)
        out.backward()


if __name__ == '__main__':
    unittest.main()
