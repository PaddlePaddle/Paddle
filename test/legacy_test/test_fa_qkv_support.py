# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import os
import re
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.nn.functional.flash_attention import (
    scaled_dot_product_attention,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


def attention_naive(q, k, v, causal=False):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    p = (
        paddle.incubate.softmax_mask_fuse_upper_triangle(s)
        if causal
        else F.softmax(s)
    )
    o = paddle.matmul(p, vt)
    return paddle.transpose(o, [0, 2, 1, 3])


def attention_naive_with_mask(q, k, v, attn_bias):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    p = F.softmax(s + attn_bias)
    o = paddle.matmul(p, vt)
    return paddle.transpose(o, [0, 2, 1, 3])


is_sm8x = (
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] == 8
    and paddle.device.cuda.get_device_capability()[1] >= 0
)

is_sm90 = (
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] == 9
    and paddle.device.cuda.get_device_capability()[1] == 0
)

is_sm_supported = is_sm8x or is_sm90


def is_flashattn_supported():
    if (
        not core.is_compiled_with_cuda()
        or get_cuda_version() < 11040
        or not is_sm_supported
    ):
        return False
    return True


@unittest.skipIf(
    not is_flashattn_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 7.5 or 8.x",
)
class TestFlashAttentionNoKVGrad(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 128, 8, 16)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = True
        self.return_softmax = False
        self.enable_math = False
        self.enable_flash = True
        self.enable_mem_efficient = False

    def _init_tensor_from_numpy(self, array, stop_gradient):
        t = paddle.to_tensor(
            array,
            place=self.place,
            dtype=self.dtype,
            stop_gradient=stop_gradient,
        )
        return t

    def test_all(self):
        logging.info(
            f"Test case shape {self.shape} dtype {self.dtype} causal {self.causal}"
        )
        # test dynamic
        paddle.disable_static()

        query = np.random.random(self.shape)
        key = np.random.random(self.shape)
        value = np.random.random(self.shape)

        q = self._init_tensor_from_numpy(query, stop_gradient=True)
        k = self._init_tensor_from_numpy(key, stop_gradient=False)
        v = self._init_tensor_from_numpy(value, stop_gradient=False)

        q_ = self._init_tensor_from_numpy(query, stop_gradient=True)
        k_ = self._init_tensor_from_numpy(key, stop_gradient=False)
        v_ = self._init_tensor_from_numpy(value, stop_gradient=False)

        with paddle.nn.functional.sdp_kernel(
            enable_math=self.enable_math,
            enable_flash=self.enable_flash,
            enable_mem_efficient=self.enable_mem_efficient,
        ):
            out = scaled_dot_product_attention(
                q, k, v, None, self.dropout, self.causal
            )

        out_ = attention_naive(q_, k_, v_, self.causal)
        np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)

        out.backward()
        out_.backward()

        self.assertEqual(k.grad.shape, k.shape)
        self.assertEqual(k_.grad.shape, k.shape)
        np.testing.assert_allclose(
            k.grad.numpy(), k_.grad.numpy(), rtol=5e-03, atol=1e-03
        )

        self.assertEqual(v.grad.shape, v.shape)
        self.assertEqual(v_.grad.shape, v.shape)
        np.testing.assert_allclose(
            k.grad.numpy(), k_.grad.numpy(), rtol=5e-03, atol=1e-03
        )


if __name__ == '__main__':
    unittest.main()
