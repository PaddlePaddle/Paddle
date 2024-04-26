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

import random
import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle
import paddle.nn.functional as F
from paddle.base import core


def get_triangle_upper_mask(x):
    mask = paddle.full_like(x, -1e4)
    mask.stop_gradient = True
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask.astype(np.float32)


def attention_naive(q, k, v, bias, is_causal=True):
    origin_dtype = q.dtype
    assert k.dtype == origin_dtype
    assert v.dtype == origin_dtype
    if q.dtype != paddle.float32:
        q = paddle.cast(q, "float32")
        k = paddle.cast(k, "float32")
        v = paddle.cast(v, "float32")
    # real calculation
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    if bias is not None:
        s = s + bias
    if is_causal:
        mask = get_triangle_upper_mask(s)
        s = s + mask
    softmax_lse = paddle.logsumexp(s, axis=3)
    p = F.softmax(s)
    o = paddle.matmul(p, vt)
    o = paddle.cast(o, np.float32)
    o = paddle.transpose(o, [0, 2, 1, 3])
    return o, softmax_lse


def is_flashattn_supported():
    xpu_version = core.get_xpu_device_version(0)
    if xpu_version != core.XPUVersion.XPU3:
        return False
    xhpc_version = paddle.version.xpu_xhpc()
    if xhpc_version == 'False':
        return False
    return True


class XPUTestFlashAttentionOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "flash_attn"
        self.use_dynamic_create_class = False

    @unittest.skipIf(
        not is_flashattn_supported(), "only available on XPU3 with XHPC"
    )
    class TestFlashAttentionOp(XPUOpTest):
        def setUp(self):
            self.op_type = "flash_attn"
            self.tmp_seed = random.randint(1, 65536)
            paddle.seed(self.tmp_seed)
            self.init_dtype()
            self.set_attrs()
            self.set_shape()
            self.init_data()
            self.tolerance = 5e-4

        def init_dtype(self):
            self.dtype = self.in_type

        def set_shape(self):
            # [b, l, h, d]
            self.q_shape = [1, 128, 2, 32]
            self.k_shape = [1, 128, 2, 32]
            self.v_shape = [1, 128, 2, 32]
            # [b, h, l, l]
            self.bias_shape = [1, 2, 128, 128]

        def set_attrs(self):
            self.is_causal = True
            self.with_bias = False

        def init_data(self):
            q = np.random.random(self.q_shape)
            k = np.random.random(self.k_shape)
            v = np.random.random(self.v_shape)
            q_ = paddle.to_tensor(q, stop_gradient=False)
            k_ = paddle.to_tensor(k, stop_gradient=False)
            v_ = paddle.to_tensor(v, stop_gradient=False)
            # fixed the seed & offset to pass the check of seed_offset
            fixed_seed_offset = paddle.to_tensor(
                np.array([self.tmp_seed, 0]).astype(np.int32)
            )
            self.inputs = {
                "q": convert_float_to_uint16(q)
                if self.dtype == np.uint16
                else q.astype(self.dtype),
                "k": convert_float_to_uint16(k)
                if self.dtype == np.uint16
                else k.astype(self.dtype),
                "v": convert_float_to_uint16(v)
                if self.dtype == np.uint16
                else v.astype(self.dtype),
                "fixed_seed_offset": fixed_seed_offset,
            }
            bias_ = None
            if self.with_bias:
                bias = np.random.random(self.bias_shape).astype(np.float32)
                self.inputs["attn_mask"] = bias
                bias_ = paddle.to_tensor(bias, stop_gradient=True)

            out, softmax_lse = attention_naive(
                q_, k_, v_, bias=bias_, is_causal=self.is_causal
            )
            out.backward()
            self.dq = q_.grad.numpy()
            self.dk = k_.grad.numpy()
            self.dv = v_.grad.numpy()
            self.dout = paddle.ones_like(out, dtype=self.dtype)
            self.attrs = {
                'dropout': 0.0,
                'causal': self.is_causal,
                'return_softmax': False,
                'rng_name': '',
            }
            softmax_lse = softmax_lse.numpy()
            self.outputs = {
                "out": convert_float_to_uint16(out)
                if self.dtype == np.uint16
                else out.astype(self.dtype),
                "softmax": np.array([]),  # not used
                "softmax_lse": softmax_lse,
                "seed_offset": fixed_seed_offset,
            }

        def test_check_output(self):
            self.check_output_with_place(
                paddle.XPUPlace(0), atol=self.tolerance, rtol=self.tolerance
            )

        def test_check_grad(self):
            self.check_grad(
                ['q', 'k', 'v'],
                'out',
                user_defined_grads=[self.dq, self.dk, self.dv],
                user_defined_grad_outputs=self.dout,
                numeric_grad_delta=self.tolerance,
                max_relative_error=self.tolerance,
            )

    # class TestFlashAttentionOp2_with_bias(TestFlashAttentionOp):
    #     def set_attrs(self):
    #         self.is_causal = True
    #         self.with_bias = True

    # class TestFlashAttentionOp3_uncausal_with_bias(TestFlashAttentionOp): //WIP

    #     def set_attrs(self):
    #         self.is_causal = False
    #         self.with_bias = True


support_types = get_xpu_op_support_types("flash_attn")
for stype in support_types:
    create_test_class(globals(), XPUTestFlashAttentionOp, stype)

if __name__ == '__main__':
    paddle.disable_static()
    unittest.main()
