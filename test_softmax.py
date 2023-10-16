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

import os
import re
import unittest
from einops import rearrange, repeat
import numpy as np
from paddle import _C_ops, in_dynamic_mode
import paddle
import paddle.nn.functional as F
from paddle import fluid
from paddle.fluid import core
from paddle.nn.functional.flash_attention import (
    flash_attention,
)

def _get_block_size(head_dim, is_dropout, is_causal):
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 256
    major, minor = paddle.device.cuda.get_device_capability()
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0
    if head_dim <= 32:
        return 128, 128
    if head_dim <= 64:
        return (128, 128) if not is_dropout else (128, 64)
    elif head_dim <= 96:
        return (64, 64) if (is_sm8x and is_causal) else (128, 64)
    elif head_dim <= 128:
        if is_sm8x:
            return (64, 64) if (not is_dropout and is_causal) else (128, 32)
        else:
            return 128, (64 if not is_dropout else 32)
    elif head_dim <= 160:
        if is_sm8x:
            return (128, 64) if not is_causal else (64, 64)
        else:
            return 128, 32
    elif head_dim <= 192:
        return (128, 64) if not is_dropout else (64, 64)
    elif head_dim <= 224:
        return (128, 64) if (is_sm80 or is_sm90) else (64, 64)
    elif head_dim <= 256:
        return (128, 64) if is_sm80 else (64, 64)

def convert_flash_attn_S_to_softmax(S, head_dim, is_dropout,
                                    causal=False):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
    """
    seqlen_q, seqlen_k = S.shape[-2:]
    warps_n = 4
    blocksize_m, blocksize_n = _get_block_size(head_dim, is_dropout, causal)
    nblocks_n = (seqlen_k + blocksize_n - 1) // blocksize_n
    nblocks_m = (seqlen_q + blocksize_m - 1) // blocksize_m
    mmas_n = (blocksize_n + 16 - 1) // 16
    S_flat = rearrange(S, 'b h (nblocks_m blocksize_m) (nblocks_n blocksize_n) -> b h nblocks_m nblocks_n (blocksize_m blocksize_n)',
                       blocksize_m=blocksize_m, blocksize_n=blocksize_n)
    S_converted = rearrange(S_flat, 'b h nblocks_m nblocks_n (mmas_n mmas_m warps_n eight four c2 c1 c0) -> b h (nblocks_m mmas_m warps_n c1 eight) (nblocks_n mmas_n c2 four c0)',
                            mmas_n=mmas_n, warps_n=warps_n, eight=8, c0=2, c1=2, c2=2, four=4)
    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    return S_converted

def attention_naive_with_mask(q, k, v, attn_bias, dropout):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    p = F.softmax(s + attn_bias)
    p = s + attn_bias
    o = paddle.matmul(p, vt)
    return paddle.transpose(o, [0, 2, 1, 3]), p



def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    training=True,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        attn_mask(Tensor,optional): A float mask of the same type as query,
                        key, value that is added to the attention score.
        dropout_p(float): The dropout ratio.
        is_causal(bool): Whether enable causal mode.
        training(bool): Whether it is in the training phase.
        name(str, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP()
            >>> import paddle
            >>> q = paddle.rand((1, 128, 2, 16), dtype=paddle.bfloat16)
            >>> output = paddle.nn.functional.scaled_dot_product_attention(q, q, q, None, 0.9, False)
            >>> print(output)
            >>> # doctest: -SKIP
    """
    if attn_mask is None:
        out, _ = flash_attention(query, key, value, dropout_p, is_causal)
    else:
        fixed_seed_offset = (None,)
        return_softmax = True
        rng_name = ""
        out, _ = _C_ops.flash_attn(
            query,
            key,
            value,
            fixed_seed_offset,
            attn_mask,
            dropout_p,
            is_causal,
            True,
            not training,
            rng_name,
        )
        return out, _
    return out
 
 
class TestFlashAttentionWithMaskAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (1, 128, 4, 16)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False

    def test_dot_scale_product(self):
        # test dynamic
        paddle.disable_static()

        query = np.ones(self.shape)
        key = np.zeros(self.shape)
        value = np.ones(self.shape)

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

        mask_shape = (self.shape[0], 1, self.shape[1], self.shape[1])
        mask = np.random.random(mask_shape)
        mask = np.random.random(mask_shape)
        m = paddle.to_tensor(
            mask, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        innum = mask_shape[0] * mask_shape[1] * mask_shape[2] * mask_shape[3]
        m = paddle.arange(start=0, end=innum, step=1, dtype = self.dtype) 
        m = paddle.reshape(m, mask_shape)
        self.dropout = 0.1
        out,softmax = scaled_dot_product_attention(
            q, k, v, m, self.dropout, False
        )
        tmp_data = paddle.zeros([1, 4, 128, 128], dtype="float16")
        check_m  = paddle.add(tmp_data, m)

        out_,softmax_ = attention_naive_with_mask(q_, k_, v_, m, self.dropout)

        softmax_ = convert_flash_attn_S_to_softmax(paddle.cast(softmax, 'float32').numpy(), self.shape[-1], True, False)
        print(softmax_)

        print(softmax)
        #np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)
        np.testing.assert_allclose(softmax_, check_m) #.numpy())

if __name__ == '__main__':
    unittest.main()
