# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
import paddle.nn.functional as F
from paddle.distributed.passes import PassManager, new_pass

paddle.enable_static()


class MultiHeadAttention(paddle.nn.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        add_residual=True,
        pre_ln=True,
        post_ln=False,
        attn_dropout=True,
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.num_heads = num_heads

        self.add_residual = add_residual
        self.pre_ln = pre_ln
        self.post_ln = post_ln
        self.attn_dropout = attn_dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.norm1 = paddle.nn.LayerNorm(embed_dim, epsilon=1e-5)
        self.norm2 = paddle.nn.LayerNorm(embed_dim, epsilon=1e-5)

        self.qkv_proj = paddle.nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = paddle.nn.Linear(embed_dim, embed_dim)
        self.dropout = paddle.nn.Dropout(0.1, mode="upscale_in_train")

    def forward(self, x, attn_mask=None):
        residual = x

        if self.pre_ln:
            # pre layer norm
            x = self.norm1(x)

        # compute qkv
        qkv = self.qkv_proj(x)
        qkv = paddle.reshape(qkv, [0, 0, self.num_heads, 3 * self.head_dim])
        qkv = paddle.transpose(qkv, [0, 2, 1, 3])
        q, k, v = paddle.split(qkv, num_or_sections=3, axis=-1)

        # compute core attention
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        product = paddle.scale(product, scale=self.head_dim**-0.5)
        if attn_mask is not None:
            product = product + attn_mask
        weights = F.softmax(product)
        if self.attn_dropout:
            weights = F.dropout(
                weights, 0.1, training=self.training, mode="upscale_in_train"
            )
        out = paddle.matmul(weights, v)
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)
        out = self.dropout(out)
        if self.add_residual:
            out = residual + out

        if self.post_ln:
            # post layer norm
            out = self.norm2(out)

        return out


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFusedAttentionPass(unittest.TestCase):
    def setUp(self):
        self.add_residual = True
        self.pre_ln = True
        self.post_ln = True
        self.attn_dropout = True
        self.add_mask = True

    def test_pass(self):
        batch_size = 2
        seq_len = 1024
        hidden_size = 768
        num_heads = 12

        x_data = np.random.rand(batch_size, seq_len, hidden_size).astype(
            'float32'
        )
        mask_data = np.random.rand(
            batch_size, num_heads, seq_len, seq_len
        ).astype('float32')

        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, startup_prog):
            data = paddle.static.data(
                name="x",
                shape=[-1, seq_len, hidden_size],
                dtype='float32',
            )
            if self.add_mask:
                attn_mask = paddle.static.data(
                    name="attn_mask",
                    shape=[-1, num_heads, seq_len, seq_len],
                    dtype='float32',
                )
            else:
                attn_mask = None
            multi_head_attn = MultiHeadAttention(
                hidden_size,
                num_heads,
                add_residual=self.add_residual,
                pre_ln=self.pre_ln,
                post_ln=self.post_ln,
                attn_dropout=self.attn_dropout,
            )
            out = multi_head_attn(data, attn_mask)
            loss = paddle.mean(out)

            sgd_optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(loss)

        pass_manager = PassManager([new_pass("fused_attention")])
        pass_manager.apply([main_prog], [startup_prog])

        ops = main_prog.global_block().ops
        assert ops[0].type == 'reduce_mean'


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
