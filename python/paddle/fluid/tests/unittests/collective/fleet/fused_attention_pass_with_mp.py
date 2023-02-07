#   Copyright (c) 2013 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle
import paddle.distributed.fleet as fleet
import paddle.fluid as fluid
import paddle.nn.functional as F

paddle.enable_static()


class MultiHeadAttentionWithMP(paddle.nn.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        add_residual=True,
        pre_ln=True,
        attn_dropout=True,
    ):
        super(MultiHeadAttentionWithMP, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.num_heads = num_heads

        self.add_residual = add_residual
        self.pre_ln = pre_ln
        self.attn_dropout = attn_dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.mp_hidden_size = self.head_dim // 2

        self.norm1 = paddle.nn.LayerNorm(embed_dim, epsilon=1e-5)
        self.norm2 = paddle.nn.LayerNorm(embed_dim, epsilon=1e-5)

        self.qkv_proj = paddle.nn.Linear(embed_dim, 3 * self.mp_hidden_size)
        self.out_proj = paddle.nn.Linear(self.mp_hidden_size, embed_dim)
        self.dropout = paddle.nn.Dropout(1e-10, mode="upscale_in_train")

    def forward(self, x, attn_mask=None):
        residual = x

        x = paddle.distributed.collective._c_identity(x)

        if self.pre_ln:
            # pre layer norm
            x = self.norm1(x)

        # compute qkv
        qkv = self.qkv_proj(x)
        qkv = paddle.reshape(qkv, [0, 0, 3 * self.num_heads, self.head_dim])
        qkv = paddle.transpose(qkv, [0, 2, 1, 3])
        q, k, v = paddle.split(qkv, num_or_sections=3, axis=1)

        # compute core attention
        q = paddle.scale(q, scale=self.head_dim**-0.5)
        product = paddle.matmul(x=q, y=k, transpose_y=True)
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
        out = paddle.distributed.collective._mp_allreduce(
            out, use_calc_stream=True, use_model_parallel=True
        )
        out = self.dropout(out)
        if self.add_residual:
            out = residual + out

        if not self.pre_ln:
            # post layer norm
            out = self.norm2(out)

        return out


class TestFusedAttentionPassWithMP(unittest.TestCase):
    def setUp(self):
        fleet.init()
        self.endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS").split(',')
        self.current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
        self.nranks = len(self.endpoints)
        self.rank = self.endpoints.index(self.current_endpoint)
        self.gpu_id = int(os.getenv("FLAGS_selected_gpus"))
        self.place = fluid.CUDAPlace(self.gpu_id)
        self.exe = fluid.Executor(self.place)
        self.endpoints.remove(self.current_endpoint)
        self.other_endpoints = self.endpoints
        self.add_residual = True
        self.pre_ln = True
        self.attn_dropout = True
        self.add_mask = True
        self.x_data = None
        self.mask_data = None

    def get_rst(self, use_pass=False):
        batch_size = 2
        seq_len = 1024
        hidden_size = 768
        num_heads = 12

        np.random.seed(1234)
        if self.x_data is None:
            self.x_data = np.random.rand(batch_size, seq_len, seq_len).astype(
                'float32'
            )
            self.mask_data = np.random.rand(
                batch_size, num_heads, seq_len, seq_len
            ).astype('float32')

        main_prog = paddle.static.Program()
        main_prog.random_seed = 1234
        startup_prog = paddle.static.Program()
        startup_prog.random_seed = 1234

        with paddle.static.program_guard(main_prog, startup_prog):
            data = paddle.static.data(
                name="x",
                shape=[-1, seq_len, seq_len],
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

            data_linear = paddle.nn.Linear(seq_len, hidden_size)
            multi_head_attn = MultiHeadAttentionWithMP(
                hidden_size,
                num_heads,
                add_residual=self.add_residual,
                pre_ln=self.pre_ln,
                attn_dropout=self.attn_dropout,
            )

            attn_input = data_linear(data)
            out = multi_head_attn(attn_input, attn_mask)
            loss = paddle.mean(out)

            sgd_optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(loss)

        return 0

    def test_pass(self):
        fused_rst = self.get_rst(use_pass=True)
        non_fused_rst = self.get_rst()


if __name__ == "__main__":
    unittest.main()
