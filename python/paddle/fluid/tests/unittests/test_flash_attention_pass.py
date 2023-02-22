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
        head_dim,
        num_heads,
        add_residual=True,
        attn_dropout=True,
    ):
        super(MultiHeadAttention, self).__init__()
        self.head_dim = head_dim
        self.kdim = head_dim
        self.vdim = head_dim
        self.num_heads = num_heads

        self.add_residual = add_residual
        self.attn_dropout = attn_dropout

        self.head_dim = head_dim // num_heads

        self.norm1 = paddle.nn.LayerNorm(head_dim, epsilon=1e-5)
        self.norm2 = paddle.nn.LayerNorm(head_dim, epsilon=1e-5)

        self.qkv_proj = paddle.nn.Linear(head_dim, 3 * head_dim)
        self.out_proj = paddle.nn.Linear(head_dim, head_dim)
        self.dropout = paddle.nn.Dropout(1e-10, mode="upscale_in_train")

    def forward(self, x, causal=False):
        residual = x

        # compute qkv
        qkv = self.qkv_proj(x)
        qkv = paddle.reshape(qkv, [0, 0, 3 * self.num_heads, self.head_dim])
        qkv = paddle.transpose(qkv, [0, 2, 1, 3])
        q, k, v = paddle.split(qkv, num_or_sections=3, axis=1)

        # compute core attention
        k = paddle.scale(k, scale=self.head_dim**-0.5)
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        # product = paddle.scale(product, scale=self.head_dim**-0.5)

        if causal:
            weights = paddle.incubate.softmax_mask_fuse_upper_triangle(product)
        else:
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

        return out


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFlashAttentionPass(unittest.TestCase):
    def setUp(self):
        self.add_residual = True
        self.attn_dropout = True
        self.add_mask = True

        self.batch_size = 2
        self.seq_len = 1024
        self.hidden_size = 768
        self.num_heads = 12
        self.head_dim = 128

        self.place = paddle.CUDAPlace(0)
        self.dtype = 'float'

    def test_pass(self):
        out_ref = self.run_program(apply_pass=False)
        print(out_ref)
        # out_pass = self.run_program(apply_pass=True)
        # print(out_pass)

    def run_program(self, apply_pass=False):
        x_data = np.random.random([self.batch_size, self.seq_len, self.seq_len])

        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, startup_prog):
            data = paddle.static.data(
                name="x",
                shape=[-1, self.seq_len, self.seq_len],
                dtype=self.dtype,
            )

            data_linear = paddle.nn.Linear(self.seq_len, self.hidden_size)
            attn_input = data_linear(data)

            multi_head_attn = MultiHeadAttention(
                self.hidden_size,
                self.num_heads,
                add_residual=self.add_residual,
                attn_dropout=self.attn_dropout,
            )
            out = multi_head_attn(attn_input, causal=True)
            loss = paddle.mean(out)

            # sgd_optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.001)
            # sgd_optimizer.minimize(loss)

            if apply_pass:
                pass_manager = PassManager([new_pass("flash_attention")])
                pass_manager.apply([main_prog], [startup_prog])

                ops = main_prog.global_block().ops
                assert "flash_attn" in [op.type for op in ops]

            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            exe.run(startup_prog)

            ret_loss = exe.run(
                main_prog,
                feed={"x": x_data.astype('float')},
                fetch_list=[loss.name],
            )

            print(main_prog)
            return ret_loss


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
