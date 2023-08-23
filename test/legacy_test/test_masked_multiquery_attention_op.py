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
from paddle.framework import core
from paddle.incubate.nn.functional import masked_multiquery_attention


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestMMHAOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.bsz = 2
        self.cache_bsz = 2
        self.num_head = 32
        self.num_head_kv = 2
        self.dim_head = 128
        self.beam_size = 1
        self.max_seq_len = 33
        self.seq_len = 32
        self.q = np.random.uniform(
            -0.05, 0.05, [self.bsz, self.num_head, self.dim_head]
        )
        self.k = np.random.uniform(
            -0.05, 0.05, [self.bsz, self.num_head_kv, self.dim_head]
        )
        self.v = np.random.uniform(
            -0.05, 0.05, [self.bsz, self.num_head_kv, self.dim_head]
        )
        self.src_mask = np.zeros([self.bsz, 1, 1, self.seq_len + 1])
        self.cum_offsets = None
        self.seq_lens = None
        self.rotary_tensor = None
        self.beam_cache_offset = None
        self.cache_kv_out = np.random.uniform(
            -0.05,
            0.05,
            [
                2,
                self.cache_bsz,
                self.num_head_kv,
                self.seq_len,
                self.dim_head,
            ],
        )
        numpy_ones = np.zeros(
            [2, self.cache_bsz, self.num_head_kv, 1, self.dim_head]
        )
        self.cache_kv_mmha_out = np.concatenate(
            (self.cache_kv_out, numpy_ones), axis=3
        )
        self.out_shift = None
        self.out_smooth = None

        self.rotary_emb_dims = 0
        self.use_neox_rotary_style = False

        self.out_scale = 10
        self.quant_round_type = 1
        self.quant_max_bound = 126
        self.quant_min_bound = -126
        self.place = paddle.CUDAPlace(0)

    def quant_helper(
        self, x, quant_scale, quant_round_type, quant_max_bound, quant_min_bound
    ):
        quant_value = quant_max_bound * quant_scale * x
        if quant_round_type == 0:
            quant_value = paddle.to_tensor(np.rint(quant_value.numpy()))
        else:
            quant_value = paddle.round(quant_value)
        return paddle.cast(
            paddle.clip(quant_value, quant_min_bound, quant_max_bound),
            paddle.int8,
        )

    def mmha_naive(
        self,
        q,
        k,
        v,
        cache_kv_out,
        src_mask,
        out_scale,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
    ):
        k_ = paddle.to_tensor(np.expand_dims(k, axis=2))
        v_ = paddle.to_tensor(np.expand_dims(v, axis=2))
        q_ = paddle.to_tensor(np.expand_dims(q, axis=2))
        cache_k, cache_v = paddle.split(cache_kv_out, 2, axis=0)
        k_ = paddle.concat([cache_k.squeeze(0), k_], axis=2)
        v_ = paddle.concat([cache_v.squeeze(0), v_], axis=2)

        for i in range(self.num_head):
            tmp = paddle.matmul(
                x=q_[:, i, :, :] * (self.dim_head**-0.5),
                y=k_[:, i // (self.num_head // self.num_head_kv), :, :],
                transpose_y=True,
            )
            if i == 0:
                product = tmp
            else:
                product = paddle.concat([product, tmp], axis=1)
        product = paddle.to_tensor(np.expand_dims(product, axis=2))
        product = product + src_mask
        product = paddle.nn.functional.softmax(product)
        for i in range(self.num_head):
            tmp = paddle.matmul(
                product[:, i, :, :],
                v_[:, i // (self.num_head // self.num_head_kv), :, :],
            )
            if i == 0:
                out = tmp
            else:
                out = paddle.concat([out, tmp], axis=1)
        normalized_out = self.quant_helper(
            out,
            out_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
        return out, normalized_out

    def check_main(
        self,
        q,
        k,
        v,
        cache_kv_out,
        cache_kv_mmha_out,
        src_mask,
        out_scale,
        dtype,
        dynamic_graph,
    ):
        paddle.disable_static()
        q_tensor = paddle.to_tensor(q).cast(dtype)
        k_tensor = paddle.to_tensor(k).cast(dtype)
        v_tensor = paddle.to_tensor(v).cast(dtype)
        src_mask_tensor = paddle.to_tensor(src_mask).cast(dtype)
        cache_kv_out_tensor = paddle.to_tensor(cache_kv_out).cast(dtype)
        cache_kv_mmha_out_tensor = paddle.to_tensor(cache_kv_mmha_out).cast(
            dtype
        )
        paddle_naive_mmha_out = 0
        paddle_naive_mmha_out = self.mmha_naive(
            q_tensor,
            k_tensor,
            v_tensor,
            cache_kv_out_tensor,
            src_mask_tensor,
            out_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )
        if dynamic_graph:
            paddle_mmha_out = masked_multiquery_attention(
                q_tensor,
                k_tensor,
                v_tensor,
                cache_kv_mmha_out_tensor,
                src_mask_tensor,
                None,
                None,
                None,
                None,
                None,
                None,
                self.seq_len,
                self.rotary_emb_dims,
                self.use_neox_rotary_style,
                out_scale,
                self.quant_round_type,
                self.quant_max_bound,
                self.quant_min_bound,
            )
        else:
            paddle.enable_static()
            with paddle.static.program_guard(paddle.static.Program()):
                q_static = paddle.static.data(
                    name="q_static",
                    shape=[self.bsz, self.num_head, self.dim_head],
                    dtype=dtype,
                )
                k_static = paddle.static.data(
                    name="k_static",
                    shape=[self.bsz, self.num_head_kv, self.dim_head],
                    dtype=dtype,
                )
                v_static = paddle.static.data(
                    name="v_static",
                    shape=[self.bsz, self.num_head_kv, self.dim_head],
                    dtype=dtype,
                )
                src_mask_static = paddle.static.data(
                    name="src_mask_static",
                    shape=[self.bsz, 1, 1, self.seq_len + 1],
                    dtype=dtype,
                )
                cache_kv_mmha_out_static = paddle.static.data(
                    name="cache_kv_mmha_out_static",
                    shape=[
                        2,
                        self.cache_bsz,
                        self.num_head_kv,
                        self.seq_len + 1,
                        self.dim_head,
                    ],
                    dtype=dtype,
                )

                outs = masked_multiquery_attention(
                    q_static,
                    k_static,
                    v_static,
                    cache_kv_mmha_out_static,
                    src_mask_static,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    32,
                    0,
                    False,
                    -1,
                    1,
                    127.0,
                    -127.0,
                )
                exe = paddle.static.Executor(self.place)
                paddle_mmha_out = exe.run(
                    feed={
                        "q_static": q.astype(dtype),
                        "k_static": k.astype(dtype),
                        "v_static": v.astype(dtype),
                        "cache_kv_mmha_out_static": cache_kv_mmha_out.astype(
                            dtype
                        ),
                        "src_mask_static": src_mask.astype(dtype),
                    },
                    fetch_list=[outs],
                )
        return paddle_naive_mmha_out, paddle_mmha_out

    def test_mmha_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle_naive_mmha, paddle_mmha_out = self.check_main(
            self.q,
            self.k,
            self.v,
            self.cache_kv_out,
            self.cache_kv_mmha_out,
            self.src_mask,
            -1,
            'float16',
            True,
        )
        np.testing.assert_allclose(
            paddle_mmha_out[0].numpy(),
            paddle_naive_mmha[0].numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_mmha_outlinear_in_scale(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle_naive_mmha, paddle_mmha_out = self.check_main(
            self.q,
            self.k,
            self.v,
            self.cache_kv_out,
            self.cache_kv_mmha_out,
            self.src_mask,
            self.out_scale,
            'float16',
            True,
        )
        np.testing.assert_allclose(
            paddle_mmha_out[0].numpy(),
            paddle_naive_mmha[1].numpy(),
            rtol=1,
            atol=1,
        )

    def test_static_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle_naive_mmha_out, paddle_mmha_out = self.check_main(
            self.q,
            self.k,
            self.v,
            self.cache_kv_out,
            self.cache_kv_mmha_out,
            self.src_mask,
            self.out_scale,
            'float16',
            False,
        )

        np.testing.assert_allclose(
            paddle_mmha_out[0],
            paddle_naive_mmha_out[0],
            rtol=1e-3,
            atol=1e-3,
        )


if __name__ == '__main__':
    unittest.main()
