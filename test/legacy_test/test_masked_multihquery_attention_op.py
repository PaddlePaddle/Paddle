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
from paddle.fluid import core


def mmha_wrapper(
    x,
    kv,
    bias,
    src_mask,
    sequence_lengths,
    rotary_tensor,
    beam_cache_offset,
    cache_kv_out,
    qkv_out_scale,
    out_linear_shift,
    out_linear_smooth,
    beam_size,
    rotary_emb_dims,
    split_kv,
    head_kv,
    mask_broadcast_num_heads,
    compute_bias,
    use_neox_rotary_style,
    out_linear_in_scale,
    quant_round_type,
    quant_max_bound,
    quant_min_bound,
    mqa,
):
    return paddle._C_ops.masked_multiquery_attention_(
        x,
        kv,
        bias,
        src_mask,
        sequence_lengths,
        rotary_tensor,
        beam_cache_offset,
        cache_kv_out,
        qkv_out_scale,
        out_linear_shift,
        out_linear_smooth,
        beam_size,
        rotary_emb_dims,
        split_kv,
        head_kv,
        mask_broadcast_num_heads,
        compute_bias,
        use_neox_rotary_style,
        out_linear_in_scale,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
    )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestMMHAOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.bsz = 2
        self.cache_bsz = 2
        self.num_head = 6
        self.dim_head = 32
        self.beam_size = 1
        self.max_seq_len = 6
        self.sequence_length = 5
        self.mqa = True
        self.num_head_kv = 2
        self.q = np.random.uniform(
            -0.05, 0.05, [self.bsz, self.num_head, self.dim_head]
        )
        self.k = np.random.uniform(
            -0.05, 0.05, [self.bsz, self.num_head_kv, self.dim_head]
        )
        self.v = np.random.uniform(
            -0.05, 0.05, [self.bsz, self.num_head_kv, self.dim_head]
        )
        self.x =  np.concatenate((self.q, self.k, self.v),axis=1)

        self.bias = np.random.uniform(
            -0.05, 0.05, [ self.num_head+2*self.num_head_kv, self.dim_head]
        )
        self.src_mask = np.zeros([self.bsz, 1, 1, self.sequence_length + 1])
        self.sequence_lengths = None
        self.rotary_tensor = None
        self.beam_cache_offset = None
        self.cache_kv_out = np.random.uniform(
            -0.05,
            0.05,
            [
                2,
                self.cache_bsz,
                self.num_head_kv,
                self.sequence_length,
                self.dim_head,
            ],
        )
        numpy_ones = np.zeros(
            [2, self.cache_bsz, self.num_head_kv, 1, self.dim_head]
        )
        self.cache_kv_mmha_out = np.concatenate(
            (self.cache_kv_out, numpy_ones), axis=3
        )

        self.qkv_out_scale = np.random.uniform(
            -0.5, 1, [3, self.num_head, self.dim_head]
        )
        self.out_linear_shift = None
        self.out_linear_smooth = None

        self.beam_size = 1
        self.rotary_emb_dims = 0
        self.mask_broadcast_num_heads = True
        self.compute_bias = True
        self.use_neox_rotary_style = False

        # self.out_linear_in_scale = 1.5
        self.out_linear_in_scale = -1
        self.quant_round_type = 1
        self.quant_max_bound = 126
        self.quant_min_bound = -126
        

    def quant_helper(
        self, x, quant_scale, quant_round_type, quant_max_bound, quant_min_bound
    ):
        x = paddle.to_tensor(x)
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
        x,
        q,
        k,
        v,
        bias,
        src_mask,
        cache_kv_out,
        qkv_out_scale,
        beam_size,
        mask_broadcast_num_heads,
        compute_bias,
        out_linear_in_scale,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
    ):
        if qkv_out_scale is not None:
            exit()
        else:
            q = q + bias[0:self.num_head,:]
            k = k + bias[self.num_head:self.num_head+self.num_head_kv,:]
            v = v + bias[self.num_head+self.num_head_kv:self.num_head+self.num_head_kv*2,:]

        
        
        cache_k, cache_v = paddle.split(cache_kv_out, 2, axis=0)
        k=paddle.to_tensor(np.expand_dims(k, axis=2))
        v=paddle.to_tensor(np.expand_dims(v, axis=2))
        q=paddle.to_tensor(np.expand_dims(q, axis=2))
        k = paddle.concat([cache_k.squeeze(0), k], axis=2)
        v = paddle.concat([cache_v.squeeze(0), v], axis=2)
        for i in range(self.num_head):
            tmp = paddle.matmul(
            x=q[:,i,:,:] * (self.dim_head ** -0.5), y=k[:,i%self.num_head_kv,:,:], transpose_y=True
            )
            if(i == 0):
                product = tmp
            else:
                product = np.concatenate((product, tmp),axis=1)
        product=np.expand_dims(product, axis=2)
        product = product + src_mask
        
        product = paddle.nn.functional.softmax(product)
        for i in range(self.num_head):
            tmp = paddle.matmul(
            product[:,i,:,:],v[:,i%self.num_head_kv,:,:] 
            )   
            if(i == 0):
                out = tmp
            else:
                out = np.concatenate((out, tmp),axis=1)
        out=np.expand_dims(out, axis=1)
        normalized_out = self.quant_helper(
            out,
            out_linear_in_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
        # print(out)
        return out, normalized_out

    def check_main(
        self,
        x,
        bias,
        src_mask,
        cache_kv_out,
        split_kv,
        cache_kv_mmha_out,
        qkv_out_scale,
        out_linear_in_scale,
        dtype,
    ):
        if(split_kv):
            kv =  np.concatenate((self.k, self.v),axis=1)
            paddle.disable_static()
            self.q = paddle.to_tensor(self.q).cast(dtype)
            self.k = paddle.to_tensor(self.k).cast(dtype)
            self.v = paddle.to_tensor(self.v).cast(dtype)
            kv = paddle.to_tensor(kv).cast(dtype)
            x = self.q
        else:
            x = paddle.to_tensor(x).cast(dtype)
            kv = None
            
        bias = paddle.to_tensor(bias).cast(dtype)
        src_mask = paddle.to_tensor(src_mask).cast(dtype)
        cache_kv_out = paddle.to_tensor(cache_kv_out).cast(dtype)
        cache_kv_mmha_out = paddle.to_tensor(cache_kv_mmha_out).cast(dtype)
        paddle_naive_mmha_out = 0
        paddle_naive_mmha_out = self.mmha_naive(
            x,
            self.q,
            self.k,
            self.v,
            bias,
            src_mask,
            cache_kv_out,
            None,
            self.beam_size,
            self.mask_broadcast_num_heads,
            self.compute_bias,
            out_linear_in_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )
        paddle_mmha_out = 0
        paddle_mmha_out = mmha_wrapper(
            x,
            kv,
            bias,
            src_mask,
            # None,
            None,
            None,
            None,
            cache_kv_mmha_out,
            # qkv_out_scale,
            None,
            None,
            None,
            self.beam_size,
            self.rotary_emb_dims,
            split_kv,
            self.num_head_kv,
            self.mask_broadcast_num_heads,
            self.compute_bias,
            self.use_neox_rotary_style,
            self.out_linear_in_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
            self.mqa,
        )
        paddle.enable_static()
        
        return paddle_naive_mmha_out, paddle_mmha_out

    def test_mmha_split_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle_naive_rmsnorm, paddle_mmha_out = self.check_main(
            self.x,
            self.bias,
            self.src_mask,
            self.cache_kv_out,
            True,
            self.cache_kv_mmha_out,
            None,
            -1,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_mmha_out[0],
            paddle_naive_rmsnorm[0],
            rtol=1e-3,
            atol=1e-3,
        )
        
    def test_mmha_fused_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle_naive_rmsnorm, paddle_mmha_out = self.check_main(
            self.x,
            self.bias,
            self.src_mask,
            self.cache_kv_out,
            False,
            self.cache_kv_mmha_out,
            None,
            -1,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_mmha_out[0],
            paddle_naive_rmsnorm[0],
            rtol=1e-3,
            atol=1e-3,
        )    

   


if __name__ == '__main__':
    unittest.main()