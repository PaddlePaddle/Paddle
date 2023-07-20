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
from paddle.fluid.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode


def mmha_wrapper(
    x,
    cache_kv,
    src_mask,
    cum_offsets,
    sequence_lengths,
    rotary_tensor,
    beam_cache_offset,
    qkv_out_scale,
    out_linear_shift,
    out_linear_smooth,
    seq_len,
    rotary_emb_dims,
    use_neox_rotary_style,
    out_linear_in_scale,
    quant_round_type,
    quant_max_bound,
    quant_min_bound,
):
    if in_dynamic_mode():
        return paddle._C_ops.masked_multihead_attention_(
            x,
            cache_kv,
            src_mask,
            cum_offsets,
            sequence_lengths,
            rotary_tensor,
            beam_cache_offset,
            qkv_out_scale,
            out_linear_shift,
            out_linear_smooth,
            seq_len,
            rotary_emb_dims,
            use_neox_rotary_style,
            out_linear_in_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
    helper = LayerHelper('masked_multihead_attention', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {}
    inputs['x'] = x
    inputs['cache_kv'] = cache_kv
    if src_mask:
        inputs['src_mask'] = src_mask
    if cum_offsets:
        inputs['cum_offsets'] = cum_offsets
    if sequence_lengths:
        inputs['sequence_lengths'] = sequence_lengths
    if rotary_tensor:
        inputs['rotary_tensor'] = rotary_tensor
    beam_cache_offset_flag = False
    if beam_cache_offset:
        inputs['beam_cache_offset'] = beam_cache_offset
        beam_cache_offset_flag = True
    else:
        beam_cache_offset = helper.create_variable_for_type_inference(
            dtype="int"
        )
    if qkv_out_scale:
        inputs['qkv_out_scale'] = qkv_out_scale
    if out_linear_shift:
        inputs['out_linear_shift'] = out_linear_shift
    if out_linear_smooth:
        inputs['out_linear_smooth'] = out_linear_smooth

    outputs = {
        'out': out,
        'cache_kv_out': cache_kv,
        'beam_cache_offset_out': beam_cache_offset,
    }
    helper.append_op(
        type='masked_multihead_attention',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'seq_len': seq_len,
            'rotary_emb_dims': rotary_emb_dims,
            'use_neox_rotary_style': use_neox_rotary_style,
            'out_linear_in_scale': out_linear_in_scale,
            'quant_round_type': quant_round_type,
            'quant_max_bound': quant_max_bound,
            'quant_min_bound': quant_min_bound,
        },
    )
    return (
        (out, cache_kv, beam_cache_offset)
        if beam_cache_offset_flag
        else (out, cache_kv)
    )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestMMHAOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.bsz = 2
        self.cache_bsz = 2
        self.num_head = 32
        self.dim_head = 128
        self.beam_size = 1
        self.max_seq_len = 33
        self.sequence_length = 32

        self.x = np.random.uniform(
            -0.05, 0.05, [self.bsz, 3, self.num_head, self.dim_head]
        )
        self.x_int = np.random.randint(
            2, 10, size=(self.bsz, 3, self.num_head, self.dim_head)
        ).astype("int")

        self.src_mask = np.zeros([self.bsz, 1, 1, self.sequence_length + 1])

        self.cum_offsets = None
        self.sequence_lengths = None
        self.rotary_tensor = None
        self.beam_cache_offset = None

        self.cache_kv_out = np.random.uniform(
            -0.05,
            0.05,
            [
                2,
                self.cache_bsz,
                self.num_head,
                self.sequence_length,
                self.dim_head,
            ],
        )
        numpy_ones = np.zeros(
            [2, self.cache_bsz, self.num_head, 1, self.dim_head]
        )
        self.cache_kv_mmha_out = np.concatenate(
            (self.cache_kv_out, numpy_ones), axis=3
        )

        self.qkv_out_scale = np.random.uniform(
            -0.05, 0.05, [3, self.num_head, self.dim_head]
        )
        self.out_linear_shift = None
        self.out_linear_smooth = None

        self.seq_len = 1
        self.rotary_emb_dims = 0
        self.use_neox_rotary_style = False

        self.out_linear_in_scale = 10
        self.quant_round_type = 1
        self.quant_max_bound = 126
        self.quant_min_bound = -126

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
        x,
        cache_kv_out,
        src_mask,
        qkv_out_scale,
        seq_len,
        out_linear_in_scale,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
    ):
        if qkv_out_scale is not None:
            x = x.cast(cache_kv_out.dtype) * qkv_out_scale
        else:
            x = x

        x = paddle.transpose(
            x, [0, 2, 1, 3]
        )  # [bz, seqlen, nhead, head_dim] --> [bz, nhead, seqlen, head_dim]
        q, k, v = paddle.split(x, 3, axis=2)
        cache_k, cache_v = paddle.split(cache_kv_out, 2, axis=0)
        k = paddle.concat([cache_k.squeeze(0), k], axis=2)
        v = paddle.concat([cache_v.squeeze(0), v], axis=2)

        product = paddle.matmul(
            x=q * (x.shape[3] ** -0.5), y=k, transpose_y=True
        )
        product = product + src_mask
        product = paddle.nn.functional.softmax(product)
        out = paddle.matmul(product, v).transpose([0, 2, 1, 3])

        normalized_out = self.quant_helper(
            out,
            out_linear_in_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
        return out, normalized_out

    def check_main(
        self,
        x,
        cache_kv_out,
        cache_kv_mmha_out,
        src_mask,
        qkv_out_scale,
        out_linear_in_scale,
        dtype,
    ):
        paddle.disable_static()
        if qkv_out_scale is not None:
            x = paddle.to_tensor(x).cast("int32")
            qkv_out_scale = paddle.to_tensor(qkv_out_scale).cast("float32")
        else:
            x = paddle.to_tensor(x).cast(dtype)
        src_mask = paddle.to_tensor(src_mask).cast(dtype)
        cache_kv_out = paddle.to_tensor(cache_kv_out).cast(dtype)
        cache_kv_mmha_out = paddle.to_tensor(cache_kv_mmha_out).cast(dtype)
        paddle_naive_mmha_out = 0
        paddle_naive_mmha_out = self.mmha_naive(
            x,
            cache_kv_out,
            src_mask,
            qkv_out_scale,
            self.seq_len,
            out_linear_in_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )

        paddle_mmha_out = mmha_wrapper(
            x,
            cache_kv_mmha_out,
            src_mask,
            None,
            None,
            None,
            None,
            qkv_out_scale,
            None,
            None,
            self.seq_len,
            self.rotary_emb_dims,
            self.use_neox_rotary_style,
            out_linear_in_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )
        paddle.enable_static()
        return paddle_naive_mmha_out, paddle_mmha_out

    def test_mmha_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle_naive_mmha, paddle_mmha_out = self.check_main(
            self.x,
            self.cache_kv_out,
            self.cache_kv_mmha_out,
            self.src_mask,
            None,
            -1,
            'float16',
        )
        np.testing.assert_allclose(
            paddle_mmha_out[0].numpy(),
            paddle_naive_mmha[0].numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_mmha_qkv_out_scale(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle_naive_mmha, paddle_mmha_out = self.check_main(
            self.x_int,
            self.cache_kv_out,
            self.cache_kv_mmha_out,
            self.src_mask,
            self.qkv_out_scale,
            -1,
            'float16',
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
            self.x,
            self.cache_kv_out,
            self.cache_kv_mmha_out,
            self.src_mask,
            None,
            self.out_linear_in_scale,
            'float16',
        )
        np.testing.assert_allclose(
            paddle_mmha_out[0].numpy(),
            paddle_naive_mmha[1].numpy(),
            rtol=1,
            atol=1,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestLayerNormStaticInt8Op(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.bsz = 2
        self.cache_bsz = 2
        self.num_head = 32
        self.dim_head = 128
        self.beam_size = 1
        self.max_seq_len = 33
        self.sequence_length = 32

        self.x = np.random.uniform(
            -0.05, 0.05, [self.bsz, 3, self.num_head, self.dim_head]
        )
        self.src_mask = np.zeros([self.bsz, 1, 1, self.sequence_length + 1])

        self.cum_offsets = None
        self.sequence_lengths = None
        self.rotary_tensor = None
        self.beam_cache_offset = None

        self.cache_kv_out = np.random.uniform(
            -0.05,
            0.05,
            [
                2,
                self.cache_bsz,
                self.num_head,
                self.sequence_length,
                self.dim_head,
            ],
        )
        numpy_ones = np.zeros(
            [2, self.cache_bsz, self.num_head, 1, self.dim_head]
        )
        self.cache_kv_mmha_out = np.concatenate(
            (self.cache_kv_out, numpy_ones), axis=3
        )

        self.qkv_out_scale = None
        self.out_linear_shift = None
        self.out_linear_smooth = None

        self.seq_len = 1
        self.rotary_emb_dims = 0
        self.use_neox_rotary_style = False

        self.out_linear_in_scale = -1
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.place = paddle.CUDAPlace(0)

    def mmha_naive(
        self,
        x,
        cache_kv_out,
        src_mask,
        qkv_out_scale,
        seq_len,
        out_linear_in_scale,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
    ):
        if qkv_out_scale is not None:
            x = x.cast(cache_kv_out.dtype) * qkv_out_scale

        x = paddle.transpose(
            x, [0, 2, 1, 3]
        )  # [bz, seqlen, nhead, head_dim] --> [bz, nhead, seqlen, head_dim]
        q, k, v = paddle.split(x, 3, axis=2)
        cache_k, cache_v = paddle.split(cache_kv_out, 2, axis=0)
        k = paddle.concat([cache_k.squeeze(0), k], axis=2)
        v = paddle.concat([cache_v.squeeze(0), v], axis=2)

        product = paddle.matmul(
            x=q * (x.shape[3] ** -0.5), y=k, transpose_y=True
        )
        product = product + src_mask
        product = paddle.nn.functional.softmax(product)
        out = paddle.matmul(product, v).transpose([0, 2, 1, 3])

        return out

    def check_main(
        self,
        x,
        src_mask,
        cache_kv_out,
        cache_kv_mmha_out,
        qkv_out_scale,
        out_linear_in_scale,
        dtype,
    ):
        paddle.disable_static()
        x_tensor = paddle.to_tensor(x).cast(dtype)
        src_mask_tensor = paddle.to_tensor(src_mask).cast(dtype)
        cache_kv_out = paddle.to_tensor(cache_kv_out).cast(dtype)

        paddle_naive_mmha_out = self.mmha_naive(
            x_tensor,
            cache_kv_out,
            src_mask_tensor,
            None,
            self.seq_len,
            out_linear_in_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )

        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x_static = paddle.static.data(
                name="x_static",
                shape=[self.bsz, 3, self.num_head, self.dim_head],
                dtype=dtype,
            )
            src_mask_static = paddle.static.data(
                name="src_mask_static",
                shape=[self.bsz, 1, 1, self.sequence_length + 1],
                dtype=dtype,
            )
            cache_kv_mmha_out_static = paddle.static.data(
                name="cache_kv_mmha_out_static",
                shape=[
                    2,
                    self.cache_bsz,
                    self.num_head,
                    self.sequence_length + 1,
                    self.dim_head,
                ],
                dtype=dtype,
            )

            outs = mmha_wrapper(
                x_static,
                cache_kv_mmha_out_static,
                src_mask_static,
                None,
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
            out_s = exe.run(
                feed={
                    "x_static": x.astype(dtype),
                    "cache_kv_mmha_out_static": cache_kv_mmha_out.astype(dtype),
                    "src_mask_static": src_mask.astype(dtype),
                },
                fetch_list=[outs],
            )

        return paddle_naive_mmha_out, out_s

    def test_mmha_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle_naive_mmha_out, paddle_mmha_out = self.check_main(
            self.x,
            self.src_mask,
            self.cache_kv_out,
            self.cache_kv_mmha_out,
            self.qkv_out_scale,
            self.out_linear_in_scale,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_mmha_out[0],
            paddle_naive_mmha_out.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )


if __name__ == '__main__':
    unittest.main()
