#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from utils import static_guard

import paddle
from paddle.base.framework import in_dygraph_mode
from paddle.incubate.nn import FusedTransformerEncoderLayer
from paddle.nn import TransformerEncoderLayer


class TestFusedTransformerEncoderLayer(unittest.TestCase):
    def setActivation(self):
        self.activation = 'gelu'

    def setPreLayerNorm(self):
        self.pre_layer_norm = False

    def setAttnMask(self):
        self.has_attn_mask = True

    def setUp(self):
        self.batch_size = np.random.randint(1, 8)
        self.query_length = np.random.randint(1, 128)
        self.nhead = 16
        self.head_dim = 4
        self.num_heads = self.nhead
        self.d_model = self.head_dim * self.num_heads
        self.embed_dim = self.d_model
        self.dim_feedforward = np.random.randint(1, 32)
        self.dropout_rate = 0
        self.attn_dropout_rate = None
        self.act_dropout_rate = None
        self.attn_mask_type = np.float64
        self.key_length = self.query_length
        self.dtype = 'float32'
        self.setActivation()
        self.setPreLayerNorm()
        self.setAttnMask()

        self.rtol = 1e-3
        # FIXME(limin29): Because there is a problem with the test precision
        #  on A100, atol is temporarily set to 1e-2, and it will be
        #  changed back after the precision problem is solved.
        self.atol = 1e-2
        if "V100" in paddle.device.cuda.get_device_name():
            self.atol = 1e-4

    def fused_weight(self, weight, num_head):
        a = paddle.transpose(weight, perm=[1, 0])
        return paddle.reshape(
            a, shape=[1, num_head, int(a.shape[0] / num_head), a.shape[1]]
        )

    def fused_qkv(self, q, k, v, num_head):
        fq = self.fused_weight(q, num_head)
        fk = self.fused_weight(k, num_head)
        fv = self.fused_weight(v, num_head)
        return paddle.concat(x=[fq, fk, fv], axis=0)

    def test_out(self):
        if in_dygraph_mode():
            return
        paddle.seed(42)
        base_encoder = TransformerEncoderLayer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout_rate,
            self.activation,
            self.attn_dropout_rate,
            self.act_dropout_rate,
            self.pre_layer_norm,
        )
        src = np.random.rand(
            self.batch_size, self.query_length, self.embed_dim
        ).astype(self.dtype)

        if self.has_attn_mask:
            attn_mask = np.ones(
                (
                    self.batch_size,
                    self.num_heads,
                    self.query_length,
                    self.key_length,
                ),
                dtype=self.attn_mask_type,
            )
            attn_mask_tensor = paddle.to_tensor(attn_mask)
        else:
            attn_mask = None
            attn_mask_tensor = None

        dout = np.random.random(src.shape).astype(self.dtype)

        base_out = base_encoder(
            paddle.to_tensor(src, stop_gradient=False), attn_mask_tensor
        )
        paddle.autograd.backward([base_out], [paddle.to_tensor(dout)], True)

        fused_encoder = FusedTransformerEncoderLayer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout_rate,
            self.activation,
            self.attn_dropout_rate,
            self.act_dropout_rate,
            self.pre_layer_norm,
        )

        fused_encoder.ffn._linear1_weight.set_value(base_encoder.linear1.weight)
        fused_encoder.ffn._linear1_bias.set_value(base_encoder.linear1.bias)
        fused_encoder.ffn._linear2_weight.set_value(base_encoder.linear2.weight)
        fused_encoder.ffn._linear2_bias.set_value(base_encoder.linear2.bias)
        if self.pre_layer_norm:
            fused_encoder.ffn._ln1_scale.set_value(base_encoder.norm2.weight)
            fused_encoder.ffn._ln1_bias.set_value(base_encoder.norm2.bias)
        else:
            fused_encoder.ffn._ln2_scale.set_value(base_encoder.norm2.weight)
            fused_encoder.ffn._ln2_bias.set_value(base_encoder.norm2.bias)

        fused_encoder.fused_attn.linear_weight.set_value(
            base_encoder.self_attn.out_proj.weight
        )
        fused_encoder.fused_attn.linear_bias.set_value(
            base_encoder.self_attn.out_proj.bias
        )
        if self.pre_layer_norm:
            fused_encoder.fused_attn.pre_ln_scale.set_value(
                base_encoder.norm1.weight
            )
            fused_encoder.fused_attn.pre_ln_bias.set_value(
                base_encoder.norm1.bias
            )
        else:
            fused_encoder.fused_attn.ln_scale.set_value(
                base_encoder.norm1.weight
            )
            fused_encoder.fused_attn.ln_bias.set_value(base_encoder.norm1.bias)

        q = base_encoder.self_attn.q_proj.weight
        q_bias = base_encoder.self_attn.q_proj.bias
        k = base_encoder.self_attn.k_proj.weight
        k_bias = base_encoder.self_attn.k_proj.bias
        v = base_encoder.self_attn.v_proj.weight
        v_bias = base_encoder.self_attn.v_proj.bias
        qkv_weight = self.fused_qkv(q, k, v, self.num_heads)
        fused_encoder.fused_attn.qkv_weight.set_value(qkv_weight)

        tmp = paddle.concat(x=[q_bias, k_bias, v_bias], axis=0)
        qkv_bias = paddle.reshape(
            tmp,
            shape=[3, self.num_heads, int(tmp.shape[0] / 3 / self.num_heads)],
        )
        fused_encoder.fused_attn.qkv_bias.set_value(qkv_bias)

        fused_out = fused_encoder(
            paddle.to_tensor(src, stop_gradient=False), attn_mask_tensor
        )
        paddle.autograd.backward([fused_out], [paddle.to_tensor(dout)], True)

        correct_ffn_str = f'd_model={self.d_model}, dim_feedforward={self.dim_feedforward}, dropout_rate={self.dropout_rate}, epsilon={fused_encoder.ffn._epsilon}, activation={self.activation}, act_dropout_rate={self.dropout_rate}, normalize_before={self.pre_layer_norm}, dtype={self.dtype}'
        self.assertTrue(fused_encoder.ffn.extra_repr(), correct_ffn_str)

        correct_attn_str = f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout_rate={self.dropout_rate}, attn_dropout_rate={self.dropout_rate}, epsilon={fused_encoder.fused_attn._epsilon}, kdim={None}, vdim={None}, normalize_before={self.pre_layer_norm}, need_weights={False}, dtype={self.dtype}'
        self.assertTrue(fused_encoder.fused_attn.extra_repr(), correct_attn_str)

        np.testing.assert_allclose(
            fused_out.numpy(), base_out.numpy(), rtol=self.rtol, atol=self.atol
        )
        np.testing.assert_allclose(
            fused_out.grad.numpy(),
            base_out.grad.numpy(),
            rtol=self.rtol,
            atol=self.atol,
        )


class TestFusedTransformerEncoderLayerAct(TestFusedTransformerEncoderLayer):
    def setActivation(self):
        self.activation = 'relu'


class TestFusedTransformerEncoderLayerPreLayerNorm(
    TestFusedTransformerEncoderLayer
):
    def setPreLayerNorm(self):
        self.pre_layer_norm = True


class TestFusedTransformerEncoderLayerAttnMaskIsNone(
    TestFusedTransformerEncoderLayer
):
    def setAttnMask(self):
        self.has_attn_mask = False


class TestFusedTransformerEncoderLayerPreLnTrueAttnMaskIsNone(
    TestFusedTransformerEncoderLayer
):
    def setPreLayerNorm(self):
        self.pre_layer_norm = True

    def setAttnMask(self):
        self.has_attn_mask = False


class TestPirFusedTransformerEncoderLayer(unittest.TestCase):
    def run_program(self):
        with static_guard():
            paddle.seed(1)
            startup = paddle.static.Program()
            main = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                enc_input = paddle.rand((2, 4, 128))
                attn_mask = paddle.rand((2, 2, 4, 4))
                encoder_layer = FusedTransformerEncoderLayer(128, 2, 512)
                enc_output = encoder_layer(enc_input, attn_mask)

                exe = paddle.static.Executor()
                exe.run(startup)
                out = exe.run(feed={}, fetch_list=[enc_output])
                return out

    def test_pir(self):
        out1 = self.run_program()
        with paddle.pir_utils.IrGuard():
            out2 = self.run_program()
        np.testing.assert_allclose(out1, out2)


if __name__ == "__main__":
    unittest.main()
