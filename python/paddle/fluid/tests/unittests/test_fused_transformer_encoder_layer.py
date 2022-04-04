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
import numpy as np

import paddle
from paddle.incubate.nn import FusedTransformerEncoderLayer
from paddle.nn import TransformerEncoderLayer
from paddle.fluid.framework import default_main_program, in_dygraph_mode
import unittest


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

    def fused_weight(self, weight, num_head):
        a = paddle.transpose(weight, perm=[1, 0])
        return paddle.reshape(
            a, shape=[1, num_head, int(a.shape[0] / num_head), a.shape[1]])

    def fused_qkv(self, q, k, v, num_head):
        fq = self.fused_weight(q, num_head)
        fk = self.fused_weight(k, num_head)
        fv = self.fused_weight(v, num_head)
        return paddle.concat(x=[fq, fk, fv], axis=0)

    def test_out(self):
        if in_dygraph_mode():
            return
        default_main_program().random_seed = 42
        base_encoder = TransformerEncoderLayer(
            self.d_model, self.nhead, self.dim_feedforward, self.dropout_rate,
            self.activation, self.attn_dropout_rate, self.act_dropout_rate,
            self.pre_layer_norm)
        src = np.random.rand(self.batch_size, self.query_length,
                             self.embed_dim).astype(self.dtype)

        if self.has_attn_mask:
            attn_mask = np.ones(
                (self.batch_size, self.num_heads, self.query_length,
                 self.key_length),
                dtype=self.attn_mask_type)
            attn_mask_tensor = paddle.to_tensor(attn_mask)
        else:
            attn_mask = None
            attn_mask_tensor = None

        dout = np.random.random(src.shape).astype(self.dtype)

        base_out = base_encoder(
            paddle.to_tensor(
                src, stop_gradient=False), attn_mask_tensor)
        paddle.autograd.backward([base_out], [paddle.to_tensor(dout)], True)

        fused_encoder = FusedTransformerEncoderLayer(
            self.d_model, self.nhead, self.dim_feedforward, self.dropout_rate,
            self.activation, self.attn_dropout_rate, self.act_dropout_rate,
            self.pre_layer_norm)

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
            base_encoder.self_attn.out_proj.weight)
        fused_encoder.fused_attn.linear_bias.set_value(
            base_encoder.self_attn.out_proj.bias)
        if self.pre_layer_norm:
            fused_encoder.fused_attn.pre_ln_scale.set_value(
                base_encoder.norm1.weight)
            fused_encoder.fused_attn.pre_ln_bias.set_value(
                base_encoder.norm1.bias)
        else:
            fused_encoder.fused_attn.ln_scale.set_value(
                base_encoder.norm1.weight)
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
            shape=[3, self.num_heads, int(tmp.shape[0] / 3 / self.num_heads)])
        fused_encoder.fused_attn.qkv_bias.set_value(qkv_bias)

        fused_out = fused_encoder(
            paddle.to_tensor(
                src, stop_gradient=False), attn_mask_tensor)
        paddle.autograd.backward([fused_out], [paddle.to_tensor(dout)], True)

        correct_ffn_str = 'd_model={}, dim_feedforward={}, dropout_rate={}, epsilon={}, activation={}, act_dropout_rate={}, normalize_before={}, dtype={}'.format(
            self.d_model, self.dim_feedforward, self.dropout_rate,
            fused_encoder.ffn._epsilon, self.activation, self.dropout_rate,
            self.pre_layer_norm, self.dtype)
        self.assertTrue(fused_encoder.ffn.extra_repr(), correct_ffn_str)

        correct_attn_str = 'embed_dim={}, num_heads={}, dropout_rate={}, attn_dropout_rate={}, epsilon={}, kdim={}, vdim={}, normalize_before={}, need_weights={}, dtype={}'.format(
            self.embed_dim, self.num_heads, self.dropout_rate,
            self.dropout_rate, fused_encoder.fused_attn._epsilon, None, None,
            self.pre_layer_norm, False, self.dtype)
        self.assertTrue(fused_encoder.fused_attn.extra_repr(), correct_attn_str)

        np.testing.assert_allclose(
            fused_out.numpy(), base_out.numpy(), rtol=1e-3, atol=1e-4)
        self.assertTrue(
            np.allclose(
                fused_out.grad.numpy(),
                base_out.grad.numpy(),
                rtol=1e-3,
                atol=1e-4))


class TestFusedTransformerEncoderLayerAct(TestFusedTransformerEncoderLayer):
    def setActivation(self):
        self.activation = 'relu'


class TestFusedTransformerEncoderLayerPreLayerNorm(
        TestFusedTransformerEncoderLayer):
    def setPreLayerNorm(self):
        self.pre_layer_norm = True


class TestFusedTransformerEncoderLayerAttnMaskIsNone(
        TestFusedTransformerEncoderLayer):
    def setAttnMask(self):
        self.has_attn_mask = False


class TestFusedTransformerEncoderLayerPreLnTrueAttnMaskIsNone(
        TestFusedTransformerEncoderLayer):
    def setPreLayerNorm(self):
        self.pre_layer_norm = True

    def setAttnMask(self):
        self.has_attn_mask = False


if __name__ == "__main__":
    unittest.main()
