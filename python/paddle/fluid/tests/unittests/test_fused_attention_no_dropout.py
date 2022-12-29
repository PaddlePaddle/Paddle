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
import paddle.nn.functional as F
from paddle.incubate.nn import FusedMultiHeadAttention


def random_init_model(model, seed):
    paddle.seed(seed)
    for p in model.parameters():
        shape = p.shape
        dtype = p.dtype
        value = paddle.randn(shape=shape, dtype=dtype)
        p.set_value(value.numpy())


class FusedAttentionTestLayer(FusedMultiHeadAttention):
    def __init__(self, embed_dim, num_heads, normalize_before=False):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout_rate=0.0,
            dropout_rate=0.0,
            normalize_before=normalize_before,
        )

    def _reshape_and_transpose(self, x):
        assert len(x.shape) == 3
        bs, seq_len = x.shape[:2]
        x = x.reshape([bs, seq_len, self.num_heads, self.head_dim])
        x = x.transpose([0, 2, 1, 3])
        return x

    def _transpose_and_reshape(self, x):
        assert len(x.shape) == 4
        x = x.transpose([0, 2, 1, 3])
        bs = x.shape[0]
        x = x.reshape([bs, -1, self.embed_dim])
        return x

    def forward(self, x, attn_mask, use_ref=False):
        if use_ref:
            return self.ref_forward(x, attn_mask)
        else:
            return super().forward(x, attn_mask)

    def ref_forward(self, x, attn_mask):
        residual = x
        if self.normalize_before:
            assert len(self.pre_ln_scale.shape) == 1
            out = F.layer_norm(
                x,
                self.pre_ln_scale.shape,
                weight=self.pre_ln_scale,
                bias=self.pre_ln_bias,
                epsilon=self._epsilon,
            )
        else:
            out = x
        qkv_weight = self.qkv_weight.reshape(
            [3 * self.embed_dim, self.embed_dim]
        )
        qkv_bias = self.qkv_bias.reshape([3 * self.embed_dim])
        out = paddle.matmul(out, qkv_weight, transpose_y=True) + qkv_bias
        # [BS, seq_len, head_dim]
        # [BS, seq_len, head_dim * 3]
        q, k, v = paddle.split(out, 3, axis=-1)
        q = self._reshape_and_transpose(q)
        k = self._reshape_and_transpose(k)
        v = self._reshape_and_transpose(v)
        q *= self.head_dim**-0.5
        out = paddle.matmul(q, k, transpose_y=True)
        if attn_mask is not None:
            out += attn_mask
        out = F.softmax(out)
        out = paddle.matmul(out, v)
        out = self._transpose_and_reshape(out)
        out = F.linear(out, weight=self.linear_weight, bias=self.linear_bias)

        add_residual = True
        if add_residual:
            out = residual + out
        if not self.normalize_before:
            assert len(self.ln_scale.shape) == 1
            out = F.layer_norm(
                out,
                self.ln_scale.shape,
                weight=self.ln_scale,
                bias=self.ln_bias,
                epsilon=self._epsilon,
            )
        return out


class TestFusedAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.num_heads = 16
        self.max_seq_len = 128
        self.hidden_size = 256
        self.dtype = "float32"
        self.normalize_before = False
        self.seed = 10
        self.use_mask = False
        self.set_configs()

    def set_configs(self):
        pass

    def generate_inputs(self):
        np.random.seed(self.seed)
        hidden_state = np.random.random(
            size=[self.batch_size, self.max_seq_len, self.hidden_size]
        ).astype(self.dtype)
        hidden_state = paddle.to_tensor(hidden_state)
        hidden_state.stop_gradient = False
        if self.use_mask:
            seq_lens = np.random.randint(
                low=int(self.max_seq_len / 3),
                high=self.max_seq_len,
                size=[self.batch_size],
            )
            mask = np.zeros(
                shape=[self.batch_size, self.max_seq_len], dtype=self.dtype
            )
            for i in range(self.batch_size):
                mask[i][0 : seq_lens[i]] = 1
            mask = mask.reshape([self.batch_size, 1, 1, self.max_seq_len])
            broadcast_shape = [
                self.batch_size,
                self.num_heads,
                self.max_seq_len,
                self.max_seq_len,
            ]
            mask = np.broadcast_to(mask, broadcast_shape)
            mask = (1 - mask) * -1e9
            return hidden_state, paddle.to_tensor(mask.astype(self.dtype))
        else:
            return hidden_state, None

    def run_fwd_bwd(self, use_ref=False):
        x, mask = self.generate_inputs()
        layer = FusedAttentionTestLayer(
            self.hidden_size,
            self.num_heads,
            normalize_before=self.normalize_before,
        )
        random_init_model(layer, self.seed + 100)
        out = layer(x, mask, use_ref)
        loss = out.mean()
        loss.backward()
        vars_need_gradients = [('out', x)] + list(layer.named_parameters())
        numpy_values = [out.numpy()]
        for i, (name, var) in enumerate(vars_need_gradients):
            tmp = var.grad.numpy()
            numpy_values.append(tmp)
        return numpy_values

    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return
        values1 = self.run_fwd_bwd(True)
        paddle.device.cuda.synchronize()
        values2 = self.run_fwd_bwd(False)
        paddle.device.cuda.synchronize()
        self.assertEqual(len(values1), len(values2))
        for i, (v1, v2) in enumerate(zip(values1, values2)):
            if not self.normalize_before:
                np.testing.assert_allclose(v1, v2, atol=1e-6, rtol=1e-5)
            else:
                np.testing.assert_equal(v1, v2)


class TestFusedAttentionNormalizeBefore(TestFusedAttention):
    def set_configs(self):
        self.normalize_before = True


if __name__ == "__main__":
    unittest.main()
