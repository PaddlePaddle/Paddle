# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

import numpy as np

import paddle
from paddle.compat.nn.transformer import MultiheadAttention


# ==============================================================================
# 1. Numpy Golden Reference Implementation
# ==============================================================================
class ReferenceImplementation:
    @staticmethod
    def softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        x_max[x_max == float('-inf')] = 0.0
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
        out = exp_x / (sum_exp + 1e-10)
        return out

    @staticmethod
    def linear(x, weight, bias=None):
        # weight: [In, Out]
        res = x @ weight
        if bias is not None:
            res += bias
        return res

    @staticmethod
    def forward(
        query,
        key,
        value,
        w_q,
        w_k,
        w_v,
        w_out,
        b_q,
        b_k,
        b_v,
        b_out,
        bias_k=None,
        bias_v=None,
        key_padding_mask=None,
        attn_mask=None,
        add_bias_kv=False,
        add_zero_attn=False,
        num_heads=4,
        need_weights=True,
        average_attn_weights=True,
    ):
        B, L, E = query.shape
        S = key.shape[1]
        head_dim = E // num_heads
        scale = head_dim**-0.5

        # 1. Linear Projections
        q = ReferenceImplementation.linear(query, w_q, b_q)
        k = ReferenceImplementation.linear(key, w_k, b_k)
        v = ReferenceImplementation.linear(value, w_v, b_v)

        # 2. Handle Special Flags
        pad_col_count = 0
        if add_bias_kv:
            # bias_k/v: [1, 1, E] -> [B, 1, E]
            if bias_k is not None:
                bk = np.tile(bias_k, (B, 1, 1))
                bv = np.tile(bias_v, (B, 1, 1))
                k = np.concatenate([k, bk], axis=1)
                v = np.concatenate([v, bv], axis=1)
                pad_col_count += 1

        if add_zero_attn:
            zeros = np.zeros((B, 1, E), dtype=q.dtype)
            k = np.concatenate([k, zeros], axis=1)
            v = np.concatenate([v, zeros], axis=1)
            pad_col_count += 1

        curr_S = k.shape[1]

        # 3. Split Heads & Transpose
        q = q.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, curr_S, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, curr_S, num_heads, head_dim).transpose(0, 2, 1, 3)

        # 4. Scaled Dot Product
        scores = np.matmul(q, k.transpose(0, 1, 3, 2))
        scores = scores * scale

        # 5. Apply Masks
        def pad_mask_width(mask_arr, pad_amt):
            if pad_amt == 0:
                return mask_arr
            shape = list(mask_arr.shape)
            shape[-1] = pad_amt
            if mask_arr.dtype == bool:
                pad = np.zeros(shape, dtype=bool)
            else:
                pad = np.zeros(shape, dtype=mask_arr.dtype)
            return np.concatenate([mask_arr, pad], axis=-1)

        # Apply Attn Mask
        if attn_mask is not None:
            am = attn_mask
            if pad_col_count > 0:
                am = pad_mask_width(am, pad_col_count)

            if am.ndim == 2:  # (L, S)
                am = am[None, None, :, :]
            elif am.ndim == 3:
                if am.shape[0] == B * num_heads:
                    am = am.reshape(B, num_heads, L, -1)
                elif am.shape[0] == B:
                    am = am[:, None, :, :]

            if am.dtype == bool:
                scores = np.where(am, float('-inf'), scores)
            else:
                scores += am

        # Apply Key Padding Mask
        if key_padding_mask is not None:
            kpm = key_padding_mask
            if pad_col_count > 0:
                kpm = pad_mask_width(kpm, pad_col_count)
            kpm = kpm[:, None, None, :]  # (B, 1, 1, S)

            if kpm.dtype == bool:
                scores = np.where(kpm, float('-inf'), scores)
            else:
                scores += kpm

        # 6. Softmax
        attn_weights = ReferenceImplementation.softmax(scores, axis=-1)

        # 7. Output
        ctx = np.matmul(attn_weights, v)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, L, E)
        output = ReferenceImplementation.linear(ctx, w_out, b_out)

        if need_weights:
            if average_attn_weights:
                # Average over heads: [B, H, L, S] -> [B, L, S]
                attn_weights = np.mean(attn_weights, axis=1)
            # else: keep [B, H, L, S]
        else:
            attn_weights = None

        return output, attn_weights


# ==============================================================================
# 2. Fuzzing Test Case
# ==============================================================================
class TestMHA_Fuzzing(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        self.atol = 2e-4
        self.num_fuzz_iter = 50

    def _extract_weights(self, layer):
        sd = layer.state_dict()
        w = {}

        def to_np(t):
            return t.cast('float32').numpy() if t is not None else None

        def safe_T(arr):
            return arr.T if arr is not None else None

        w['w_out'] = safe_T(to_np(sd.get('out_proj.weight')))
        w['b_out'] = to_np(sd.get('out_proj.bias'))
        w['bias_k'] = to_np(sd.get('bias_k'))
        w['bias_v'] = to_np(sd.get('bias_v'))

        if layer._qkv_same_embed_dim:
            in_w = to_np(sd['in_proj_weight'])
            if in_w is not None:
                in_w_t = in_w.T
                w['w_q'], w['w_k'], w['w_v'] = np.split(in_w_t, 3, axis=1)
            else:
                w['w_q'] = w['w_k'] = w['w_v'] = None

            if sd.get('in_proj_bias') is not None:
                in_b = to_np(sd['in_proj_bias'])
                w['b_q'], w['b_k'], w['b_v'] = np.split(in_b, 3, axis=0)
            else:
                w['b_q'] = w['b_k'] = w['b_v'] = None
        else:
            w['w_q'] = safe_T(to_np(sd.get('q_proj_weight')))
            w['w_k'] = safe_T(to_np(sd.get('k_proj_weight')))
            w['w_v'] = safe_T(to_np(sd.get('v_proj_weight')))
            w['b_q'] = to_np(sd.get('q_proj_bias'))
            w['b_k'] = to_np(sd.get('k_proj_bias'))
            w['b_v'] = to_np(sd.get('v_proj_bias'))

        return w

    def run_single_case(self, **kwargs):
        # Random parameters
        B = random.randint(1, 4)
        L = random.randint(2, 8)
        S = random.randint(2, 8)
        H = random.choice([1, 2, 4])
        D = random.randint(4, 12) * H

        batch_first = kwargs.get('batch_first', random.choice([True, False]))
        is_cross = kwargs.get('is_cross', random.choice([True, False]))
        use_bias = kwargs.get('bias', random.choice([True, False]))
        avg_weights = kwargs.get(
            'average_attn_weights', random.choice([True, False])
        )

        if not is_cross:
            S = L
            kdim = vdim = D
        else:
            kdim = random.randint(4, 12)
            vdim = random.randint(4, 12)

        # 1. Init Model
        model = MultiheadAttention(
            embed_dim=D,
            num_heads=H,
            bias=use_bias,
            batch_first=batch_first,
            kdim=kdim,
            vdim=vdim,
            add_bias_kv=kwargs.get('add_bias_kv', False),
            add_zero_attn=kwargs.get('add_zero_attn', False),
        )
        model.eval()

        # 2. Inputs
        q_shape = [B, L, D] if batch_first else [L, B, D]
        k_shape = [B, S, kdim] if batch_first else [S, B, kdim]
        v_shape = [B, S, vdim] if batch_first else [S, B, vdim]

        q_pd = paddle.randn(q_shape)
        k_pd = paddle.randn(k_shape) if is_cross else q_pd
        v_pd = paddle.randn(v_shape) if is_cross else q_pd

        # 3. Masks (Improved Sanitization: Column-0 Safety Rule)
        attn_mask = None
        key_padding_mask = None

        has_kp_mask = random.random() < 0.5
        has_attn_mask = random.random() < 0.5

        kp_tensor = None
        attn_tensor = None

        if has_kp_mask:
            # (B, S)
            kp_np = np.random.choice([True, False], size=(B, S), p=[0.2, 0.8])
            kp_np[:, 0] = False
            kp_tensor = paddle.to_tensor(kp_np)

        if has_attn_mask:
            mask_type = random.choice(['2d', '3d_bool', '3d_float'])
            if mask_type == '2d':
                # (L, S)
                attn_np = np.random.rand(L, S) > 0.8
                attn_np[:, 0] = False
                attn_tensor = paddle.to_tensor(attn_np)
            elif mask_type == '3d_bool':
                # (B*H, L, S)
                attn_np = np.random.rand(B * H, L, S) > 0.8
                attn_np[:, :, 0] = False
                attn_tensor = paddle.to_tensor(attn_np)
            else:  # 3d_float
                # (B*H, L, S)
                attn_np = np.zeros((B * H, L, S), dtype='float32')
                attn_np[np.random.rand(B * H, L, S) > 0.8] = float('-inf')
                attn_np[:, :, 0] = 0.0
                attn_tensor = paddle.to_tensor(attn_np)

        if has_kp_mask:
            key_padding_mask = kp_tensor
        if has_attn_mask:
            attn_mask = attn_tensor

        # 4. Forward Paddle
        with paddle.no_grad():
            out_pd, w_pd = model(
                q_pd,
                k_pd,
                v_pd,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=True,
                average_attn_weights=avg_weights,
            )

        # 5. Forward Numpy Reference
        q_np = q_pd.numpy()
        k_np = k_pd.numpy()
        v_np = v_pd.numpy()

        if not batch_first:
            q_np = q_np.transpose(1, 0, 2)
            k_np = k_np.transpose(1, 0, 2)
            v_np = v_np.transpose(1, 0, 2)

        weights = self._extract_weights(model)

        kp_np = (
            key_padding_mask.numpy() if key_padding_mask is not None else None
        )
        am_np = attn_mask.numpy() if attn_mask is not None else None

        out_ref, w_ref = ReferenceImplementation.forward(
            q_np,
            k_np,
            v_np,
            w_q=weights['w_q'],
            w_k=weights['w_k'],
            w_v=weights['w_v'],
            w_out=weights['w_out'],
            b_q=weights['b_q'],
            b_k=weights['b_k'],
            b_v=weights['b_v'],
            b_out=weights['b_out'],
            bias_k=weights['bias_k'],
            bias_v=weights['bias_v'],
            key_padding_mask=kp_np,
            attn_mask=am_np,
            add_bias_kv=kwargs.get('add_bias_kv', False),
            add_zero_attn=kwargs.get('add_zero_attn', False),
            num_heads=H,
            need_weights=True,
            average_attn_weights=avg_weights,
        )

        if not batch_first:
            out_ref = out_ref.transpose(1, 0, 2)

        # 6. Assertions
        np.testing.assert_allclose(
            out_pd.numpy(),
            out_ref,
            atol=self.atol,
            rtol=self.atol,
            err_msg=f"Output Mismatch. Config: bias={use_bias}, avg={avg_weights}, masks={has_kp_mask}&{has_attn_mask}",
        )

        if w_pd is not None:
            np.testing.assert_allclose(
                w_pd.numpy(),
                w_ref,
                atol=self.atol,
                rtol=self.atol,
                err_msg="Weights Mismatch.",
            )

    def test_ultimate_fuzz(self):
        print("\nRunning Ultimate Fuzz (All Params Mixed)...")
        for i in range(self.num_fuzz_iter):
            self.run_single_case()


if __name__ == '__main__':
    unittest.main()
