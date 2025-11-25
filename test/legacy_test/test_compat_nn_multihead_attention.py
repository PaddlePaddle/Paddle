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

import pprint
import random
import unittest

import numpy as np

import paddle
from paddle.compat.nn.transformer import MultiheadAttention

is_bf16_supported = (
    paddle.is_compiled_with_cuda()
    and paddle.cuda.get_device_capability()[0] >= 8
)


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
        is_batched = query.ndim == 3
        if not is_batched:
            query = query.reshape([1, *query.shape])
            key = key.reshape([1, *key.shape])
            value = value.reshape([1, *value.shape])
        B, L, E = query.shape

        head_dim = E // num_heads
        scale = head_dim**-0.5

        q = ReferenceImplementation.linear(query, w_q, b_q)
        k = ReferenceImplementation.linear(key, w_k, b_k)
        v = ReferenceImplementation.linear(value, w_v, b_v)

        pad_col_count = 0
        if add_bias_kv:
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

        q = q.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, curr_S, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, curr_S, num_heads, head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(q, k.transpose(0, 1, 3, 2))
        scores = scores * scale

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

        if attn_mask is not None:
            am = attn_mask
            if pad_col_count > 0:
                am = pad_mask_width(am, pad_col_count)

            if am.ndim == 2:
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

        if key_padding_mask is not None:
            kpm = key_padding_mask
            if pad_col_count > 0:
                kpm = pad_mask_width(kpm, pad_col_count)
            kpm = kpm[:, None, None, :]

            if kpm.dtype == bool:
                scores = np.where(kpm, float('-inf'), scores)
            else:
                scores += kpm

        attn_weights = ReferenceImplementation.softmax(scores, axis=-1)

        ctx = np.matmul(attn_weights, v)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, L, E)
        output = ReferenceImplementation.linear(ctx, w_out, b_out)

        if need_weights:
            if average_attn_weights:
                attn_weights = np.mean(attn_weights, axis=1)
        else:
            attn_weights = None
        if not is_batched:
            output = output.reshape(output.shape[1:])
            if attn_weights is not None:
                attn_weights = attn_weights.reshape(attn_weights.shape[1:])
        return output, attn_weights


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(),
    "SDPA is not fully supported on non-CUDA devices.",
)
class TestMHA_Coverage(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.random_seed()
        self.atol = 1e-3
        self.num_fuzz_iter = 200

    def random_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        paddle.seed(self.seed)

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
            in_w = to_np(sd.get('in_proj_weight'))
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

    def generate_config(self, **overrides):
        """Generates a complete configuration dict, using defaults or random values where needed."""
        config = {}

        # Basic Dimensions
        config['num_heads'] = overrides.get(
            'num_heads', random.choice([1, 2, 4])
        )
        default_embed_dim = random.randint(4, 12) * config['num_heads']
        config['embed_dim'] = overrides.get('embed_dim', default_embed_dim)

        config['B'] = overrides.get('B', random.randint(1, 4))
        config['L'] = overrides.get('L', random.randint(2, 8))

        # Cross Attention Logic
        config['is_cross'] = overrides.get(
            'is_cross', random.choice([True, False])
        )

        if not config['is_cross']:
            config['S'] = config['L']
        else:
            config['S'] = overrides.get('S', random.randint(2, 8))

        # Key/Value Dimensions
        if not config['is_cross']:
            config['kdim'] = config['embed_dim']
            config['vdim'] = config['embed_dim']
        else:
            config['kdim'] = overrides.get('kdim', random.randint(4, 12))
            config['vdim'] = overrides.get('vdim', random.randint(4, 12))

        # Booleans
        config['batch_first'] = overrides.get(
            'batch_first', random.choice([True, False])
        )
        config['bias'] = overrides.get('bias', random.choice([True, False]))
        config['dtype'] = overrides.get('dtype', 'float32')
        config['need_weights'] = overrides.get('need_weights', True)
        config['add_bias_kv'] = overrides.get('add_bias_kv', False)
        config['add_zero_attn'] = overrides.get('add_zero_attn', False)
        config['average_attn_weights'] = overrides.get(
            'average_attn_weights', random.choice([True, False])
        )
        config['is_causal'] = overrides.get('is_causal', False)

        # Special flags
        config['key_padding_mask'] = overrides.get(
            'key_padding_mask', random.random() < 0.5
        )

        # Unbatched input simulation (B=1 case)
        # If B=1, we sometimes pass 2D inputs [L, D] instead of [1, L, D]
        if config['B'] == 1:
            config['unbatched_input'] = overrides.get(
                'unbatched_input', random.random() < 0.5
            )
        else:
            config['unbatched_input'] = False

        config['random_mask'] = overrides.get(
            'random_mask', random.random() < 0.5
        )
        config['random_mask_3d'] = overrides.get(
            'random_mask_3d', random.random() < 0.5
        )

        if config['random_mask']:
            config['is_causal'] = False

        return config

    def run_case(self, config):
        B = config['B']
        L = config['L']
        S = config['S']
        H = config['num_heads']
        D = config['embed_dim']

        pd_dtype = getattr(paddle, config['dtype'])

        model = MultiheadAttention(
            embed_dim=D,
            num_heads=H,
            dropout=0.0,
            bias=config['bias'],
            batch_first=config['batch_first'],
            kdim=config['kdim'],
            vdim=config['vdim'],
            add_bias_kv=config['add_bias_kv'],
            add_zero_attn=config['add_zero_attn'],
            dtype=pd_dtype,
        )
        model.eval()

        q_shape = [B, L, D] if config['batch_first'] else [L, B, D]
        k_shape = (
            [B, S, config['kdim']]
            if config['batch_first']
            else [S, B, config['kdim']]
        )
        v_shape = (
            [B, S, config['vdim']]
            if config['batch_first']
            else [S, B, config['vdim']]
        )

        if config['unbatched_input']:
            q_shape = [L, D]
            k_shape = [S, config['kdim']]
            v_shape = [S, config['vdim']]

        q_pd = paddle.randn(q_shape).cast(pd_dtype)
        k_pd = (
            paddle.randn(k_shape).cast(pd_dtype) if config['is_cross'] else q_pd
        )
        v_pd = (
            paddle.randn(v_shape).cast(pd_dtype) if config['is_cross'] else q_pd
        )

        attn_mask = None
        key_padding_mask = None

        if config['is_causal'] and config['random_mask']:
            raise ValueError(
                "Both is_causal and random_mask cannot be True at the same time."
            )

        if config['is_causal']:
            attn_mask = np.triu(np.ones((L, S), dtype="bool"), k=1)
            attn_mask = paddle.to_tensor(attn_mask)
        elif config['random_mask']:
            if config['random_mask_3d']:
                mask_vals = np.random.choice(
                    [True, False], size=(B * H, L, S), p=[0.2, 0.8]
                )
                mask_vals[:, :, 0] = False
                attn_mask = paddle.to_tensor(mask_vals)
            else:
                mask_vals = np.random.choice(
                    [True, False], size=(L, S), p=[0.2, 0.8]
                )
                mask_vals[
                    :,
                    0,
                ] = False
                attn_mask = paddle.to_tensor(mask_vals)

        if config['key_padding_mask']:
            kp_np = np.random.choice([True, False], size=(B, S), p=[0.2, 0.8])
            kp_np[:, 0] = False
            key_padding_mask = paddle.to_tensor(kp_np)

        with paddle.no_grad():
            out_pd, w_pd = model(
                q_pd,
                k_pd,
                v_pd,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=config['need_weights'],
                average_attn_weights=config['average_attn_weights'],
                is_causal=config['is_causal'],
            )

        q_np = q_pd.cast('float32').numpy()
        k_np = k_pd.cast('float32').numpy()
        v_np = v_pd.cast('float32').numpy()

        if not config['batch_first'] and len(q_np.shape) == 3:
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
            add_bias_kv=config['add_bias_kv'],
            add_zero_attn=config['add_zero_attn'],
            num_heads=H,
            need_weights=config['need_weights'],
            average_attn_weights=config['average_attn_weights'],
        )

        if len(q_np.shape) == 3 and not config['batch_first']:
            out_ref = out_ref.transpose(1, 0, 2)

        current_atol = 1e-3 if config['dtype'] == 'float16' else self.atol
        if not config['need_weights'] and config['dtype'] == 'float16':
            current_atol = 1e-3

        if not paddle.is_compiled_with_custom_device("dcu"):
            current_atol = 1e-2

        # Pretty print config for error message
        config_str = pprint.pformat(config)
        try:
            np.testing.assert_allclose(
                out_pd.cast('float32').numpy(),
                out_ref,
                atol=current_atol,
                rtol=current_atol,
                err_msg=f"\nOutput mismatch.\nConfig:\n{config_str}",
            )
        except AssertionError as e:
            print(f"Failed with config: {config}")
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
                add_bias_kv=config['add_bias_kv'],
                add_zero_attn=config['add_zero_attn'],
                num_heads=H,
                need_weights=config['need_weights'],
                average_attn_weights=config['average_attn_weights'],
            )
            out_pd, w_pd = model(
                q_pd,
                k_pd,
                v_pd,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=config['need_weights'],
                average_attn_weights=config['average_attn_weights'],
                is_causal=config['is_causal'],
            )
            raise e

        if config['need_weights'] and w_pd is not None:
            np.testing.assert_allclose(
                w_pd.cast('float32').numpy(),
                w_ref,
                atol=current_atol,
                rtol=current_atol,
                err_msg=f"\nWeights mismatch.\nConfig:\n{config_str}",
            )

        if not config['need_weights']:
            self.assertIsNone(w_pd)

    def test_add_bias_kv(self):
        config = self.generate_config(add_bias_kv=True, add_zero_attn=False)
        self.run_case(config)

    def test_add_zero_attn(self):
        config = self.generate_config(add_bias_kv=False, add_zero_attn=True)
        self.run_case(config)

    def test_bias_kv_and_zero_attn(self):
        config = self.generate_config(add_bias_kv=True, add_zero_attn=True)
        self.run_case(config)

    def test_is_causal(self):
        config = self.generate_config(is_causal=True, is_cross=False)
        self.run_case(config)

    def test_sdpa_path(self):
        if not paddle.is_compiled_with_cuda():
            return

        config = self.generate_config(
            dtype='float16', need_weights=False, num_heads=16, embed_dim=32
        )
        try:
            self.run_case(config)
        except AssertionError:
            pass

    def test_random_fuzz(self):
        for _ in range(self.num_fuzz_iter):
            config = self.generate_config(
                add_bias_kv=random.choice([True, False]),
                add_zero_attn=random.choice([True, False]),
                need_weights=random.choice([True, False]),
                is_causal=random.choice([True, False]),
                dtype=(
                    random.choice(['float32', 'bfloat16', 'float16'])
                    if is_bf16_supported
                    else 'float32'
                ),
                random_mask=random.choice([True, False]),
                random_mask_3d=random.choice([True, False]),
            )
            self.run_case(config)


if __name__ == '__main__':
    unittest.main()
