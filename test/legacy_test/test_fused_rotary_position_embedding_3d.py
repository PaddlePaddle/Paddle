#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import parameterized as param

import paddle
from paddle.base import core
from paddle.incubate.nn.functional import fused_rotary_position_embedding_3d
from paddle.pir_utils import test_with_pir_api


def rotate_permutation(x):
    x1 = x[..., : x.shape[-1] // 6]
    x2 = x[..., x.shape[-1] // 6 : x.shape[-1] // 3]
    x3 = x[..., x.shape[-1] // 3 : x.shape[-1] // 2]
    x4 = x[..., x.shape[-1] // 2 : x.shape[-1] // 3 * 2]
    x5 = x[..., x.shape[-1] // 3 * 2 : x.shape[-1] // 6 * 5]
    x6 = x[..., x.shape[-1] // 6 * 5 :]
    return paddle.concat((-x2, x1, -x4, x3, -x6, x5), axis=-1)


def apply_rotary_pos_emb(q, k, v, sin, cos):
    if len(sin.shape) == 6:
        target_shape = [1, q.shape[1], 1, q.shape[3]]
        sin = paddle.reshape(sin, target_shape)
        cos = paddle.reshape(cos, target_shape)
    q_embed = (
        (q * cos) + (rotate_permutation(q) * sin) if q is not None else None
    )
    k_embed = (
        (k * cos) + (rotate_permutation(k) * sin) if k is not None else None
    )
    v_embed = (
        (v * cos) + (rotate_permutation(v) * sin) if v is not None else None
    )
    return q_embed, k_embed, v_embed


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA ",
)
@param.parameterized_class(
    ("name", 'shape_q', 'shape_k', 'shape_v', 'shape_sin'),
    [
        (
            "qkv_input",
            [2, 8, 2, 24],  # bs, seq_len, num_heads, head_dim
            [2, 8, 2, 24],  # bs, seq_len, num_heads, head_dim
            [2, 8, 2, 24],  # bs, seq_len, num_heads, head_dim
            [1, 8, 1, 24],  # bs, seq_len, num_heads, head_dim
        ),
        ("qk_input", [2, 8, 2, 24], [2, 8, 2, 24], None, [1, 2, 2, 2, 1, 24]),
        ("qv_input", [2, 8, 2, 24], None, [2, 8, 2, 24], [1, 2, 2, 2, 1, 24]),
        ("q_input", [2, 8, 2, 24], None, None, [1, 2, 2, 2, 1, 24]),
        (
            "qkv_input_mqa",
            [2, 8, 4, 24],
            [2, 8, 1, 24],
            [2, 8, 1, 24],
            [1, 2, 2, 2, 1, 24],
        ),
        ("qk_input_mqa", [2, 8, 4, 24], [2, 8, 1, 24], None, [1, 8, 1, 24]),
        ("qv_input_mqa", [2, 8, 4, 24], None, [2, 8, 1, 24], [1, 8, 1, 24]),
        (
            "qkv_input_gqa",
            [1, 8, 4, 24],
            [1, 8, 2, 24],
            [1, 8, 2, 24],
            [1, 2, 2, 2, 1, 24],
        ),
        (
            "qk_input_gqa",
            [1, 8, 4, 24],
            [1, 8, 2, 24],
            None,
            [1, 8, 1, 24],
        ),
        (
            "qv_input_gqa",
            [1, 8, 4, 24],
            None,
            [1, 8, 2, 24],
            [1, 8, 1, 24],
        ),
    ],
)
class TestFusedRotaryPositionEmbedding(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float32'
        self.training = True
        self.seed = 1203
        self.rtol = 2e-5

    def get_paddle_tensor(self, shape):
        if shape is None:
            return None

        tmp = paddle.randn(shape, self.dtype)
        tmp.stop_gradient = False
        return tmp

    def get_inputs(self, seed):
        paddle.disable_static()
        paddle.seed(seed)
        # tensor_q shape: [batch_size, seq_len, num_heads, head_dim]
        tensor_q = self.get_paddle_tensor(self.shape_q)
        tensor_k = self.get_paddle_tensor(self.shape_k)
        tensor_v = self.get_paddle_tensor(self.shape_v)
        tensor_sin = self.get_paddle_tensor(self.shape_sin)
        tensor_cos = self.get_paddle_tensor(self.shape_sin)

        return tensor_q, tensor_k, tensor_v, tensor_sin, tensor_cos

    def get_forward_backward(self, rope_function, seed):
        paddle.disable_static()
        fw = []
        bw = []

        tensor_q, tensor_k, tensor_v, tensor_sin, tensor_cos = self.get_inputs(
            seed
        )
        out_q, out_k, out_v = rope_function(
            tensor_q,
            tensor_k,
            tensor_v,
            tensor_sin,
            tensor_cos,
        )

        out_init_grad = []
        for out_value in [out_q, out_k, out_v]:
            if out_value is None or not out_value._is_initialized():
                continue
            fw.append(out_value)
            out_init_grad.append(paddle.randn(out_value.shape, self.dtype))

        paddle.autograd.backward(fw, out_init_grad, True)
        bw = list(
            filter(lambda x: x is not None, [tensor_q, tensor_k, tensor_v])
        )

        return fw, bw

    def check_results(self, p_results, f_results):
        for i in range(len(p_results)):
            np.testing.assert_allclose(
                p_results[i].numpy(),
                f_results[i].numpy(),
                rtol=self.rtol,
            )

    def test_fused_rope(self):
        p_fw, p_bw = self.get_forward_backward(
            apply_rotary_pos_emb, seed=self.seed
        )
        f_fw, f_bw = self.get_forward_backward(
            fused_rotary_position_embedding_3d,
            seed=self.seed,
        )

        self.check_results(p_fw, f_fw)
        self.check_results(p_bw, f_bw)

    @test_with_pir_api
    def test_static(self):
        paddle.disable_static()
        tensor_q, tensor_k, tensor_v, tensor_sin, tensor_cos = self.get_inputs(
            self.seed
        )
        p_fw, p_bw = self.get_forward_backward(
            apply_rotary_pos_emb,
            seed=self.seed,
        )
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            q = (
                None
                if self.shape_q is None
                else paddle.static.data(
                    name="q", shape=self.shape_q, dtype=self.dtype
                )
            )

            k = (
                None
                if self.shape_k is None
                else paddle.static.data(
                    name="k", shape=self.shape_k, dtype=self.dtype
                )
            )

            v = (
                None
                if self.shape_v is None
                else paddle.static.data(
                    name="v", shape=self.shape_v, dtype=self.dtype
                )
            )

            sin = paddle.static.data(
                name="sin",
                shape=self.shape_sin,
                dtype=self.dtype,
            )
            cos = paddle.static.data(
                name="cos",
                shape=self.shape_sin,
                dtype=self.dtype,
            )

            out_q, out_k, out_v = fused_rotary_position_embedding_3d(
                q,
                k,
                v,
                sin,
                cos,
            )

        exe = paddle.static.Executor()

        feed = {
            'sin': tensor_sin.numpy(),
            'cos': tensor_cos.numpy(),
        }
        for var_name, input_tensor in zip(
            ['q', 'k', 'v'], [tensor_q, tensor_k, tensor_v]
        ):
            if input_tensor is not None:
                feed[var_name] = input_tensor.numpy()

        fetch_list = []
        for x, out in zip([q, k, v], [out_q, out_k, out_v]):
            # The reason why fetch `out` based on `x` is that
            # if input is None, the output of static function might be not NoneType
            # but pir.Value with type builtin.tensor<0xf32> in pir mode.
            if x is not None:
                fetch_list.append(out)

        outs = exe.run(
            main,
            feed=feed,
            fetch_list=fetch_list,
        )

        for i in range(len(p_fw)):
            np.testing.assert_allclose(
                p_fw[i].numpy(),
                outs[i],
                rtol=self.rtol,
            )
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
