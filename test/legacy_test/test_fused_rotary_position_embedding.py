#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core
from paddle.incubate.nn.functional import fused_rotary_position_embedding


def deal_qkv(init_q, init_k, init_v):
    perm = [0, 2, 1, 3]
    q = paddle.transpose(x=init_q, perm=perm)
    k = paddle.transpose(x=init_k, perm=perm)
    v = paddle.transpose(x=init_v, perm=perm)
    return q, k, v


def mult_qkv(value, cos_tensor, sin_tensor):
    rotate_half_q = paddle.reshape(
        paddle.stack([value[:, :, :, 1::2], value[:, :, :, 0::2]], axis=-1),
        paddle.shape(value),
    )
    query = paddle.add(
        paddle.multiply(value, cos_tensor),
        paddle.multiply(rotate_half_q, sin_tensor),
    )
    return query


def mult_qkv_rotate_half(value, cos_tensor, sin_tensor):
    rotate_half_q = paddle.reshape(
        paddle.concat(
            [
                -value[..., value.shape[-1] // 2 :],
                value[..., : value.shape[-1] // 2],
            ],
            axis=-1,
        ),
        paddle.shape(value),
    )
    query = paddle.add(
        paddle.multiply(value, cos_tensor),
        paddle.multiply(rotate_half_q, sin_tensor),
    )
    return query


def get_sin_cos_tensor(seq_len, head_dim, sign):
    pos_seq = paddle.arange(0, seq_len, 1, dtype="float32")
    indices = paddle.arange(0, head_dim, 2, dtype="float32")

    indices = 1 / 10000 ** (indices / head_dim)
    sinusoid_inp = pos_seq.unsqueeze(1) * indices.unsqueeze(0)

    sin_sin = np.empty((seq_len * head_dim), dtype=np.float32)
    cos_cos = np.empty((seq_len * head_dim), dtype=np.float32)
    numpy_array = sinusoid_inp.numpy()
    iter_array = np.nditer(numpy_array)

    i = 0

    for value in iter_array:
        sin_sin[i * 2] = sign * np.sin(value)
        cos_cos[i * 2 + 0] = np.cos(value)
        sin_sin[i * 2 + 1] = np.sin(value)
        cos_cos[i * 2 + 1] = np.cos(value)
        i += 1

    tensor_sin = paddle.reshape(
        paddle.to_tensor(sin_sin),
        [1, seq_len, 1, head_dim],
    )
    tensor_cos = paddle.reshape(
        paddle.to_tensor(cos_cos),
        [1, seq_len, 1, head_dim],
    )

    return tensor_sin, tensor_cos


def paddle_fused_rotary_position_embedding(
    init_q, init_k, init_v, position_ids=None, use_neox_rotary_style=True
):
    # permute q, k, v from [batch_size, seq_len, num_heads, head_dim]
    # to [batch_size, num_heads, seq_len, head_dim]
    q, k, v = deal_qkv(init_q, init_k, init_v)

    sign = -1 if use_neox_rotary_style else 1
    sin_tensor, cos_tensor = get_sin_cos_tensor(q.shape[2], q.shape[3], sign)

    if position_ids is not None:
        sin_tensor = sin_tensor.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos_tensor = cos_tensor.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin_tensor = sin_tensor[position_ids].unsqueeze(
            2
        )  # [bs, seq_len, 1, dim]
        cos_tensor = cos_tensor[position_ids].unsqueeze(
            2
        )  # [bs, seq_len, 1, dim]

    perm = [0, 2, 1, 3]
    sin_tensor = paddle.transpose(x=sin_tensor, perm=perm)
    cos_tensor = paddle.transpose(x=cos_tensor, perm=perm)

    if use_neox_rotary_style:
        query = mult_qkv(q, cos_tensor, sin_tensor)
        value = mult_qkv(v, cos_tensor, sin_tensor)
        key = mult_qkv(k, cos_tensor, sin_tensor)
    else:
        query = mult_qkv_rotate_half(q, cos_tensor, sin_tensor)
        value = mult_qkv_rotate_half(v, cos_tensor, sin_tensor)
        key = mult_qkv_rotate_half(k, cos_tensor, sin_tensor)

    # permute the result back to [batch_size, seq_len, num_heads, head_dim]
    r_query, r_key, r_value = deal_qkv(query, key, value)
    return r_query, r_key, r_value


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA ",
)
class TestFusedRotaryPositionEmbedding(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 8, 2, 16]
        self.dtype = 'float32'
        self.training = True
        self.seed = 1203

    def get_paddle_tensor(self):
        tmp = paddle.randn(self.shape, self.dtype)
        tmp.stop_gradient = False
        return tmp

    def get_forward_backward(
        self,
        rope_function,
        seed,
        flag=False,
        use_neox_rotary_style=True,
        position_ids=None,
    ):
        paddle.disable_static()
        paddle.seed(seed)
        fw = []
        bw = []
        tensor_q = self.get_paddle_tensor()
        tensor_k = self.get_paddle_tensor()
        tensor_v = self.get_paddle_tensor()
        if use_neox_rotary_style:
            if flag:
                tensor_sin, tensor_cos = get_sin_cos_tensor(
                    tensor_q.shape[1], tensor_q.shape[3], 1
                )
                out_q, out_k, out_v = rope_function(
                    tensor_q,
                    tensor_k,
                    tensor_v,
                    tensor_sin,
                    tensor_cos,
                    position_ids=position_ids,
                )
            else:
                out_q, out_k, out_v = rope_function(
                    tensor_q, tensor_k, tensor_v, position_ids=position_ids
                )
        else:
            if flag:
                tensor_sin, tensor_cos = get_sin_cos_tensor(
                    tensor_q.shape[1], tensor_q.shape[3], 1
                )
                out_q, out_k, out_v = rope_function(
                    tensor_q,
                    tensor_k,
                    tensor_v,
                    tensor_sin,
                    tensor_cos,
                    position_ids=position_ids,
                    use_neox_rotary_style=False,
                )
            else:
                out_q, out_k, out_v = rope_function(
                    tensor_q,
                    tensor_k,
                    tensor_v,
                    position_ids=position_ids,
                    use_neox_rotary_style=False,
                )

        fw.append(out_q)
        fw.append(out_k)
        fw.append(out_v)

        out_gq = paddle.randn(out_q.shape, self.dtype)
        out_gk = paddle.randn(out_q.shape, self.dtype)
        out_gv = paddle.randn(out_q.shape, self.dtype)
        paddle.autograd.backward(
            [out_q, out_k, out_v], [out_gq, out_gk, out_gv], True
        )
        bw.append(tensor_q)
        bw.append(tensor_k)
        bw.append(tensor_v)

        return fw, bw

    def test_fused_rope(self):
        p_fw, p_bw = self.get_forward_backward(
            paddle_fused_rotary_position_embedding, seed=self.seed
        )
        f_fw, f_bw = self.get_forward_backward(
            fused_rotary_position_embedding, seed=self.seed
        )
        for i in range(len(p_fw)):
            np.testing.assert_allclose(
                p_fw[i].numpy(), f_fw[i].numpy(), rtol=1e-05
            )
            np.testing.assert_allclose(
                p_bw[i].numpy(), f_bw[i].numpy(), rtol=1e-05
            )

    def test_fused_rope_with_sin_cos(self):
        p_fw, p_bw = self.get_forward_backward(
            paddle_fused_rotary_position_embedding, seed=self.seed
        )
        f_fw, f_bw = self.get_forward_backward(
            fused_rotary_position_embedding, seed=self.seed, flag=True
        )
        for i in range(len(p_fw)):
            np.testing.assert_allclose(
                p_fw[i].numpy(), f_fw[i].numpy(), rtol=1e-05
            )
            np.testing.assert_allclose(
                p_bw[i].numpy(), f_bw[i].numpy(), rtol=1e-05
            )

    def test_fused_rope_rotate_half(self):
        p_fw, p_bw = self.get_forward_backward(
            paddle_fused_rotary_position_embedding,
            seed=self.seed,
            use_neox_rotary_style=False,
        )
        f_fw, f_bw = self.get_forward_backward(
            fused_rotary_position_embedding,
            seed=self.seed,
            use_neox_rotary_style=False,
        )
        for i in range(len(p_fw)):
            np.testing.assert_allclose(
                p_fw[i].numpy(), f_fw[i].numpy(), rtol=1e-05
            )
            np.testing.assert_allclose(
                p_bw[i].numpy(), f_bw[i].numpy(), rtol=1e-05
            )

    def test_fused_rope_position_ids(self):
        position_ids = paddle.to_tensor(
            [[7, 5, 4, 6, 3, 1, 2, 0], [3, 1, 4, 0, 7, 6, 5, 2]]
        )
        p_fw, p_bw = self.get_forward_backward(
            paddle_fused_rotary_position_embedding,
            seed=self.seed,
            position_ids=position_ids,
        )
        f_fw, f_bw = self.get_forward_backward(
            fused_rotary_position_embedding,
            seed=self.seed,
            flag=True,
            position_ids=position_ids,
        )
        for i in range(len(p_fw)):
            np.testing.assert_allclose(
                p_fw[i].numpy(), f_fw[i].numpy(), rtol=1e-05
            )
            np.testing.assert_allclose(
                p_bw[i].numpy(), f_bw[i].numpy(), rtol=1e-05
            )

    def test_error(self):
        paddle.enable_static()
        with self.assertRaises(RuntimeError):
            static_q = paddle.static.data(
                name="q", shape=self.shape, dtype=self.dtype
            )
            fused_rotary_position_embedding(static_q, static_q, static_q)
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
