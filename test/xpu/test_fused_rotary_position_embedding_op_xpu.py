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
from op_test import convert_uint16_to_float

import paddle
from paddle import base
from paddle.incubate.nn.functional import fused_rotary_position_embedding


def transpose_qkv(init_q, init_k, init_v):
    perm = [0, 2, 1, 3]
    q = paddle.transpose(x=init_q, perm=perm)
    k = paddle.transpose(x=init_k, perm=perm)
    v = paddle.transpose(x=init_v, perm=perm)
    return q, k, v


def mult_qkv_rotate_every_two(value, cos_tensor, sin_tensor):
    rotate_two_q = paddle.stack(
        [-value[:, :, :, 1::2], value[:, :, :, 0::2]], axis=-1
    ).reshape(value.shape)
    query = paddle.add(
        paddle.multiply(value, cos_tensor),
        paddle.multiply(rotate_two_q, sin_tensor),
    )
    return query


def mult_qkv_rotate_half(value, cos_tensor, sin_tensor):
    rotate_half_q = paddle.concat(
        [
            -value[..., value.shape[-1] // 2 :],
            value[..., : value.shape[-1] // 2],
        ],
        axis=-1,
    ).reshape(value.shape)
    query = paddle.add(
        paddle.multiply(value, cos_tensor),
        paddle.multiply(rotate_half_q, sin_tensor),
    )
    return query


def get_sin_cos_tensor(seq_len, head_dim, dtype="float32"):
    pos_seq = paddle.arange(0, seq_len, 1).astype("float32")
    indices = paddle.arange(0, head_dim, 2).astype("float32")

    indices = 1 / (10000 ** (indices / head_dim))
    sinusoid_inp = pos_seq.unsqueeze(1) * indices.unsqueeze(0)
    sinusoid_inp = paddle.stack([sinusoid_inp, sinusoid_inp], axis=-1)
    sin = paddle.sin(sinusoid_inp)
    cos = paddle.cos(sinusoid_inp)
    sin = sin.astype(dtype).reshape([1, seq_len, 1, head_dim])
    cos = cos.astype(dtype).reshape([1, seq_len, 1, head_dim])
    return sin, cos


def ref_rotary_position_embedding(
    init_q,
    init_k=None,
    init_v=None,
    sin_tensor=None,
    cos_tensor=None,
    position_ids=None,
    use_neox_rotary_style=True,
):
    # permute q, k, v from [batch_size, seq_len, num_heads, head_dim]
    # to [batch_size, num_heads, seq_len, head_dim]
    q, k, v = transpose_qkv(init_q, init_k, init_v)

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
        query = mult_qkv_rotate_every_two(q, cos_tensor, sin_tensor)
        value = mult_qkv_rotate_every_two(v, cos_tensor, sin_tensor)
        key = mult_qkv_rotate_every_two(k, cos_tensor, sin_tensor)
    else:
        query = mult_qkv_rotate_half(q, cos_tensor, sin_tensor)
        value = mult_qkv_rotate_half(v, cos_tensor, sin_tensor)
        key = mult_qkv_rotate_half(k, cos_tensor, sin_tensor)

    # permute the result back to [batch_size, seq_len, num_heads, head_dim]
    r_query, r_key, r_value = transpose_qkv(query, key, value)
    return r_query, r_key, r_value


class XPUTestFusedRotaryPositionEmbedding(unittest.TestCase):
    def setUp(self):
        self.init_case()
        self.seed = 1234
        self.init_threshold()

    def init_case(self):
        self.shape_q = [2, 8, 2, 128]
        self.shape_k = [2, 8, 2, 128]
        self.shape_v = [2, 8, 2, 128]
        self.dtype = 'float32'

    def get_paddle_tensor(self, shape):
        if shape is None:
            return None

        tmp = paddle.uniform(shape, self.dtype, -1.0, 1.0)
        tmp.stop_gradient = False
        return tmp

    def init_threshold(self):
        if self.dtype == 'bfloat16':
            self.atol = 1e-2
            self.rtol = 1e-2
        elif self.dtype == 'float16':
            self.atol = 1e-3
            self.rtol = 1e-3
        else:
            self.atol = 1e-4
            self.rtol = 1e-4

    def get_inputs(self, seed, with_sin_cos, dtype="float32"):
        paddle.disable_static()
        paddle.seed(seed)
        tensor_q = self.get_paddle_tensor(self.shape_q)
        tensor_k = self.get_paddle_tensor(self.shape_k)
        tensor_v = self.get_paddle_tensor(self.shape_v)

        tensor_sin, tensor_cos = (
            get_sin_cos_tensor(tensor_q.shape[1], tensor_q.shape[3], dtype)
            if with_sin_cos
            else (None, None)
        )
        return tensor_q, tensor_k, tensor_v, tensor_sin, tensor_cos

    def get_forward_backward(
        self,
        rope_function,
        seed,
        with_sin_cos=True,
        use_neox_rotary_style=True,
        position_ids=None,
    ):
        paddle.disable_static()
        fw = []
        bw = []

        tensor_q, tensor_k, tensor_v, tensor_sin, tensor_cos = self.get_inputs(
            seed, with_sin_cos, self.dtype
        )

        out_q, out_k, out_v = rope_function(
            tensor_q,
            tensor_k,
            tensor_v,
            tensor_sin,
            tensor_cos,
            position_ids=position_ids,
            use_neox_rotary_style=use_neox_rotary_style,
        )
        fw.append(out_q)
        fw.append(out_k)
        fw.append(out_v)
        paddle.seed(seed + 1)
        out_gq = paddle.uniform(out_q.shape, self.dtype, -1.0, 1.0)
        out_gk = paddle.uniform(out_k.shape, self.dtype, -1.0, 1.0)
        out_gv = paddle.uniform(out_v.shape, self.dtype, -1.0, 1.0)

        paddle.autograd.backward(
            [out_q, out_k, out_v], [out_gq, out_gk, out_gv], True
        )
        bw.append(tensor_q.grad)
        bw.append(tensor_k.grad)
        bw.append(tensor_v.grad)

        return fw, bw

    def check_forward_backward(
        self, ref_fwd, fused_fwd, ref_bwd=None, fused_bwd=None
    ):
        for i in range(len(ref_fwd)):
            ref_fwd_np = ref_fwd[i].numpy()
            fused_fwd_np = fused_fwd[i].numpy()
            if ref_bwd is not None:
                ref_bwd_np = ref_bwd[i].numpy()
                fused_bwd_np = fused_bwd[i].numpy()
            if self.dtype == "bfloat16":
                ref_fwd_np = convert_uint16_to_float(ref_fwd_np)
                fused_fwd_np = convert_uint16_to_float(fused_fwd_np)
                if ref_bwd is not None:
                    ref_bwd_np = convert_uint16_to_float(ref_bwd_np)
                    fused_bwd_np = convert_uint16_to_float(fused_bwd_np)
            np.testing.assert_allclose(
                ref_fwd_np, fused_fwd_np, rtol=self.rtol, atol=self.atol
            )
            if ref_bwd is not None:
                np.testing.assert_allclose(
                    ref_bwd_np, fused_bwd_np, rtol=self.rtol, atol=self.atol
                )

    def test_fused_rope(self):
        paddle.set_device('xpu')
        p_fw, p_bw = self.get_forward_backward(
            ref_rotary_position_embedding,
            seed=self.seed,
            use_neox_rotary_style=True,
        )
        f_fw, f_bw = self.get_forward_backward(
            fused_rotary_position_embedding,
            seed=self.seed,
            use_neox_rotary_style=True,
        )
        self.check_forward_backward(p_fw, f_fw, p_bw, f_bw)

    def test_fused_rope_without_sin_cos(self):
        paddle.set_device('xpu')
        p_fw, p_bw = self.get_forward_backward(
            ref_rotary_position_embedding,
            seed=self.seed,
            with_sin_cos=True,
            use_neox_rotary_style=True,
        )
        f_fw, f_bw = self.get_forward_backward(
            fused_rotary_position_embedding,
            seed=self.seed,
            with_sin_cos=False,
            use_neox_rotary_style=True,
        )
        self.check_forward_backward(p_fw, f_fw, p_bw, f_bw)

    def test_fused_rope_rotate_half(self):
        paddle.set_device('xpu')
        p_fw, p_bw = self.get_forward_backward(
            ref_rotary_position_embedding,
            seed=self.seed,
            use_neox_rotary_style=False,
        )
        f_fw, f_bw = self.get_forward_backward(
            fused_rotary_position_embedding,
            seed=self.seed,
            use_neox_rotary_style=False,
        )
        self.check_forward_backward(p_fw, f_fw, p_bw, f_bw)

    def test_fused_rope_position_ids(self):
        paddle.set_device('xpu')
        position_ids = paddle.to_tensor(
            [[7, 5, 4, 6, 3, 1, 2, 0], [3, 1, 4, 0, 7, 6, 5, 2]]
        )
        p_fw, p_bw = self.get_forward_backward(
            ref_rotary_position_embedding,
            seed=self.seed,
            position_ids=position_ids,
            use_neox_rotary_style=False,
        )
        f_fw, f_bw = self.get_forward_backward(
            fused_rotary_position_embedding,
            seed=self.seed,
            position_ids=position_ids,
            use_neox_rotary_style=False,
        )
        self.check_forward_backward(p_fw, f_fw, p_bw, f_bw)

    def test_static(self):
        paddle.set_device('xpu')
        tensor_q, tensor_k, tensor_v, tensor_sin, tensor_cos = self.get_inputs(
            self.seed, True, self.dtype
        )
        p_fw, p_bw = self.get_forward_backward(
            ref_rotary_position_embedding,
            seed=self.seed,
            use_neox_rotary_style=True,
        )

        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            q = paddle.static.data(
                name="q", shape=self.shape_q, dtype=self.dtype
            )
            k = paddle.static.data(
                name="k", shape=self.shape_k, dtype=self.dtype
            )
            v = paddle.static.data(
                name="v", shape=self.shape_v, dtype=self.dtype
            )
            sin = paddle.static.data(
                name="sin",
                shape=(1, tensor_q.shape[1], 1, tensor_q.shape[3]),
                dtype=self.dtype,
            )
            cos = paddle.static.data(
                name="cos",
                shape=(1, tensor_q.shape[1], 1, tensor_q.shape[3]),
                dtype=self.dtype,
            )

            out_q, out_k, out_v = fused_rotary_position_embedding(
                q,
                k,
                v,
                sin,
                cos,
                position_ids=None,
                use_neox_rotary_style=True,
            )

            exe = paddle.static.Executor()

            feed = {
                'q': tensor_q.numpy(),
                'k': tensor_k.numpy(),
                'v': tensor_v.numpy(),
                'sin': tensor_sin.numpy(),
                'cos': tensor_cos.numpy(),
            }
            outs = exe.run(
                paddle.static.default_main_program(),
                feed=feed,
                fetch_list=[out_q, out_k, out_v],
            )
            for i in range(3):
                ref_fwd_np = p_fw[i].numpy()
                fused_fwd_np = outs[i]
                if self.dtype == "bfloat16":
                    ref_fwd_np = convert_uint16_to_float(ref_fwd_np)
                    fused_fwd_np = convert_uint16_to_float(fused_fwd_np)
                np.testing.assert_allclose(
                    ref_fwd_np, fused_fwd_np, rtol=self.rtol, atol=self.atol
                )

        paddle.disable_static()


class XPUTestFusedRotaryPositionEmbeddingFp16_1(
    XPUTestFusedRotaryPositionEmbedding
):
    def init_case(self):
        self.shape_q = [2, 8, 2, 16]
        self.shape_k = [2, 8, 2, 16]
        self.shape_v = [2, 8, 2, 16]
        self.dtype = "float16"


class XPUTestFusedRotaryPositionEmbeddingBf16_1(
    XPUTestFusedRotaryPositionEmbedding
):
    def init_case(self):
        self.shape_q = [2, 8, 2, 16]
        self.shape_k = [2, 8, 2, 16]
        self.shape_v = [2, 8, 2, 16]
        self.dtype = "bfloat16"


class XPUTestFusedRotaryPositionEmbeddingBf16_2(unittest.TestCase):
    def setUp(self):
        self.shape_q = [2, 2048, 16, 128]
        self.shape_k = [2, 2048, 16, 128]
        self.shape_v = [2, 2048, 16, 128]

    def test_api(self):
        paddle.disable_static()
        q_bf16 = paddle.uniform(self.shape_q, "bfloat16", -1.0, 1.0)
        k_bf16 = paddle.uniform(self.shape_k, "bfloat16", -1.0, 1.0)
        v_bf16 = paddle.uniform(self.shape_v, "bfloat16", -1.0, 1.0)
        sin_bf16 = paddle.uniform(
            [1, self.shape_q[1], 1, self.shape_q[3]], "bfloat16", -1.0, 1.0
        )
        cos_bf16 = paddle.uniform(
            [1, self.shape_q[1], 1, self.shape_q[3]], "bfloat16", -1.0, 1.0
        )
        q_bf16.stop_gradient = False
        k_bf16.stop_gradient = False
        v_bf16.stop_gradient = False
        q_fp32 = paddle.to_tensor(q_bf16, dtype="float32", stop_gradient=False)
        k_fp32 = paddle.to_tensor(k_bf16, dtype="float32", stop_gradient=False)
        v_fp32 = paddle.to_tensor(v_bf16, dtype="float32", stop_gradient=False)
        sin_fp32 = paddle.to_tensor(sin_bf16, dtype="float32")
        cos_fp32 = paddle.to_tensor(cos_bf16, dtype="float32")

        position_ids = paddle.arange(0, self.shape_q[1], dtype="int64")
        position_ids = paddle.stack(
            [position_ids for _ in range(self.shape_q[0])], axis=0
        )
        out_bf16 = fused_rotary_position_embedding(
            q_bf16,
            k_bf16,
            v_bf16,
            sin_bf16,
            cos_bf16,
            position_ids=position_ids,
            use_neox_rotary_style=True,
        )

        grad_out_q_bf16 = paddle.uniform(self.shape_q, "bfloat16", -1.0, 1.0)
        grad_out_k_bf16 = paddle.uniform(self.shape_k, "bfloat16", -1.0, 1.0)
        grad_out_v_bf16 = paddle.uniform(self.shape_v, "bfloat16", -1.0, 1.0)

        paddle.autograd.backward(
            out_bf16, [grad_out_q_bf16, grad_out_k_bf16, grad_out_v_bf16], True
        )
        grad_bf16 = [q_bf16.grad, k_bf16.grad, v_bf16.grad]

        out_fp32 = ref_rotary_position_embedding(
            q_fp32,
            k_fp32,
            v_fp32,
            sin_fp32,
            cos_fp32,
            position_ids=position_ids,
            use_neox_rotary_style=True,
        )

        grad_out_q_fp32 = paddle.to_tensor(grad_out_q_bf16, dtype="float32")
        grad_out_k_fp32 = paddle.to_tensor(grad_out_k_bf16, dtype="float32")
        grad_out_v_fp32 = paddle.to_tensor(grad_out_v_bf16, dtype="float32")
        paddle.autograd.backward(
            out_fp32, [grad_out_q_fp32, grad_out_k_fp32, grad_out_v_fp32], True
        )
        grad_fp32 = [q_fp32.grad, k_fp32.grad, v_fp32.grad]

        for fp32_val, bf16_val in zip(out_fp32, out_bf16):
            bf16_val = convert_uint16_to_float(bf16_val.numpy())
            np.testing.assert_allclose(
                fp32_val.numpy(), bf16_val, rtol=1e-2, atol=1e-2
            )
        for grad_fp32_val, grad_bf16_val in zip(grad_fp32, grad_bf16):
            grad_bf16_val = convert_uint16_to_float(grad_bf16_val.numpy())
            np.testing.assert_allclose(
                grad_fp32_val.numpy(), grad_bf16_val, rtol=1e-2, atol=1e-2
            )


class XPUTestFusedRotaryPositionEmbeddingGQA(
    XPUTestFusedRotaryPositionEmbedding
):
    def init_case(self):
        self.shape_q = [2, 8, 2, 16]
        self.shape_k = [2, 8, 1, 16]
        self.shape_v = [2, 8, 1, 16]
        self.dtype = "float32"


# too long for CI
# class XPUTestFusedRotaryPositionEmbeddingBf16_3(XPUTestFusedRotaryPositionEmbeddingBf16_1):
#     def setUp(self):
#         self.shape = [2, 8192, 8, 128]


if __name__ == '__main__':
    unittest.main()
