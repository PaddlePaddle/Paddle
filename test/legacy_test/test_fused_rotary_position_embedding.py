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
import parameterized as param

import paddle
from paddle.base import core
from paddle.incubate.nn.functional import fused_rotary_position_embedding
from paddle.pir_utils import test_with_pir_api

position_ids_list = [[7, 5, 4, 6, 3, 1, 2, 0], [3, 1, 4, 0, 7, 6, 5, 2]]


def mult_qkv(value, cos_tensor, sin_tensor):
    if value is None:
        return None

    rotate_half_q = paddle.reshape(
        paddle.stack([-value[:, :, :, 1::2], value[:, :, :, 0::2]], axis=-1),
        paddle.shape(value),
    )
    query = paddle.add(
        paddle.multiply(value, cos_tensor),
        paddle.multiply(rotate_half_q, sin_tensor),
    )
    return query


def mult_qkv_rotate_half(value, cos_tensor, sin_tensor):
    if value is None:
        return None

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


def get_sin_cos_tensor(seq_len, head_dim, sign=1):
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
    q,
    k,
    v,
    sin_tensor=None,
    cos_tensor=None,
    position_ids=None,
    use_neox_rotary_style=True,
    **kwargs
):
    if position_ids is not None:
        sin_tensor = sin_tensor.squeeze()  # [seq_len, dim]
        cos_tensor = cos_tensor.squeeze()  # [seq_len, dim]
        if len(sin_tensor.shape) == 2 and len(cos_tensor.shape) == 2:
            sin_tensor = sin_tensor[position_ids]
            cos_tensor = cos_tensor[position_ids]
        else:
            sin_tensor = paddle.stack(
                [sin_tensor[i, j] for i, j in enumerate(position_ids)], axis=0
            )
            cos_tensor = paddle.stack(
                [cos_tensor[i, j] for i, j in enumerate(position_ids)], axis=0
            )

        sin_tensor = sin_tensor.unsqueeze(2)  # [bs, seq_len, 1, dim]
        cos_tensor = cos_tensor.unsqueeze(2)  # [bs, seq_len, 1, dim]

    if use_neox_rotary_style:
        query = mult_qkv(q, cos_tensor, sin_tensor)
        value = mult_qkv(v, cos_tensor, sin_tensor)
        key = mult_qkv(k, cos_tensor, sin_tensor)
    else:
        query = mult_qkv_rotate_half(q, cos_tensor, sin_tensor)
        value = mult_qkv_rotate_half(v, cos_tensor, sin_tensor)
        key = mult_qkv_rotate_half(k, cos_tensor, sin_tensor)

    return query, key, value


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA ",
)
@param.parameterized_class(
    (
        "name",
        'shape_q',
        'shape_k',
        'shape_v',
        'position_ids_list',
        'shape_sin',
        'return_random_sin_cos',
    ),
    [
        (
            "qkv_input",
            [2, 8, 2, 16],  # bs, seq_len, num_heads, head_dim
            [2, 8, 2, 16],  # bs, seq_len, num_heads, head_dim
            [2, 8, 2, 16],  # bs, seq_len, num_heads, head_dim
            position_ids_list,
        ),
        ("qk_input", [2, 8, 2, 16], [2, 8, 2, 16], None, position_ids_list),
        ("qv_input", [2, 8, 2, 16], None, [2, 8, 2, 16], position_ids_list),
        ("q_input", [2, 8, 2, 16], None, None, position_ids_list),
        (
            "q_input",
            [2, 8, 2, 16],
            None,
            None,
            position_ids_list,
            [2, 8, 1, 16],
            True,
        ),
        (
            "qkv_input_mqa",
            [2, 8, 4, 8],
            [2, 8, 1, 8],
            [2, 8, 1, 8],
            position_ids_list,
        ),
        ("qk_input_mqa", [2, 8, 4, 8], [2, 8, 1, 8], None, position_ids_list),
        ("qv_input_mqa", [2, 8, 4, 8], None, [2, 8, 1, 8], position_ids_list),
        (
            "qv_input_mqa",
            [2, 8, 4, 8],
            None,
            [2, 8, 1, 8],
            position_ids_list,
            [2, 8, 1, 8],
            True,
        ),
        (
            "qkv_input_gqa",
            [1, 8, 4, 8],
            [1, 8, 2, 8],
            [1, 8, 2, 8],
            position_ids_list[:1],
        ),
        (
            "qk_input_gqa",
            [1, 8, 4, 8],
            [1, 8, 2, 8],
            None,
            position_ids_list[:1],
        ),
        (
            "qv_input_gqa",
            [1, 8, 4, 8],
            None,
            [1, 8, 2, 8],
            position_ids_list[:1],
        ),
        (
            "qv_input_gqa",
            [2, 8, 4, 8],
            None,
            [2, 8, 2, 8],
            position_ids_list,
            [2, 8, 1, 8],
            True,
        ),
    ],
)
class TestFusedRotaryPositionEmbedding(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float32'
        self.training = True
        self.seed = 1203
        self.rtol = 1e-5

    def get_paddle_tensor(self, shape):
        if shape is None:
            return None

        tmp = paddle.randn(shape, self.dtype)
        tmp.stop_gradient = False
        return tmp

    def get_inputs(self, seed, with_sin_cos, return_random_sin_cos=False):
        paddle.disable_static()
        paddle.seed(seed)
        # tensor_q shape: [batch_size, seq_len, num_heads, head_dim]
        tensor_q = self.get_paddle_tensor(self.shape_q)
        tensor_k = self.get_paddle_tensor(self.shape_k)
        tensor_v = self.get_paddle_tensor(self.shape_v)

        if return_random_sin_cos and with_sin_cos:
            tensor_sin = paddle.uniform(self.shape_sin, self.dtype, -1, 1.0)
            tensor_cos = paddle.uniform(self.shape_sin, self.dtype, -1, 1.0)
        else:
            tensor_sin, tensor_cos = (
                get_sin_cos_tensor(tensor_q.shape[1], tensor_q.shape[3], 1)
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
        test_time_major=False,
    ):
        paddle.disable_static()
        fw = []
        bw = []

        tensor_q, tensor_k, tensor_v, tensor_sin, tensor_cos = self.get_inputs(
            seed,
            with_sin_cos,
            return_random_sin_cos=(
                self.return_random_sin_cos
                if hasattr(self, 'return_random_sin_cos')
                else False
            ),
        )

        if test_time_major:
            # [batch_size, seq_len, num_heads, head_dim] -> [seq_len, batch_size, num_heads, head_dim]
            if tensor_q is not None:
                tensor_q = paddle.transpose(tensor_q, perm=[1, 0])
            if tensor_k is not None:
                tensor_k = paddle.transpose(tensor_k, perm=[1, 0])
            if tensor_v is not None:
                tensor_v = paddle.transpose(tensor_v, perm=[1, 0])

        out_q, out_k, out_v = rope_function(
            tensor_q,
            tensor_k,
            tensor_v,
            tensor_sin,
            tensor_cos,
            position_ids=position_ids,
            use_neox_rotary_style=use_neox_rotary_style,
            time_major=test_time_major,
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

        if test_time_major:
            # transpose back
            # [seq_len, batch_size, num_heads, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
            fw = [paddle.transpose(x, perm=[1, 0]) for x in fw]
            bw = [paddle.transpose(x, perm=[1, 0]) for x in bw]

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
            paddle_fused_rotary_position_embedding, seed=self.seed
        )
        f_fw, f_bw = self.get_forward_backward(
            fused_rotary_position_embedding,
            seed=self.seed,
            test_time_major=False,
        )
        f_fw_time_major, f_bw_time_major = self.get_forward_backward(
            fused_rotary_position_embedding,
            seed=self.seed,
            test_time_major=True,
        )

        self.check_results(p_fw, f_fw)
        self.check_results(p_bw, f_bw)
        self.check_results(p_fw, f_fw_time_major)
        self.check_results(p_bw, f_bw_time_major)

    def test_fused_rope_with_sin_cos(self):
        p_fw, p_bw = self.get_forward_backward(
            paddle_fused_rotary_position_embedding,
            seed=self.seed,
            with_sin_cos=True,
        )
        f_fw, f_bw = self.get_forward_backward(
            fused_rotary_position_embedding,
            seed=self.seed,
            with_sin_cos=True,
            test_time_major=False,
        )
        f_fw_time_major, f_bw_time_major = self.get_forward_backward(
            fused_rotary_position_embedding,
            seed=self.seed,
            with_sin_cos=True,
            test_time_major=True,
        )

        self.check_results(p_fw, f_fw)
        self.check_results(p_bw, f_bw)
        self.check_results(p_fw, f_fw_time_major)
        self.check_results(p_bw, f_bw_time_major)

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
            test_time_major=False,
        )
        f_fw_time_major, f_bw_time_major = self.get_forward_backward(
            fused_rotary_position_embedding,
            seed=self.seed,
            use_neox_rotary_style=False,
            test_time_major=True,
        )

        self.check_results(p_fw, f_fw)
        self.check_results(p_bw, f_bw)
        self.check_results(p_fw, f_fw_time_major)
        self.check_results(p_bw, f_bw_time_major)

    def test_fused_rope_position_ids(self):
        position_ids = paddle.to_tensor(self.position_ids_list)
        p_fw, p_bw = self.get_forward_backward(
            paddle_fused_rotary_position_embedding,
            seed=self.seed,
            position_ids=position_ids,
        )
        f_fw, f_bw = self.get_forward_backward(
            fused_rotary_position_embedding,
            seed=self.seed,
            position_ids=position_ids,
            test_time_major=False,
        )
        f_fw_time_major, f_bw_time_major = self.get_forward_backward(
            fused_rotary_position_embedding,
            seed=self.seed,
            position_ids=position_ids,
            test_time_major=True,
        )

        self.check_results(p_fw, f_fw)
        self.check_results(p_bw, f_bw)
        self.check_results(p_fw, f_fw_time_major)
        self.check_results(p_bw, f_bw_time_major)

    @test_with_pir_api
    def test_static(self):
        paddle.disable_static()
        tensor_q, tensor_k, tensor_v, tensor_sin, tensor_cos = self.get_inputs(
            self.seed,
            True,
            return_random_sin_cos=(
                self.return_random_sin_cos
                if hasattr(self, 'return_random_sin_cos')
                else False
            ),
        )
        p_fw, p_bw = self.get_forward_backward(
            paddle_fused_rotary_position_embedding,
            seed=self.seed,
            use_neox_rotary_style=False,
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
                shape=tensor_sin.shape,
                dtype=self.dtype,
            )
            cos = paddle.static.data(
                name="cos",
                shape=tensor_cos.shape,
                dtype=self.dtype,
            )

            out_q, out_k, out_v = fused_rotary_position_embedding(
                q,
                k,
                v,
                sin,
                cos,
                position_ids=None,
                use_neox_rotary_style=False,
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

    @test_with_pir_api
    def test_static_time_major(self):
        paddle.disable_static()
        tensor_q, tensor_k, tensor_v, tensor_sin, tensor_cos = self.get_inputs(
            self.seed,
            True,
            return_random_sin_cos=(
                self.return_random_sin_cos
                if hasattr(self, 'return_random_sin_cos')
                else False
            ),
        )
        p_fw, p_bw = self.get_forward_backward(
            paddle_fused_rotary_position_embedding,
            seed=self.seed,
            use_neox_rotary_style=False,
            test_time_major=False,
        )

        paddle.enable_static()

        shape_q = (
            [self.shape_q[1], self.shape_q[0], self.shape_q[2], self.shape_q[3]]
            if self.shape_q
            else None
        )
        shape_k = (
            [self.shape_k[1], self.shape_k[0], self.shape_k[2], self.shape_k[3]]
            if self.shape_k
            else None
        )
        shape_v = (
            [self.shape_v[1], self.shape_v[0], self.shape_v[2], self.shape_v[3]]
            if self.shape_v
            else None
        )

        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            q = (
                None
                if shape_q is None
                else paddle.static.data(
                    name="q", shape=shape_q, dtype=self.dtype
                )
            )

            k = (
                None
                if shape_k is None
                else paddle.static.data(
                    name="k", shape=shape_k, dtype=self.dtype
                )
            )

            v = (
                None
                if shape_v is None
                else paddle.static.data(
                    name="v", shape=shape_v, dtype=self.dtype
                )
            )

            sin = paddle.static.data(
                name="sin",
                shape=tensor_sin.shape,
                dtype=self.dtype,
            )
            cos = paddle.static.data(
                name="cos",
                shape=tensor_cos.shape,
                dtype=self.dtype,
            )

            out_q, out_k, out_v = fused_rotary_position_embedding(
                q,
                k,
                v,
                sin,
                cos,
                position_ids=None,
                use_neox_rotary_style=False,
                time_major=True,
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
                feed[var_name] = input_tensor.numpy().transpose((1, 0, 2, 3))

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
                outs[i].transpose((1, 0, 2, 3)),
                rtol=self.rtol,
            )
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
