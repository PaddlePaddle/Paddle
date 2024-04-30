# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from pass_test import PassTest

import paddle
from paddle.base import core

paddle.enable_static()

position_ids_list = [[7, 5, 4, 6, 3, 1, 2, 0], [3, 1, 4, 0, 7, 6, 5, 2]]


class TestFusedRotaryPositionEmbeddingPass(PassTest):
    r"""
    k                        k                       cos       position_ids               sin        position_ids                   q             q
   /  \                       \                         |             |                      |              |                      /             / \
slice slice                    \                   squeeze       unsqueeze              squeeze       unsqueeze                   /          slice slice
  |     |                       \                      \             /                      \             /                      /             |     |
  |   scale                      \                        gather_nd                            gather_nd                        /              |   scale
   \     /                        \                           |                                    |                           /               \     /
   concat                          \                      unsqueeze                            unsqueeze                      /                 concat
      \                              \                        / \                                  /\                         /                   /
       \                              \                      /   \                                /  \                       /                   /
        \                              \                    /     \                              /    \                    /                    /
         \                              \                  /       \                            /      \                   /                   /
          \                              \                /         \                          /        \                 /                   /
           \                              \              /           \                        /          \               /                   /
            \                                 multiply                \                      /            \             /                   /
             \                                     \                   \                    /              \           /                   /
              \                                     \                   \                  /                \         /                   /
               \                                     \                   \                /                  \       /                   /
                \                                     \                   \              /                    \     /                   /
                 \                                     \                   \            /                      \   /                   /
                  \                                     \                   \          /                        \ /                   /
                   \                                     \                   \        /                         /\                   /
                    \                                     \                   \      /                         /  \                 /
                     \                                     \                   \    /                         /    \               /
                      \                                     \                   \  /                         /      \             /
                       \                                     \                   \/                         /          multiply
                        \                                     \                  /\                        /              /
                         \                                     \                /  \                      /              /
                          \                                     \              /    \                    /              /
                           \                                     \            /      \                  /              /
                            \                                     \          /        \                /              /
                             \                                     \        /          \              /              /
                              \                                     \      /              multiply                  /
                               \                                     \    /                    \                   /
                                \                                     \  /                      \                 /
                                 \                                     \/                        \               /
                                  \                                    /\                         \             /
                                   \                                  /  \                         \           /
                                    \                                /    \                            add
                                     \                              /      \
                                      \                            /        \
                                       \                          /          \
                                        \                        /            /
                                         \                      /            /
                                          \                    /            /
                                           \                 /             /
                                                multiply                  /
                                                       \                 /
                                                        \               /
                                                         \             /
                                                          \           /
                                                           \         /
                                                               add
  """

    def is_program_valid(self, program=None):
        return True

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x

    def deal_qkv(self, init_value):
        if init_value is None:
            return None
        perm = [0, 2, 1, 3]
        return paddle.transpose(x=init_value, perm=perm)

    def mult_qkv(self, value, cos_tensor, sin_tensor):
        if value is None:
            return None
        rotate_half_q = paddle.reshape(
            paddle.stack(
                [-value[:, :, :, 1::2], value[:, :, :, 0::2]], axis=-1
            ),
            paddle.shape(value),
        )
        query = paddle.add(
            paddle.multiply(value, cos_tensor),
            paddle.multiply(rotate_half_q, sin_tensor),
        )
        return query

    def multi_qkv_rotate_half(self, value, cos_tensor, sin_tensor):
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

    def get_sin_cos_tensor(self, seq_len, head_dim, sign=1):
        pos_seq = np.arange(0, seq_len, 1, dtype=np.float16)
        indices = np.arange(0, head_dim, 2, dtype=np.float16)

        indices = 1 / 10000 ** (indices / head_dim)
        sinusoid_inp = (
            pos_seq[:, np.newaxis] * indices[np.newaxis, :]
        )  # 使用 np.newaxis 增加维度
        sin_sin = np.empty((seq_len * head_dim), dtype=np.float16)
        cos_cos = np.empty((seq_len * head_dim), dtype=np.float16)
        numpy_array = sinusoid_inp
        print("numpy_array:", numpy_array)
        iter_array = np.nditer(numpy_array)

        i = 0

        for value in iter_array:
            sin_sin[i * 2] = sign * np.sin(value)
            cos_cos[i * 2 + 0] = np.cos(value)
            sin_sin[i * 2 + 1] = np.sin(value)
            cos_cos[i * 2 + 1] = np.cos(value)
            i += 1
        tensor_sin = sin_sin.reshape(1, seq_len, 1, head_dim)
        tensor_cos = cos_cos.reshape(1, seq_len, 1, head_dim)

        return tensor_sin, tensor_cos

    def get_numpy_tensor(self, shape, dtype):
        if shape is None:
            return None
        # 使用 numpy 创建随机数组，并指定数据类型
        return np.random.randn(*shape).astype(dtype)

    def sample_program(self):
        for q_shape in [[2, 8, 2, 16]]:
            for k_shape in [[2, 8, 2, 16]]:
                for cos_shape in [[1, 8, 1, 16]]:
                    for sin_shape in [[1, 8, 1, 16]]:
                        for position_ids_shape in [[2, 8]]:
                            with paddle.pir_utils.IrGuard():
                                start_prog = paddle.static.Program()
                                main_prog = paddle.static.Program()
                                with paddle.pir.core.program_guard(
                                    main_prog, start_prog
                                ):
                                    q = paddle.static.data(
                                        name="q", shape=q_shape, dtype='float16'
                                    )
                                    k = paddle.static.data(
                                        name="k", shape=k_shape, dtype='float16'
                                    )
                                    cos = paddle.static.data(
                                        name="cos",
                                        shape=cos_shape,
                                        dtype='float16',
                                    )
                                    sin = paddle.static.data(
                                        name="sin",
                                        shape=sin_shape,
                                        dtype='float16',
                                    )
                                    position_ids = paddle.static.data(
                                        name="position_ids",
                                        shape=position_ids_shape,
                                        dtype='int64',
                                    )

                                    cos = cos.squeeze(axis=[0, 2])
                                    sin = sin.squeeze(axis=[0, 2])
                                    cos = cos[position_ids].unsqueeze(2)
                                    sin = sin[position_ids].unsqueeze(2)

                                    q_embed = (q * cos) + (
                                        self.rotate_half(q) * sin
                                    )
                                    k_embed = (k * cos) + (
                                        self.rotate_half(k) * sin
                                    )
                                    print("q_embed", q_embed)
                                    print("k_embed", k_embed)
                                    out1 = paddle.assign(q_embed)
                                    out2 = paddle.assign(k_embed)

                                    position_ids_values = np.array(
                                        [
                                            [7, 5, 4, 6, 3, 1, 2, 0],
                                            [3, 1, 4, 0, 7, 6, 5, 2],
                                        ],
                                        dtype='int64',
                                    )
                                    print(
                                        "positon_ids_values",
                                        position_ids_values,
                                    )

                                    # print("q_embed", q_embed)
                                    # print("k_embed", k_embed)
                                    self.pass_list = [
                                        'fused_rotary_position_embedding_pass'
                                    ]
                                    tensor_q = self.get_numpy_tensor(
                                        q_shape, 'float16'
                                    )
                                    tensor_k = self.get_numpy_tensor(
                                        k_shape, 'float16'
                                    )
                                    print("tensor_q", tensor_q)
                                    print("tensor_k", tensor_k)
                                    (
                                        tensor_sin,
                                        tensor_cos,
                                    ) = self.get_sin_cos_tensor(
                                        tensor_q.shape[1], tensor_q.shape[3]
                                    )
                                    print("tensor_sin", tensor_sin)
                                    print("tensor_cos", tensor_cos)
                                    position_ids_value = paddle.to_tensor(
                                        position_ids_list, dtype='int64'
                                    )
                                    print(
                                        "已转换 position_ids 为张量:",
                                        position_ids_value,
                                    )
                                    # position_ids_data = np.array([[7, 5, 4, 6, 3, 1, 2, 0], [3, 1, 4, 0, 7, 6, 5, 2]], dtype='int64')

                                    # print("paddle.get_default_dtype()",paddle.get_default_dtype())
                                    self.feeds = {
                                        "q": tensor_q,
                                        "k": tensor_k,
                                        "cos": tensor_cos,
                                        "sin": tensor_sin,
                                        "position_ids": position_ids_values,
                                    }
                                    self.fetch_list = [out1, out2]
                                    self.valid_op_map = {
                                        "pd_op.squeeze": 0,
                                        "pd_op.unsqueeze": 0,
                                        "pd_op.concat": 0,
                                        "pd_op.multiply": 0,
                                        "pd_op.add": 0,
                                        "pd_op.slice": 0,
                                        "pd_op.scale": 0,
                                        "builtin.combine": 0,
                                        "pd_op.gather_nd": 0,
                                        "pd_op.fused_rotary_position_embedding": 1,
                                    }
                                    yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
