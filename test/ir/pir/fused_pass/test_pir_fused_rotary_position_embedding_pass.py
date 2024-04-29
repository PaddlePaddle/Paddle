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

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x

    def sample_program(self):
        for q_shape in [[1, 1, 32, 128]]:
            for k_shape in [[1, 1, 32, 128]]:
                for cos_shape in [[1, 1, 1, 128]]:
                    for sin_shape in [[1, 1, 1, 128]]:
                        for position_ids_shape in [[1, 1]]:
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
                                        TestFusedRotaryPositionEmbeddingPass.rotate_half(
                                            q
                                        )
                                        * sin
                                    )
                                    k_embed = (k * cos) + (
                                        TestFusedRotaryPositionEmbeddingPass.rotate_half(
                                            k
                                        )
                                        * sin
                                    )
                                    print("q_embed", q_embed)
                                    print("k_embed", k_embed)
                                    self.pass_list = [
                                        'fused_rotary_position_embedding_pass'
                                    ]
                                    # print("paddle.get_default_dtype()",paddle.get_default_dtype())
                                    self.feeds = {
                                        "q": np.random.random(q_shape).astype(
                                            'float16'
                                        ),
                                        "k": np.random.random(k_shape).astype(
                                            'float16'
                                        ),
                                        "cos": np.random.random(
                                            cos_shape
                                        ).astype('float16'),
                                        "sin": np.random.random(
                                            sin_shape
                                        ).astype('float16'),
                                        "position_ids": np.random.random(
                                            position_ids_shape
                                        ).astype('int64'),
                                    }
                                    self.fetch_list = [q_embed, k_embed]
                                    self.valid_op_map = {
                                        "pd_op.squeeze": 0,
                                        "pd_op.unsqueeze": 0,
                                        "pd_op.concat": 0,
                                        "pd_op.full": 0,
                                        "pd_op.full_int_array": 0,
                                        "pd_op.add": 0,
                                        "pd_op.slice": 0,
                                        "pd_op.scale": 0,
                                        "pd_op.multiply": 0,
                                        "builtin.combine": 0,
                                        "pd_op.gather_nd": 0,
                                        "pd_op.fused_rotary_position_embedding": 1,
                                    }
                                    yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
