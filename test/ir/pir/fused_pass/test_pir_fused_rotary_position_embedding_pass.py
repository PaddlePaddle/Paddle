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

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        x3 = paddle.concat([-x2, x1], axis=-1)
        return x3

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
                                        name="q",
                                        shape=q_shape,
                                        dtype='float32',
                                    )
                                    k = paddle.static.data(
                                        name="k",
                                        shape=k_shape,
                                        dtype='float32',
                                    )
                                    cos = paddle.static.data(
                                        name="cos",
                                        shape=cos_shape,
                                        dtype='float32',
                                    )
                                    sin = paddle.static.data(
                                        name="sin",
                                        shape=sin_shape,
                                        dtype='float32',
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

                                    out1 = paddle.assign(q_embed)
                                    out2 = paddle.assign(k_embed)

                                    position_ids_values = np.array(
                                        [
                                            [7, 5, 4, 6, 3, 1, 2, 0],
                                            [3, 1, 4, 0, 7, 6, 5, 2],
                                        ],
                                        dtype='int64',
                                    )

                                    self.pass_attr_list = [
                                        {
                                            'fused_rotary_position_embedding_pass': {}
                                        }
                                    ]

                                    self.feeds = {
                                        "q": np.random.random(q_shape).astype(
                                            "float32"
                                        ),
                                        "k": np.random.random(k_shape).astype(
                                            "float32"
                                        ),
                                        "cos": np.random.random(
                                            cos_shape
                                        ).astype("float32"),
                                        "sin": np.random.random(
                                            sin_shape
                                        ).astype("float32"),
                                        "position_ids": np.broadcast_to(
                                            np.arange(8, dtype='int64'), (2, 8)
                                        ),
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
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
