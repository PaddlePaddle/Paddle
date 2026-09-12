# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


class TestQkvAttentionXpuFusePattern(PassTest):
    r"""
       qkv_input  (B, N, 3*H*D)
           |
         reshape  -> (B, N, 3, H, D)
           |
       transpose perm=[2,0,3,1,4]  -> (3, B, H, N, D)
           |
      +----+----+
      |    |    |
    slice slice slice  (axes=[0])     -> Q/K/V each (B, H, N, D)
      |    |    |
    scale  T(0,1,3,2)
      |    |
       matmul -> softmax -> matmul (V)
                               |
                      transpose [0,2,1,3] -> reshape -> out
    """

    # static dims used in the program
    B = 2
    N = 4
    H = 8
    D = 16
    ALPHA = 1.0 / (D**0.5)

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        B, N, H, D = self.B, self.N, self.H, self.D
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                qkv = paddle.static.data(
                    name='qkv', shape=[B, N, 3 * H * D], dtype='float32'
                )
                qkv_5d = paddle.reshape(qkv, [B, N, 3, H, D])
                qkv_t = paddle.transpose(qkv_5d, perm=[2, 0, 3, 1, 4])
                # Indexing on axis 0 lowers to pd_op.slice with
                # decrease_axis=[0], producing 4D Q / K / V (B, H, N, D)
                # which is exactly what the fusion pass expects.
                q = qkv_t[0]
                k = qkv_t[1]
                v = qkv_t[2]
                q_scaled = paddle.scale(q, scale=self.ALPHA, bias=0.0)
                k_t = paddle.transpose(k, perm=[0, 1, 3, 2])
                qk = paddle.matmul(q_scaled, k_t)
                attn = paddle.nn.functional.softmax(qk, axis=-1)
                attn_v = paddle.matmul(attn, v)
                attn_t = paddle.transpose(attn_v, perm=[0, 2, 1, 3])
                out = paddle.reshape(attn_t, [B, N, H * D])
                out = paddle.assign(out)
                self.feeds = {
                    "qkv": np.random.random((B, N, 3 * H * D)).astype(
                        "float32"
                    ),
                }
                self.fetch_list = [out]
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def test_check_output(self):
        # qkv_attention_xpu uses fp16 accumulation on XPU; relax tolerance.
        self.check_pass_correct(atol=1e-2, rtol=1e-2)

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.device.XPUPlace(0))
            self.pass_attr_list = [{'qkv_attention_xpu_fuse_pass': {}}]
            self.valid_op_map = {
                "pd_op.qkv_attention_xpu": 1,
                "pd_op.softmax": 0,
                "pd_op.matmul": 0,
            }


if __name__ == "__main__":
    unittest.main()
