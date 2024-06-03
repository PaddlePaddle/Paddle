# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.nn.layer.transformer import _convert_attention_mask


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 8
    or paddle.get_cudnn_version() < 8906,
    "cudnn flash attn is only supported after Ampere and need version >= 8906",
)
class TestFusedDotProductAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_len = 1024
        self.num_heads = 12
        self.head_size = 64
        self.default_dtype = "float16"

    def test_fused_dot_product_attention(self):
        paddle.set_default_dtype("float16")
        self.qkv_shape = (
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.head_size,
        )
        self.mask_shape = (self.batch_size, 1, self.seq_len, self.seq_len)
        q_np = np.random.normal(loc=0, scale=0.02, size=self.qkv_shape).astype(
            "float16"
        )
        k_np = np.random.normal(loc=0, scale=0.02, size=self.qkv_shape).astype(
            "float16"
        )
        v_np = np.random.normal(loc=0, scale=0.02, size=self.qkv_shape).astype(
            "float16"
        )
        mask_np = np.ones(self.mask_shape).astype("int32")

        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program):
                q_ = paddle.static.data(
                    name="q", shape=self.qkv_shape, dtype="float16"
                )
                k_ = paddle.static.data(
                    name="k", shape=self.qkv_shape, dtype="float16"
                )
                v_ = paddle.static.data(
                    name="v", shape=self.qkv_shape, dtype="float16"
                )
                mask = paddle.static.data(
                    name="mask", shape=self.mask_shape, dtype="int32"
                )

                q_.stop_gradient = False
                k_.stop_gradient = False
                v_.stop_gradient = False
                mask.stop_gradient = True

                qt = paddle.transpose(q_, [0, 2, 1, 3])
                kt = paddle.transpose(k_, [0, 2, 1, 3])
                vt = paddle.transpose(v_, [0, 2, 1, 3])

                product = paddle.matmul(
                    x=qt * (self.head_size**-0.5), y=kt, transpose_y=True
                )
                attn_mask = _convert_attention_mask(mask, product.dtype)
                product = product + attn_mask
                weights = paddle.nn.functional.softmax(product)
                out = paddle.matmul(weights, vt)
                out = paddle.transpose(out, [0, 2, 1, 3])
                res1 = paddle.reshape(
                    out,
                    [
                        self.batch_size,
                        self.seq_len,
                        self.num_heads * self.head_size,
                    ],
                )

                res2 = paddle.assign(res1)

                res3, res4, res5 = paddle.autograd.ir_backward.grad(
                    res2, [q_, k_, v_]
                )
                res3_ = paddle.assign(res3)
                res4_ = paddle.assign(res4)
                res5_ = paddle.assign(res5)

                op_names = [op.name() for op in main_program.global_block().ops]

                with paddle.static.scope_guard(paddle.static.Scope()):
                    exe = paddle.base.Executor(paddle.base.CUDAPlace(0))
                    fetches0 = exe.run(
                        main_program,
                        feed={"q": q_np, "k": k_np, "v": v_np, "mask": mask_np},
                        fetch_list=[res2, res3_, res4_, res5_],
                    )
                pm = paddle.pir.PassManager()
                pm.add_pass('fused_dot_product_attention_pass', {})
                pm.run(main_program)
                op_names = [op.name() for op in main_program.global_block().ops]

                self.assertTrue('pd_op.fused_dot_product_attention' in op_names)
                self.assertTrue(
                    'pd_op.fused_dot_product_attention_grad' in op_names
                )

                with paddle.static.scope_guard(paddle.static.Scope()):
                    exe = paddle.base.Executor(paddle.base.CUDAPlace(0))
                    fetches1 = exe.run(
                        main_program,
                        feed={"q": q_np, "k": k_np, "v": v_np, "mask": mask_np},
                        fetch_list=[res2, res3_, res4_, res5_],
                    )
        for i in range(len(fetches0)):
            np.testing.assert_allclose(
                fetches0[i], fetches1[i], rtol=1e-3, atol=1e-3
            )


if __name__ == "__main__":
    unittest.main()
