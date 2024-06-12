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

import test_op_translator

import paddle
from paddle.base.layer_helper import LayerHelper

paddle.pir_utils._switch_to_old_ir_()


class TestDgcOpTranslator(test_op_translator.TestOpTranslator):
    def append_op(self):
        self.op_type = "dgc"
        g_array_size = 102400
        u = paddle.ones(shape=(g_array_size,), dtype='float32')
        v = paddle.ones(shape=(g_array_size,), dtype='float32')
        grad = paddle.ones(shape=(g_array_size,), dtype='float32')
        param = paddle.ones(shape=(g_array_size,), dtype='float32')
        current_step = paddle.to_tensor([0.0], dtype='float32')
        nranks = paddle.to_tensor([2.0], dtype='float32')

        u_out = paddle.ones(shape=(g_array_size,), dtype='float32')
        v_out = paddle.ones(shape=(g_array_size,), dtype='float32')
        encode_grad = paddle.ones(shape=(g_array_size,), dtype='float32')
        grad_out = paddle.ones(shape=(g_array_size,), dtype='float32')
        k = paddle.to_tensor([0.0], dtype='float32')
        gather_buff = paddle.ones(shape=(g_array_size,), dtype='float32')
        attrs = {
            'm': 0.9,
            'use_nesterov': True,
            'sparsity': [],
            'padding_idx': -1,
            'rampup_begin_step': 0.0,
            'rampup_step': 0.0,
            'regular_coeff': 0.0,
            'regular_type': 0,
        }
        helper = LayerHelper(self.op_type)
        helper.append_op(
            type=self.op_type,
            inputs={
                "U": u,
                "V": v,
                "Grad": grad,
                "Param": param,
                "current_step": current_step,
                "nranks": nranks,
            },
            outputs={
                "U_out": u_out,
                "V_out": v_out,
                "EncodeGrad": encode_grad,
                "Grad_out": grad_out,
                "k": k,
                "GatherBuff": gather_buff,
            },
            attrs=attrs,
        )

    def test_translator(self):
        self.check()


if __name__ == "__main__":
    unittest.main()
