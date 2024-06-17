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


class TestDgcMomemtumOpTranslator(test_op_translator.TestOpTranslator):
    def append_op(self):
        self.op_type = "dgc_momentum"

        grad = paddle.ones(shape=(123, 321), dtype='float32')
        param = paddle.ones(shape=(123, 321), dtype='float32')
        velocity = paddle.zeros(shape=(123, 321), dtype='float32')
        learning_rate = paddle.to_tensor([0.001], dtype='float32')
        current_step = paddle.to_tensor([1], dtype='float32')
        nranks = paddle.to_tensor([1, 1], dtype='float32')

        param_out = paddle.ones(shape=(123, 321), dtype='float32')
        velocity_out = paddle.ones(shape=(123, 321), dtype='float32')
        grad_out = paddle.ones(shape=(123, 321), dtype='float32')

        attrs = {
            'mu': 0.0001,
            'use_nesterov': False,
            'rampup_begin_step': 10.0,
        }
        helper = LayerHelper(self.op_type)
        helper.append_op(
            type=self.op_type,
            inputs={
                "Param": param,
                "Grad": grad,
                "Velocity": velocity,
                "LearningRate": learning_rate,
                "current_step": current_step,
                "nranks": nranks,
            },
            outputs={
                "ParamOut": param_out,
                "VelocityOut": velocity_out,
                "Grad_out": grad_out,
            },
            attrs=attrs,
        )

    def test_translator(self):
        self.check()


if __name__ == "__main__":
    unittest.main()
