# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import random
import numpy as np
import paddle

from paddle.distributed.fleet import auto
from get_gpt_model import generate_model, create_data_holder, FakeDataset

paddle.enable_static()


def apply_pass():
    dist_strategy = auto.Strategy()
    dist_strategy.auto_mode = "semi"
    qat = dist_strategy.qat
    qat.enable = True
    qat.channel_wise_abs_max = True
    qat.weight_bits = 8
    qat.activation_bits = 8
    qat.not_quant_pattern = ['skip_quant']
    return dist_strategy


class TestQuantizationPass(unittest.TestCase):

    def test_qat_pass(self):

        batch_size = 8
        batch_num = 10

        strategy = apply_pass()
        model, loss = generate_model("serial")
        opt = paddle.optimizer.AdamW(learning_rate=0.00001)
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        dataset = FakeDataset(batch_size * batch_num)
        engine.fit(dataset, 3, batch_size=batch_size)

        self.check_program(engine.main_program)

    def check_program(self, program):

        quantizable_op_and_inputs = {'matmul_v2': ['X', 'Y']}
        quantizable_grad_op_inputs = {'matmul_v2_grad': ['X', 'Y']}

        quantized_ops = set()
        for block in program.blocks:
            for op in block.ops:
                is_quntized = False
                if op.type in quantizable_op_and_inputs:
                    for arg_name in op.input_arg_names:
                        if ".quantized" in arg_name:
                            is_quntized = True

                if not is_quntized:
                    continue

                # check forward
                if op.type in quantizable_op_and_inputs:
                    for arg_name in op.input_arg_names:
                        assert arg_name.endswith('.quantized.dequantized')
                        quantized_ops.add(arg_name)

            for op in block.ops:
                is_quntized = False
                if op.type in quantizable_grad_op_inputs:
                    for pname in quantizable_grad_op_inputs[op.type]:
                        arg_name = op.input(pname)[0]
                        if ".quantized" in arg_name:
                            is_quntized = True

                if not is_quntized:
                    continue

                # check backward
                if op.type in quantizable_grad_op_inputs:
                    for pname in quantizable_grad_op_inputs[op.type]:
                        arg_name = op.input(pname)[0]
                        assert arg_name.endswith('.quantized.dequantized')
                        assert arg_name in quantized_ops


if __name__ == "__main__":
    unittest.main()
