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

import os
import tempfile
import unittest

import numpy as np
from get_gpt_model import FakeDataset, create_data_holder, generate_model

import paddle
from paddle.distributed.fleet import auto

paddle.enable_static()


def apply_pass():
    dist_strategy = auto.Strategy()
    dist_strategy.auto_mode = "semi"

    amp = dist_strategy.amp
    amp.enable = True
    amp.dtype = "float16"
    amp.level = "o2"
    amp.custom_white_list = ["lookup_table", "lookup_table_v2"]
    amp.custom_black_list = [
        "reduce_sum",
        "c_softmax_with_cross_entropy",
        "elementwise_div",
    ]
    amp.init_loss_scaling = 32768

    qat = dist_strategy.qat
    qat.enable = True
    qat.channel_wise_abs_max = True
    qat.weight_bits = 8
    qat.activation_bits = 8
    qat.not_quant_pattern = ['skip_quant']
    qat.onnx_format = True
    return dist_strategy


class TestQuantizationPassTrain(unittest.TestCase):
    def test_qat_pass_training(self):
        batch_size = 1
        batch_num = 10

        strategy = apply_pass()
        model, loss = generate_model("mp")
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
            for idx, op in enumerate(block.ops):
                is_quantized = False
                if op.type in quantizable_op_and_inputs:
                    for arg_name in op.input_arg_names:
                        if ".quantized" in arg_name:
                            is_quantized = True

                if not is_quantized:
                    continue

                # check forward
                if op.type in quantizable_op_and_inputs:
                    for arg_name in op.input_arg_names:
                        if "c_identity" in arg_name:
                            arg_name = block.ops[idx - 1].input_arg_names[0]
                        assert arg_name.endswith('.quantized.dequantized')
                        quantized_ops.add(arg_name)

            for op in block.ops:
                is_quantized = False
                if op.type in quantizable_grad_op_inputs:
                    for pname in quantizable_grad_op_inputs[op.type]:
                        arg_name = op.input(pname)[0]
                        if ".quantized" in arg_name:
                            is_quantized = True

                if not is_quantized:
                    continue

                # check backward
                if op.type in quantizable_grad_op_inputs:
                    for pname in quantizable_grad_op_inputs[op.type]:
                        arg_name = op.input(pname)[0]
                        assert arg_name.endswith('.quantized.dequantized')
                        assert arg_name in quantized_ops


class TestQuantizationPassExport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_qat_pass_2(self):
        strategy = apply_pass()
        model, loss = generate_model("mp")
        engine = auto.Engine(model, loss, strategy=strategy)
        inputs_spec, labels_spec = create_data_holder(batch_size=1)
        engine.prepare(inputs_spec, labels_spec, mode="predict")

        path = os.path.join(self.temp_dir.name, 'inf')
        engine.save(path, training=False)
        self.check_export(engine._executor)

    def check_export(self, exe):
        sequence_len = 512
        vocab_size = 1000

        tokens = [np.random.randint(vocab_size, size=sequence_len)]
        position_ids = [np.arange(sequence_len)]
        attention_mask = [np.tril(np.ones(sequence_len))]

        path_prefix = os.path.join(
            self.temp_dir.name,
            f'inf_dist{paddle.distributed.get_rank()}',
        )
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.load_inference_model(
            path_prefix=path_prefix, executor=exe
        )

        out = exe.run(
            inference_program,
            feed={
                "tokens": tokens,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            },
            fetch_list=fetch_targets,
        )


if __name__ == "__main__":
    unittest.main()
