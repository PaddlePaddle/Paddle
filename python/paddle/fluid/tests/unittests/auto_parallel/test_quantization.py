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
import numpy as np
import paddle

import paddle.distributed.fleet as fleet
import paddle.distributed.auto_parallel as auto

from paddle.distributed.auto_parallel.engine import Engine
from paddle.distributed.auto_parallel.utils import print_program_with_dist_attr

sys.path.append("..")
import auto_parallel_gpt_model as modeling
from auto_parallel_gpt_model import GPTModel, GPTForPretraining, GPTPretrainingCriterion

paddle.enable_static()


class FakeDataset:

    def __init__(self, num_samples, sequence_len, vocab_size):
        self.num_samples = num_samples
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size

    def __getitem__(self, idx):
        tokens = np.random.randint(self.vocab_size, size=self.sequence_len)
        position_ids = np.arange(self.sequence_len)
        attention_mask = np.tril(np.ones(self.sequence_len)).reshape(
            (1, self.sequence_len, self.sequence_len)).astype(np.float32)
        labels = np.random.randint(self.vocab_size, size=self.sequence_len)
        loss_mask = np.ones(self.sequence_len).astype(np.float32)
        return tokens, position_ids, attention_mask, labels, loss_mask

    def __len__(self):
        return self.num_samples


def apply_pass():
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    dist_strategy.qat = True
    dist_strategy.qat_configs = {
        'channel_wise_abs_max': True,
        'weight_bits': 8,
        'activation_bits': 8,
        'not_quant_pattern': ['skip_quant'],
    }
    return dist_strategy


def create_data_holder(batch_size, sequence_len):
    tokens = paddle.static.InputSpec(name="tokens",
                                     shape=[batch_size, sequence_len],
                                     dtype='int64')
    position_ids = paddle.static.InputSpec(name="position_ids",
                                           shape=[batch_size, sequence_len],
                                           dtype='int64')
    attention_mask = paddle.static.InputSpec(
        name="attention_mask",
        shape=[batch_size, 1, sequence_len, sequence_len],
        dtype='float32')
    labels = paddle.static.InputSpec(name="labels",
                                     shape=[batch_size, sequence_len],
                                     dtype='int64')
    loss_mask = paddle.static.InputSpec(name="loss_mask",
                                        shape=[batch_size, sequence_len],
                                        dtype='float32')
    return [tokens, position_ids, attention_mask], [labels, loss_mask]


def get_gpt_model():
    modeling.init_global()
    modeling._global_parallel_strategy = "serial"
    modeling._global_process_mesh = auto.ProcessMesh(mesh=[0])

    gpt = GPTModel(vocab_size=1000,
                   hidden_size=64,
                   num_hidden_layers=2,
                   num_attention_heads=8,
                   intermediate_size=256,
                   hidden_act="gelu",
                   hidden_dropout_prob=0.0,
                   attention_probs_dropout_prob=0.0,
                   max_position_embeddings=1024,
                   type_vocab_size=1,
                   initializer_range=0.02,
                   pad_token_id=0,
                   eos_token_id=7,
                   bos_token_id=0,
                   eol_token_id=3)
    model = GPTForPretraining(gpt,
                              vocab_size=1000,
                              hidden_size=64,
                              initializer_range=0.02)
    criterion = GPTPretrainingCriterion()
    return model, criterion


class TestQuantizationPass(unittest.TestCase):

    def test_qat_pass(self):

        batch_size = 8
        batch_num = 10
        sequence_len = 512
        vocab_size = 1000

        strategy = apply_pass()
        model, loss = get_gpt_model()
        opt = paddle.optimizer.AdamW(learning_rate=0.00001)
        inputs_spec, labels_spec = create_data_holder(batch_size=batch_size,
                                                      sequence_len=sequence_len)

        engine = Engine(model, inputs_spec, labels_spec, strategy=strategy)
        engine.prepare(optimizer=opt, loss=loss)

        dataset = FakeDataset(batch_size * batch_num, sequence_len, vocab_size)
        engine.fit(train_data=dataset, batch_size=batch_size)

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
