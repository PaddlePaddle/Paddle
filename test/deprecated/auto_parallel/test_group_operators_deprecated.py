# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


import sys
import unittest

sys.path.append("../../legacy_test")
import auto_parallel_gpt_model as modeling
import numpy as np
from auto_parallel_gpt_model import (
    GPTForPretraining,
    GPTModel,
    GPTPretrainingCriterion,
)

import paddle
from paddle import static


def get_gpt_model(
    train_program, start_program, place, batch_size, sequence_len, vocab_size
):
    with static.program_guard(train_program, start_program):
        tokens = paddle.static.data(
            name="tokens", shape=[batch_size, sequence_len], dtype='int64'
        )
        position_ids = paddle.static.data(
            name="position_ids", shape=[batch_size, sequence_len], dtype='int64'
        )
        attention_mask = paddle.static.data(
            name="attention_mask",
            shape=[batch_size, 1, sequence_len, sequence_len],
            dtype='float32',
        )
        labels = paddle.static.data(
            name="labels", shape=[batch_size, sequence_len], dtype='int64'
        )
        loss_mask = paddle.static.data(
            name="loss_mask", shape=[batch_size, sequence_len], dtype='float32'
        )

        gpt = GPTModel(
            vocab_size=1000,
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
            eol_token_id=3,
        )

        model = GPTForPretraining(
            gpt, vocab_size=1000, hidden_size=64, initializer_range=0.02
        )
        preds = model(tokens, position_ids, attention_mask)
        criterion = GPTPretrainingCriterion()
        loss = criterion(preds, labels, loss_mask)

    def gen_data():
        np.random.seed(2021)
        tokens = []
        position_ids = []
        attention_mask = []
        labels = []
        loss_mask = []
        for _ in range(batch_size):
            tokens.append(np.random.randint(vocab_size, size=sequence_len))
            position_ids.append(np.arange(sequence_len))
            attention_mask.append([np.tril(np.ones(sequence_len))])
            labels.append(np.random.randint(vocab_size, size=sequence_len))
            loss_mask.append(np.ones(sequence_len))

        return tokens, position_ids, attention_mask, labels, loss_mask

    return train_program, start_program, loss, gen_data


class TestGroupOperators(unittest.TestCase):
    def test_gpt(self):
        modeling.init_global()
        train_program = static.Program()
        start_program = static.Program()
        place = paddle.set_device("gpu")
        batch_size = 8
        sequence_len = 512
        vocab_size = 1000
        train_program, start_program, loss, gen_data = get_gpt_model(
            train_program,
            start_program,
            place,
            batch_size,
            sequence_len,
            vocab_size,
        )
        from paddle.distributed.auto_parallel.static.dist_context import (
            DistributedContext,
        )
        from paddle.distributed.auto_parallel.static.tuner.rule_based_tuner import (
            RuleBasedTuner,
        )

        dist_context = DistributedContext(train_program)
        dist_context.initialize()
        tuner = RuleBasedTuner(dist_context)
        layers = tuner.cluster_operators()
        op_types = []
        for layer in layers:
            tmp = []
            for op in layer:
                tmp.append(op.type)
            op_types.append(tmp)


if __name__ == "__main__":
    unittest.main()
