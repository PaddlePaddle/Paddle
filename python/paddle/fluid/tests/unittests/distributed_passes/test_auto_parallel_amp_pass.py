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

import random
import numpy as np

import unittest
import paddle
import paddle.nn as nn
import paddle.distributed.fleet as fleet
import paddle.distributed.auto_parallel as auto
import auto_parallel_gpt_modeling as modeling
from paddle.distributed.passes import new_pass, PassManager
from auto_parallel_pass_test_base import AutoPallelPassTestBase
from auto_parallel_gpt_modeling import GPTModel, GPTForPretraining, GPTPretrainingCriterion


class TestAMPPass(AutoPallelPassTestBase):
    def init(self):
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        self.rtol = 1e-5
        self.atol = 1e-8

        rank = paddle.distributed.get_rank()
        paddle.seed(rank + 2021)
        random.seed(rank + 2021)
        np.random.seed(rank + 2021)

        modeling.init_global()
        modeling._global_parallel_strategy = "mp"
        modeling._global_process_mesh = auto.ProcessMesh(mesh=[0, 1])

    def apply_passes(self):
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            "custom_white_list": [
                'softmax',
                'layer_norm',
                'gelu',
            ],
            "custom_black_list": ['c_softmax_with_cross_entropy'],
            "init_loss_scaling": 32768,
            "use_dynamic_loss_scaling": True,
        }
        dist_strategy.pipeline = False
        dist_strategy.recompute = False
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)

    def apply_no_passes(self):
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.pipeline = False
        dist_strategy.recompute = False
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)

    def test_bs_8(self):
        self.check_main(
            gpus=[0, 1], batch_size=8, sequence_len=512, vocab_size=1000)

    def get_model(self, place, batch_size, sequence_len, vocab_size):
        tokens = paddle.static.data(
            name="tokens", shape=[batch_size, sequence_len], dtype='int64')
        position_ids = paddle.static.data(
            name="position_ids",
            shape=[batch_size, sequence_len],
            dtype='int64')
        attention_mask = paddle.static.data(
            name="attention_mask",
            shape=[batch_size, 1, sequence_len, sequence_len],
            dtype='float32')
        labels = paddle.static.data(
            name="labels", shape=[batch_size, sequence_len], dtype='int64')
        loss_mask = paddle.static.data(
            name="loss_mask", shape=[batch_size, sequence_len], dtype='float32')
        data_holder = [tokens, position_ids, attention_mask, labels, loss_mask]

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
            eol_token_id=3)

        model = GPTForPretraining(
            gpt, vocab_size=1000, hidden_size=64, initializer_range=0.02)
        preds = model(tokens, position_ids, attention_mask)
        criterion = GPTPretrainingCriterion()
        loss = criterion(preds, labels, loss_mask)

        optimizer = paddle.fluid.optimizer.AdamOptimizer(
            learning_rate=0.00001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            grad_clip=None)
        optimizer = fleet.distributed_optimizer(optimizer)
        startup_program = paddle.static.default_startup_program()
        _, _, dist_startup_prog, dist_main_prog = optimizer.minimize(
            loss, startup_program)

        def gen_data():
            np.random.seed(2021)
            for _ in range(10):
                tokens = []
                position_ids = []
                attention_mask = []
                labels = []
                loss_mask = []
                for _ in range(batch_size):
                    tokens.append(
                        np.random.randint(
                            vocab_size, size=sequence_len))
                    position_ids.append(np.arange(sequence_len))
                    attention_mask.append([np.tril(np.ones(sequence_len))])
                    labels.append(
                        np.random.randint(
                            vocab_size, size=sequence_len))
                    loss_mask.append(np.ones(sequence_len))

                yield tokens, position_ids, attention_mask, labels, loss_mask

        return dist_main_prog, dist_startup_prog, data_holder, [loss], gen_data


if __name__ == "__main__":
    unittest.main()
