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

import paddle
import paddle.static as static
from paddle.distributed import fleet
import sys

import numpy as np

import paddle.distributed.auto_parallel as auto
from auto_parallel_relaunch_model import mlp_pretrain_forward
from auto_parallel_relaunch_model import batch_generator_creator
sys.path.append("..")
import auto_parallel_gpt_model as modeling
from auto_parallel_gpt_model import GPTModel, GPTForPretraining, GPTPretrainingCriterion


def get_gpt_model(train_program, start_program, place, batch_size, sequence_len,
                  vocab_size):
    modeling.init_global()
    with static.program_guard(train_program, start_program):
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


def train():
    dist_strategy = fleet.DistributedStrategy()
    # init parallel optimizer
    dist_strategy.auto_search = True
    fleet.init(is_collective=True, strategy=dist_strategy)
    train_program = static.Program()
    start_program = static.Program()
    place = paddle.set_device("gpu")
    gpus = [0, 1]
    batch_size = 8
    sequence_len = 512
    vocab_size = 1000
    train_program, start_program, loss, gen_data = get_gpt_model(
        train_program, start_program, place, batch_size, sequence_len,
        vocab_size)

    optimizer = paddle.fluid.optimizer.AdamOptimizer(
        learning_rate=0.00001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        grad_clip=None)
    optimizer = fleet.distributed_optimizer(optimizer)
    _, _, distributed_startup_program, distributed_main_program = optimizer.minimize(
        loss, start_program)

    places = static.cuda_places()
    exe = paddle.static.Executor(places[0])
    exe.run(distributed_startup_program)

    for step in range(10):
        tokens, position_ids, attention_mask, labels, loss_mask = gen_data()
        if loss.name in distributed_main_program.global_block().vars:
            loss_print, = exe.run(distributed_main_program,
                                  feed={
                                      "tokens": tokens,
                                      "position_ids": position_ids,
                                      "attention_mask": attention_mask,
                                      "labels": labels,
                                      "loss_mask": loss_mask
                                  },
                                  fetch_list=[loss])
            print("step: %s, loss: %f" % (step, loss_print[0]))
        else:
            exe.run(distributed_main_program,
                    feed={
                        "tokens": tokens,
                        "position_ids": position_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                        "loss_mask": loss_mask
                    })
            print("step: %s, loss: %s" % (step, "None"))


if __name__ == "__main__":
    train()
