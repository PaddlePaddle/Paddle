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

<<<<<<< HEAD
import random
import sys

import numpy as np
=======
import sys
import numpy as np
import random
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

import paddle
from paddle.distributed.fleet import auto

sys.path.append("..")
import auto_parallel_gpt_model as modeling
<<<<<<< HEAD
from auto_parallel_gpt_model import (
    GPTForPretraining,
    GPTModel,
    GPTPretrainingCriterion,
)


class FakeDataset(paddle.io.Dataset):
    def __init__(self, num_samples, vocab_size=1000, sequence_len=512):
=======
from auto_parallel_gpt_model import GPTModel, GPTForPretraining, GPTPretrainingCriterion

sequence_len = 512
vocab_size = 1000


class FakeDataset(paddle.io.Dataset):

    def __init__(self, num_samples):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.num_samples = num_samples
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size

    def __getitem__(self, idx):
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        tokens = np.random.randint(self.vocab_size, size=self.sequence_len)
        position_ids = np.arange(self.sequence_len)
<<<<<<< HEAD
        attention_mask = (
            np.tril(np.ones(self.sequence_len))
            .reshape((1, self.sequence_len, self.sequence_len))
            .astype(np.float32)
        )
=======
        attention_mask = np.tril(np.ones(self.sequence_len)).reshape(
            (1, self.sequence_len, self.sequence_len)).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        labels = np.random.randint(self.vocab_size, size=self.sequence_len)
        loss_mask = np.ones(self.sequence_len).astype(np.float32)
        return tokens, position_ids, attention_mask, labels, loss_mask

    def __len__(self):
        return self.num_samples


<<<<<<< HEAD
def create_data_holder(batch_size, vocab_size=1000, sequence_len=512):
    tokens = paddle.static.InputSpec(
        name="tokens", shape=[batch_size, sequence_len], dtype='int64'
    )
    position_ids = paddle.static.InputSpec(
        name="position_ids", shape=[batch_size, sequence_len], dtype='int64'
    )
    attention_mask = paddle.static.InputSpec(
        name="attention_mask",
        shape=[batch_size, 1, sequence_len, sequence_len],
        dtype='float32',
    )
    labels = paddle.static.InputSpec(
        name="labels", shape=[batch_size, sequence_len], dtype='int64'
    )
    loss_mask = paddle.static.InputSpec(
        name="loss_mask", shape=[batch_size, sequence_len], dtype='float32'
    )
=======
def create_data_holder(batch_size):
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    return [tokens, position_ids, attention_mask], [labels, loss_mask]


def generate_model(strategy):
    modeling.init_global()
    ranks = list(range(paddle.distributed.get_world_size()))
<<<<<<< HEAD
    modeling._global_process_mesh = auto.ProcessMesh(
        mesh=ranks, dim_names=["x"]
    )
=======
    modeling._global_process_mesh = auto.ProcessMesh(mesh=ranks,
                                                     dim_names=["x"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if strategy == "serial":
        modeling._global_parallel_strategy = "serial"
    elif strategy == "mp":
        modeling._global_parallel_strategy = "mp"
    elif strategy == "dp":
        modeling._global_parallel_strategy = "dp"
    else:
        raise ValueError("Only support serial, mp2 and dp2.")

<<<<<<< HEAD
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
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    criterion = GPTPretrainingCriterion()
    return model, criterion
