# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import os

import paddle
import paddle.static as static
import paddle.nn as nn

paddle.enable_static()


class RNNEncoder(nn.Layer):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.0,
                 pooling_type=None,
                 **kwargs):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._direction = direction
        self._pooling_type = pooling_type

        self.rnn_layer = nn.SimpleRNN(input_size=input_size,
                                      hidden_size=hidden_size,
                                      num_layers=num_layers,
                                      direction=direction,
                                      dropout=dropout,
                                      **kwargs)

    def get_input_dim(self):
        return self._input_size

    def get_output_dim(self):
        if self._direction == "bidirect":
            return self._hidden_size * 2
        else:
            return self._hidden_size

    def forward(self, inputs, sequence_length):
        encoded_text, last_hidden = self.rnn_layer(
            inputs, sequence_length=sequence_length)
        output = paddle.max(encoded_text, axis=1)
        return output


class RNNModel(nn.Layer):

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 rnn_hidden_size=198,
                 direction='forward',
                 rnn_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=emb_dim,
                                     padding_idx=padding_idx)
        self.rnn_encoder = RNNEncoder(emb_dim,
                                      rnn_hidden_size,
                                      num_layers=rnn_layers,
                                      direction=direction,
                                      dropout=dropout_rate,
                                      pooling_type=pooling_type)
        self.fc = nn.Linear(self.rnn_encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        embedded_text = self.embedder(text)
        text_repr = self.rnn_encoder(embedded_text, sequence_length=seq_len)
        fc_out = paddle.tanh(self.fc(text_repr))
        logits = self.output_layer(fc_out)
        return logits


def rnn_pretrain_forward(train_program, start_program, topo=None):
    with static.program_guard(train_program,
                              start_program), paddle.utils.unique_name.guard():
        batch_size = 1
        tokens = static.data(name="tokens",
                             shape=[batch_size, -1],
                             dtype="int64")
        seq_len = static.data(name="ids", shape=[batch_size], dtype="int64")
        labels = static.data(name="labels", shape=[batch_size], dtype="int64")
        data_holders = [tokens, seq_len, labels]
        vocab_size = 10
        num_classes = 2
        pad_token_id = 0
        model = RNNModel(vocab_size,
                         num_classes,
                         direction='forward',
                         padding_idx=pad_token_id,
                         pooling_type='max')

        optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                          learning_rate=0.001)
        criterion = paddle.nn.CrossEntropyLoss()
        preds = model(tokens, seq_len)
        loss = criterion(preds, labels)

    return train_program, start_program, loss, optimizer, data_holders


class TestFleetMetaOptimizer(unittest.TestCase):

    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "1"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001,127.0.0.1:36002"

    def test_rnn_raw_optimizer(self):
        import paddle.distributed.fleet as fleet
        import paddle.distributed.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        train_program = static.Program()
        start_program = static.Program()
        train_program, start_program, loss, optimizer, data_holders = \
            rnn_pretrain_forward(train_program, start_program)
        with paddle.static.program_guard(
                train_program, start_program), paddle.utils.unique_name.guard():
            strategy = fleet.DistributedStrategy()
            strategy.without_graph_optimization = True
            strategy.fuse_all_reduce_ops = True
            fleet.init(is_collective=True, strategy=strategy)
            optimizer = fleet.distributed_optimizer(optimizer)
            optimizer.minimize(loss)


if __name__ == "__main__":
    unittest.main()
