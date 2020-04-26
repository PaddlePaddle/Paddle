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
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, Embedding
from paddle.fluid.dygraph.base import to_variable
import numpy as np
from hapi.model import Model
from hapi.text.text import GRUEncoderLayer as BiGRUEncoder
from hapi.text.text import BOWEncoder, CNNEncoder, GRUEncoder


class CNN(Model):
    def __init__(self,  dict_dim, batch_size, seq_len):
        super(CNN, self).__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.channels = 1
        self.win_size = [3, self.hid_dim]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self._encoder = CNNEncoder(
            dict_size=self.dict_dim + 1,
            emb_dim=self.emb_dim,
            seq_len=self.seq_len,
            filter_size= self.win_size,
            num_filters= self.hid_dim,
            hidden_dim= self.hid_dim,
            padding_idx=None,
            act='tanh')
        self._fc1 = Linear(input_dim = self.hid_dim*self.seq_len, output_dim=self.fc_hid_dim, act="softmax")
        self._fc_prediction = Linear(input_dim = self.fc_hid_dim,
                                 output_dim = self.class_dim,
                                 act="softmax")

    def forward(self, inputs):
        conv_3 = self._encoder(inputs)
        fc_1 = self._fc1(conv_3)
        prediction = self._fc_prediction(fc_1)
        return prediction


class BOW(Model):
    def __init__(self, dict_dim, batch_size, seq_len):
        super(BOW, self).__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self._encoder = BOWEncoder(
            dict_size=self.dict_dim + 1,
            emb_dim=self.emb_dim,
            padding_idx=None,
            bow_dim=self.hid_dim,
            seq_len=self.seq_len)
        self._fc1 = Linear(input_dim = self.hid_dim, output_dim=self.hid_dim, act="tanh")
        self._fc2 = Linear(input_dim = self.hid_dim, output_dim=self.fc_hid_dim, act="tanh")
        self._fc_prediction = Linear(input_dim = self.fc_hid_dim,
                                 output_dim = self.class_dim,
                                 act="softmax")

    def forward(self, inputs):
        bow_1 = self._encoder(inputs)
        bow_1 = fluid.layers.tanh(bow_1)
        fc_1 = self._fc1(bow_1)
        fc_2 = self._fc2(fc_1)
        prediction = self._fc_prediction(fc_2)
        return prediction


class GRU(Model):
    def __init__(self, dict_dim, batch_size, seq_len):
        super(GRU, self).__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self._fc1 = Linear(input_dim=self.hid_dim, output_dim=self.fc_hid_dim, act="tanh")
        self._fc_prediction = Linear(input_dim=self.fc_hid_dim,
                                 output_dim=self.class_dim,
                                 act="softmax")
        self._encoder = GRUEncoder(
            dict_size=self.dict_dim + 1,
            emb_dim=self.emb_dim,
            gru_dim=self.hid_dim,
            hidden_dim=self.hid_dim,
            padding_idx=None,
            seq_len=self.seq_len)

    def forward(self, inputs):
        emb = self._encoder(inputs)
        fc_1 = self._fc1(emb)
        prediction = self._fc_prediction(fc_1)
        return prediction

        
class BiGRU(Model):
    def __init__(self, dict_dim, batch_size, seq_len):
        super(BiGRU, self).__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding = Embedding(
            size=[self.dict_dim + 1, self.emb_dim],
            dtype='float32',
            param_attr=fluid.ParamAttr(learning_rate=30),
            is_sparse=False)
        h_0 = np.zeros((self.batch_size, self.hid_dim), dtype="float32")
        h_0 = to_variable(h_0)
        self._fc1 = Linear(input_dim = self.hid_dim, output_dim=self.hid_dim*3)
        self._fc2 = Linear(input_dim = self.hid_dim*2, output_dim=self.fc_hid_dim, act="tanh")
        self._fc_prediction = Linear(input_dim=self.fc_hid_dim,
                                 output_dim=self.class_dim,
                                 act="softmax")
        self._encoder = BiGRUEncoder(
            grnn_hidden_dim=self.hid_dim,
            input_dim=self.hid_dim * 3,
            h_0=h_0,
            init_bound=0.1,
            is_bidirection=True)

    def forward(self, inputs):
        emb = self.embedding(inputs)
        emb = fluid.layers.reshape(emb, shape=[self.batch_size, -1, self.hid_dim])
        fc_1 = self._fc1(emb)
        encoded_vector = self._encoder(fc_1)
        encoded_vector = fluid.layers.tanh(encoded_vector)
        encoded_vector = fluid.layers.reduce_max(encoded_vector, dim=1)
        fc_2 = self._fc2(encoded_vector)
        prediction = self._fc_prediction(fc_2)
        return prediction
