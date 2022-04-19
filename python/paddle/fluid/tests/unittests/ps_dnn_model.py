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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import paddle.distributed.fleet as fleet
from paddle.distributed.ps.utils.public import logger


class DNNLayer(nn.Layer):
    def __init__(self,
                 sparse_feature_number,
                 sparse_feature_dim,
                 dense_feature_dim,
                 num_field,
                 layer_sizes,
                 sync_mode=None):
        super(DNNLayer, self).__init__()
        self.sync_mode = sync_mode
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        sizes = [sparse_feature_dim * num_field + dense_feature_dim
                 ] + self.layer_sizes + [2]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

    def forward(self, sparse_inputs, dense_inputs):

        sparse_embs = []
        for s_input in sparse_inputs:
            if self.sync_mode == "gpubox":
                emb = paddle.fluid.contrib.sparse_embedding(
                    input=s_input,
                    size=[self.sparse_feature_number, self.sparse_feature_dim],
                    param_attr=paddle.ParamAttr(name="embedding"))
            else:
                emb = self.embedding(s_input)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            # emb.stop_gradient = True
            sparse_embs.append(emb)

        y_dnn = paddle.concat(x=sparse_embs + [dense_inputs], axis=1)

        if self.sync_mode == 'heter':
            with paddle.fluid.device_guard('gpu'):
                for n_layer in self._mlp_layers:
                    y_dnn = n_layer(y_dnn)
        else:
            for n_layer in self._mlp_layers:
                y_dnn = n_layer(y_dnn)

        return y_dnn


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.infer_target_var = None
        self.config = config
        self._init_hyper_parameters()
        self.sync_mode = config.get("runner.sync_mode")

    def _init_hyper_parameters(self):
        self.is_distributed = False
        self.distributed_embedding = False

        if self.config.get("hyper_parameters.distributed_embedding", 0) == 1:
            self.distributed_embedding = True

        self.sparse_feature_number = self.config.get(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = self.config.get(
            "hyper_parameters.sparse_feature_dim")
        self.sparse_inputs_slots = self.config.get(
            "hyper_parameters.sparse_inputs_slots")
        self.dense_input_dim = self.config.get(
            "hyper_parameters.dense_input_dim")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.fc_sizes = self.config.get("hyper_parameters.fc_sizes")

    def create_feeds(self, is_infer=False):
        dense_input = paddle.static.data(
            name="dense_input",
            shape=[None, self.dense_input_dim],
            dtype="float32")

        sparse_input_ids = [
            paddle.static.data(
                name="C" + str(i), shape=[None, 1], dtype="int64")
            for i in range(1, self.sparse_inputs_slots)
        ]

        label = paddle.static.data(name="label", shape=[None, 1], dtype="int64")

        feeds_list = [label] + sparse_input_ids + [dense_input]
        return feeds_list

    def net(self, input, is_infer=False):
        self.label_input = input[0]
        self.sparse_inputs = input[1:self.sparse_inputs_slots]
        self.dense_input = input[-1]
        sparse_number = self.sparse_inputs_slots - 1

        dnn_model = DNNLayer(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            self.dense_input_dim,
            sparse_number,
            self.fc_sizes,
            sync_mode=self.sync_mode)

        raw_predict_2d = dnn_model.forward(self.sparse_inputs, self.dense_input)

        predict_2d = paddle.nn.functional.softmax(raw_predict_2d)

        self.predict = predict_2d

        auc, batch_auc, [
            self.batch_stat_pos, self.batch_stat_neg, self.stat_pos,
            self.stat_neg
        ] = paddle.static.auc(input=self.predict,
                              label=self.label_input,
                              num_thresholds=2**12,
                              slide_steps=20)
        self.inference_target_var = auc
        if is_infer:
            fetch_dict = {'auc': auc}
            return fetch_dict

        cost = paddle.nn.functional.cross_entropy(
            input=raw_predict_2d, label=self.label_input)
        avg_cost = paddle.mean(x=cost)
        self._cost = avg_cost

        fetch_dict = {'cost': avg_cost, 'auc': auc}
        return fetch_dict
