# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from paddle.trainer_config_helpers import *

import math

define_py_data_sources2(
    train_list="data/train.list",
    test_list="data/test.list",
    module="dataprovider",
    obj="process")

batch_size = 16
settings(
    learning_method=MomentumOptimizer(),
    batch_size=batch_size,
    regularization=L2Regularization(batch_size * 1e-5),
    model_average=ModelAverage(0.5),
    learning_rate=2e-3,
    learning_rate_decay_a=5e-7,
    learning_rate_decay_b=0.5, )

word_dim = 128
hidden_dim = 128
with_rnn = True

initial_std = 1 / math.sqrt(hidden_dim)
param_attr = ParamAttr(initial_std=initial_std)
cpu_layer_attr = ExtraLayerAttribute(device=-1)

default_device(0)

num_label_types = 23

features = data_layer(name="features", size=76328)
word = data_layer(name="word", size=6778)
pos = data_layer(name="pos", size=44)
chunk = data_layer(
    name="chunk", size=num_label_types, layer_attr=cpu_layer_attr)

emb = embedding_layer(
    input=word, size=word_dim, param_attr=ParamAttr(initial_std=0))

hidden1 = mixed_layer(
    size=hidden_dim,
    act=STanhActivation(),
    bias_attr=True,
    input=[
        full_matrix_projection(emb), table_projection(
            pos, param_attr=param_attr)
    ])

if with_rnn:
    rnn1 = recurrent_layer(
        act=ReluActivation(),
        bias_attr=True,
        input=hidden1,
        param_attr=ParamAttr(initial_std=0), )

hidden2 = mixed_layer(
    size=hidden_dim,
    act=STanhActivation(),
    bias_attr=True,
    input=[full_matrix_projection(hidden1)] +
    ([full_matrix_projection(
        rnn1, param_attr=ParamAttr(initial_std=0))] if with_rnn else []), )

if with_rnn:
    rnn2 = recurrent_layer(
        reverse=True,
        act=ReluActivation(),
        bias_attr=True,
        input=hidden2,
        param_attr=ParamAttr(initial_std=0), )

crf_input = mixed_layer(
    size=num_label_types,
    bias_attr=False,
    input=[full_matrix_projection(hidden2), ] +
    ([full_matrix_projection(
        rnn2, param_attr=ParamAttr(initial_std=0))] if with_rnn else []), )

crf = crf_layer(
    input=crf_input,
    label=chunk,
    param_attr=ParamAttr(
        name="crfw", initial_std=0),
    layer_attr=cpu_layer_attr, )

crf_decoding = crf_decoding_layer(
    size=num_label_types,
    input=crf_input,
    label=chunk,
    param_attr=ParamAttr(name="crfw"),
    layer_attr=cpu_layer_attr, )

sum_evaluator(
    name="error",
    input=crf_decoding, )

chunk_evaluator(
    name="chunk_f1",
    input=crf_decoding,
    label=chunk,
    chunk_scheme="IOB",
    num_chunk_types=11, )

inputs(word, pos, chunk, features)
outputs(crf)
