# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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

import math
import os
import sys
from paddle.trainer_config_helpers import *

#file paths
word_dict_file = './data/src.dict'
label_dict_file = './data/tgt.dict'
train_list_file = './data/train.list'
test_list_file = './data/test.list'

is_test = get_config_arg('is_test', bool, False)
is_predict = get_config_arg('is_predict', bool, False)

if not is_predict:
    #load dictionaries
    word_dict = dict()
    label_dict = dict()
    with open(word_dict_file, 'r') as f_word, \
         open(label_dict_file, 'r') as f_label:
        for i, line in enumerate(f_word):
            w = line.strip()
            word_dict[w] = i

        for i, line in enumerate(f_label):
            w = line.strip()
            label_dict[w] = i

    if is_test:
        train_list_file = None

    #define data provider
    define_py_data_sources2(
        train_list=train_list_file,
        test_list=test_list_file,
        module='dataprovider',
        obj='process',
        args={'word_dict': word_dict,
              'label_dict': label_dict})

    word_dict_len = len(word_dict)
    label_dict_len = len(label_dict)

else:
    word_dict_len = get_config_arg('dict_len', int)
    label_dict_len = get_config_arg('label_len', int)

mark_dict_len = 2
word_dim = 32
mark_dim = 5
hidden_dim = 128
depth = 8
emb_lr = 1e-2
fc_lr = 1e-2
lstm_lr = 2e-2

settings(
    batch_size=150,
    learning_method=AdamOptimizer(),
    learning_rate=1e-3,
    regularization=L2Regularization(8e-4),
    gradient_clipping_threshold=25)

#6 features
word = data_layer(name='word_data', size=word_dict_len)
predicate = data_layer(name='verb_data', size=word_dict_len)
ctx_n1 = data_layer(name='ctx_n1_data', size=word_dict_len)
ctx_0 = data_layer(name='ctx_0_data', size=word_dict_len)
ctx_p1 = data_layer(name='ctx_p1_data', size=word_dict_len)
mark = data_layer(name='mark_data', size=mark_dict_len)

if not is_predict:
    target = data_layer(name='target', size=label_dict_len)

ptt = ParameterAttribute(name='src_emb', learning_rate=emb_lr)
layer_attr = ExtraLayerAttribute(drop_rate=0.5)
fc_para_attr = ParameterAttribute(learning_rate=fc_lr)
lstm_para_attr = ParameterAttribute(initial_std=0., learning_rate=lstm_lr)
para_attr = [fc_para_attr, lstm_para_attr]

word_embedding = embedding_layer(size=word_dim, input=word, param_attr=ptt)
predicate_embedding = embedding_layer(
    size=word_dim, input=predicate, param_attr=ptt)
ctx_n1_embedding = embedding_layer(size=word_dim, input=ctx_n1, param_attr=ptt)
ctx_0_embedding = embedding_layer(size=word_dim, input=ctx_0, param_attr=ptt)
ctx_p1_embedding = embedding_layer(size=word_dim, input=ctx_p1, param_attr=ptt)
mark_embedding = embedding_layer(size=mark_dim, input=mark)

hidden_0 = mixed_layer(
    size=hidden_dim,
    input=[
        full_matrix_projection(input=word_embedding),
        full_matrix_projection(input=predicate_embedding),
        full_matrix_projection(input=ctx_n1_embedding),
        full_matrix_projection(input=ctx_0_embedding),
        full_matrix_projection(input=ctx_p1_embedding),
        full_matrix_projection(input=mark_embedding),
    ])

lstm_0 = lstmemory(input=hidden_0, layer_attr=layer_attr)

#stack L-LSTM and R-LSTM with direct edges
input_tmp = [hidden_0, lstm_0]

for i in range(1, depth):

    fc = fc_layer(input=input_tmp, size=hidden_dim, param_attr=para_attr)

    lstm = lstmemory(
        input=fc,
        act=ReluActivation(),
        reverse=(i % 2) == 1,
        layer_attr=layer_attr)
    input_tmp = [fc, lstm]

prob = fc_layer(
    input=input_tmp,
    size=label_dict_len,
    act=SoftmaxActivation(),
    param_attr=para_attr)

if not is_predict:
    cls = classification_cost(input=prob, label=target)
    outputs(cls)
else:
    outputs(prob)
