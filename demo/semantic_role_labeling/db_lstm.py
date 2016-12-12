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

import math
import os
import sys
from paddle.trainer_config_helpers import *

#file paths
word_dict_file = './data/wordDict.txt'
label_dict_file = './data/targetDict.txt'
predicate_file = './data/verbDict.txt'
train_list_file = './data/train.list'
test_list_file = './data/test.list'

is_test = get_config_arg('is_test', bool, False)
is_predict = get_config_arg('is_predict', bool, False)

if not is_predict:
    #load dictionaries
    word_dict = dict()
    label_dict = dict()
    predicate_dict = dict()
    with open(word_dict_file, 'r') as f_word, \
         open(label_dict_file, 'r') as f_label, \
         open(predicate_file, 'r') as f_pre:
        for i, line in enumerate(f_word):
            w = line.strip()
            word_dict[w] = i

        for i, line in enumerate(f_label):
            w = line.strip()
            label_dict[w] = i

        for i, line in enumerate(f_pre):
            w = line.strip()
            predicate_dict[w] = i

    if is_test:
        train_list_file = None

    #define data provider
    define_py_data_sources2(
        train_list=train_list_file,
        test_list=test_list_file,
        module='dataprovider',
        obj='process',
        args={
            'word_dict': word_dict,
            'label_dict': label_dict,
            'predicate_dict': predicate_dict
        })

    word_dict_len = len(word_dict)
    label_dict_len = len(label_dict)
    pred_len = len(predicate_dict)

else:
    word_dict_len = get_config_arg('dict_len', int)
    label_dict_len = get_config_arg('label_len', int)
    pred_len = get_config_arg('pred_len', int)

############################## Hyper-parameters ##################################
mark_dict_len = 2
word_dim = 32
mark_dim = 5
hidden_dim = 512
depth = 8

########################### Optimizer #######################################

settings(
    batch_size=150,
    learning_method=MomentumOptimizer(momentum=0),
    learning_rate=2e-2,
    regularization=L2Regularization(8e-4),
    is_async=False,
    model_average=ModelAverage(
        average_window=0.5, max_average_window=10000), )

####################################### network ##############################
#8 features and 1 target
word = data_layer(name='word_data', size=word_dict_len)
predicate = data_layer(name='verb_data', size=pred_len)

ctx_n2 = data_layer(name='ctx_n2_data', size=word_dict_len)
ctx_n1 = data_layer(name='ctx_n1_data', size=word_dict_len)
ctx_0 = data_layer(name='ctx_0_data', size=word_dict_len)
ctx_p1 = data_layer(name='ctx_p1_data', size=word_dict_len)
ctx_p2 = data_layer(name='ctx_p2_data', size=word_dict_len)
mark = data_layer(name='mark_data', size=mark_dict_len)

if not is_predict:
    target = data_layer(name='target', size=label_dict_len)

default_std = 1 / math.sqrt(hidden_dim) / 3.0

emb_para = ParameterAttribute(name='emb', initial_std=0., learning_rate=0.)
std_0 = ParameterAttribute(initial_std=0.)
std_default = ParameterAttribute(initial_std=default_std)

predicate_embedding = embedding_layer(
    size=word_dim,
    input=predicate,
    param_attr=ParameterAttribute(
        name='vemb', initial_std=default_std))
mark_embedding = embedding_layer(
    name='word_ctx-in_embedding', size=mark_dim, input=mark, param_attr=std_0)

word_input = [word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2]
emb_layers = [
    embedding_layer(
        size=word_dim, input=x, param_attr=emb_para) for x in word_input
]
emb_layers.append(predicate_embedding)
emb_layers.append(mark_embedding)

hidden_0 = mixed_layer(
    name='hidden0',
    size=hidden_dim,
    bias_attr=std_default,
    input=[
        full_matrix_projection(
            input=emb, param_attr=std_default) for emb in emb_layers
    ])

mix_hidden_lr = 1e-3
lstm_para_attr = ParameterAttribute(initial_std=0.0, learning_rate=1.0)
hidden_para_attr = ParameterAttribute(
    initial_std=default_std, learning_rate=mix_hidden_lr)

lstm_0 = lstmemory(
    name='lstm0',
    input=hidden_0,
    act=ReluActivation(),
    gate_act=SigmoidActivation(),
    state_act=SigmoidActivation(),
    bias_attr=std_0,
    param_attr=lstm_para_attr)

#stack L-LSTM and R-LSTM with direct edges
input_tmp = [hidden_0, lstm_0]

for i in range(1, depth):

    mix_hidden = mixed_layer(
        name='hidden' + str(i),
        size=hidden_dim,
        bias_attr=std_default,
        input=[
            full_matrix_projection(
                input=input_tmp[0], param_attr=hidden_para_attr),
            full_matrix_projection(
                input=input_tmp[1], param_attr=lstm_para_attr)
        ])

    lstm = lstmemory(
        name='lstm' + str(i),
        input=mix_hidden,
        act=ReluActivation(),
        gate_act=SigmoidActivation(),
        state_act=SigmoidActivation(),
        reverse=((i % 2) == 1),
        bias_attr=std_0,
        param_attr=lstm_para_attr)

    input_tmp = [mix_hidden, lstm]

feature_out = mixed_layer(
    name='output',
    size=label_dict_len,
    bias_attr=std_default,
    input=[
        full_matrix_projection(
            input=input_tmp[0], param_attr=hidden_para_attr),
        full_matrix_projection(
            input=input_tmp[1], param_attr=lstm_para_attr)
    ], )

if not is_predict:
    crf_l = crf_layer(
        name='crf',
        size=label_dict_len,
        input=feature_out,
        label=target,
        param_attr=ParameterAttribute(
            name='crfw', initial_std=default_std, learning_rate=mix_hidden_lr))

    crf_dec_l = crf_decoding_layer(
        name='crf_dec_l',
        size=label_dict_len,
        input=feature_out,
        label=target,
        param_attr=ParameterAttribute(name='crfw'))

    eval = sum_evaluator(input=crf_dec_l)

    outputs(crf_l)

else:
    crf_dec_l = crf_decoding_layer(
        name='crf_dec_l',
        size=label_dict_len,
        input=feature_out,
        param_attr=ParameterAttribute(name='crfw'))

    outputs(crf_dec_l)
