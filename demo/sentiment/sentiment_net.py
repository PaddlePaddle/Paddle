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

from os.path import join as join_path

from paddle.trainer_config_helpers import *


def sentiment_data(data_dir=None,
                   is_test=False,
                   is_predict=False,
                   train_list="train.list",
                   test_list="test.list",
                   dict_file="dict.txt"):
    """
    Predefined data provider for sentiment analysis.
    is_test: whether this config is used for test.
    is_predict: whether this config is used for prediction.
    train_list: text file name, containing a list of training set.
    test_list: text file name, containing a list of testing set.
    dict_file: text file name, containing dictionary.
    """
    dict_dim = len(open(join_path(data_dir, "dict.txt")).readlines())
    class_dim = len(open(join_path(data_dir, 'labels.list')).readlines())
    if is_predict:
        return dict_dim, class_dim

    if data_dir is not None:
        train_list = join_path(data_dir, train_list)
        test_list = join_path(data_dir, test_list)
        dict_file = join_path(data_dir, dict_file)

    train_list = train_list if not is_test else None
    word_dict = dict()
    with open(dict_file, 'r') as f:
        for i, line in enumerate(open(dict_file, 'r')):
            word_dict[line.split('\t')[0]] = i

    define_py_data_sources2(
        train_list,
        test_list,
        module="dataprovider",
        obj="process",
        args={'dictionary': word_dict})

    return dict_dim, class_dim


def bidirectional_lstm_net(input_dim,
                           class_dim=2,
                           emb_dim=128,
                           lstm_dim=128,
                           is_predict=False):
    data = data_layer("word", input_dim)
    emb = embedding_layer(input=data, size=emb_dim)
    bi_lstm = bidirectional_lstm(input=emb, size=lstm_dim)
    dropout = dropout_layer(input=bi_lstm, dropout_rate=0.5)
    output = fc_layer(input=dropout, size=class_dim, act=SoftmaxActivation())

    if not is_predict:
        lbl = data_layer("label", 1)
        outputs(classification_cost(input=output, label=lbl))
    else:
        outputs(output)


def stacked_lstm_net(input_dim,
                     class_dim=2,
                     emb_dim=128,
                     hid_dim=512,
                     stacked_num=3,
                     is_predict=False):
    """
    A Wrapper for sentiment classification task.
    This network uses bi-directional recurrent network,
    consisting three LSTM layers. This configure is referred to
    the paper as following url, but use fewer layrs.
        http://www.aclweb.org/anthology/P15-1109

    input_dim: here is word dictionary dimension.
    class_dim: number of categories.
    emb_dim: dimension of word embedding.
    hid_dim: dimension of hidden layer.
    stacked_num: number of stacked lstm-hidden layer.
    is_predict: is predicting or not.
                Some layers is not needed in network when predicting.
    """
    hid_lr = 1e-3
    assert stacked_num % 2 == 1

    layer_attr = ExtraLayerAttribute(drop_rate=0.5)
    fc_para_attr = ParameterAttribute(learning_rate=hid_lr)
    lstm_para_attr = ParameterAttribute(initial_std=0., learning_rate=1.)
    para_attr = [fc_para_attr, lstm_para_attr]
    bias_attr = ParameterAttribute(initial_std=0., l2_rate=0.)
    relu = ReluActivation()
    linear = LinearActivation()

    data = data_layer("word", input_dim)
    emb = embedding_layer(input=data, size=emb_dim)

    fc1 = fc_layer(input=emb, size=hid_dim, act=linear, bias_attr=bias_attr)
    lstm1 = lstmemory(
        input=fc1, act=relu, bias_attr=bias_attr, layer_attr=layer_attr)

    inputs = [fc1, lstm1]
    for i in range(2, stacked_num + 1):
        fc = fc_layer(
            input=inputs,
            size=hid_dim,
            act=linear,
            param_attr=para_attr,
            bias_attr=bias_attr)
        lstm = lstmemory(
            input=fc,
            reverse=(i % 2) == 0,
            act=relu,
            bias_attr=bias_attr,
            layer_attr=layer_attr)
        inputs = [fc, lstm]

    fc_last = pooling_layer(input=inputs[0], pooling_type=MaxPooling())
    lstm_last = pooling_layer(input=inputs[1], pooling_type=MaxPooling())
    output = fc_layer(
        input=[fc_last, lstm_last],
        size=class_dim,
        act=SoftmaxActivation(),
        bias_attr=bias_attr,
        param_attr=para_attr)

    if is_predict:
        outputs(output)
    else:
        outputs(classification_cost(input=output, label=data_layer('label', 1)))
