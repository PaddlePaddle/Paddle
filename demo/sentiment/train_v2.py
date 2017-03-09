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

import sys
import paddle.v2 as paddle


def convolution_net(input_dim, class_dim=2, emb_dim=128, hid_dim=128):
    data = paddle.layer.data("word",
                             paddle.data_type.integer_value_sequence(input_dim))
    emb = paddle.layer.embedding(input=data, size=emb_dim)
    conv_3 = paddle.networks.sequence_conv_pool(
        input=emb, context_len=3, hidden_size=hid_dim)
    conv_4 = paddle.networks.sequence_conv_pool(
        input=emb, context_len=4, hidden_size=hid_dim)
    output = paddle.layer.fc(input=[conv_3, conv_4],
                             size=class_dim,
                             act=paddle.activation.Softmax())
    lbl = paddle.layer.data("label", paddle.data_type.integer_value(2))
    cost = paddle.layer.classification_cost(input=output, label=lbl)
    return cost


def stacked_lstm_net(input_dim,
                     class_dim=2,
                     emb_dim=128,
                     hid_dim=512,
                     stacked_num=3):
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
    """
    assert stacked_num % 2 == 1

    layer_attr = paddle.attr.Extra(drop_rate=0.5)
    fc_para_attr = paddle.attr.Param(learning_rate=1e-3)
    lstm_para_attr = paddle.attr.Param(initial_std=0., learning_rate=1.)
    para_attr = [fc_para_attr, lstm_para_attr]
    bias_attr = paddle.attr.Param(initial_std=0., l2_rate=0.)
    relu = paddle.activation.Relu()
    linear = paddle.activation.Linear()

    data = paddle.layer.data("word",
                             paddle.data_type.integer_value_sequence(input_dim))
    emb = paddle.layer.embedding(input=data, size=emb_dim)

    fc1 = paddle.layer.fc(input=emb,
                          size=hid_dim,
                          act=linear,
                          bias_attr=bias_attr)
    lstm1 = paddle.layer.lstmemory(
        input=fc1, act=relu, bias_attr=bias_attr, layer_attr=layer_attr)

    inputs = [fc1, lstm1]
    for i in range(2, stacked_num + 1):
        fc = paddle.layer.fc(input=inputs,
                             size=hid_dim,
                             act=linear,
                             param_attr=para_attr,
                             bias_attr=bias_attr)
        lstm = paddle.layer.lstmemory(
            input=fc,
            reverse=(i % 2) == 0,
            act=relu,
            bias_attr=bias_attr,
            layer_attr=layer_attr)
        inputs = [fc, lstm]

    fc_last = paddle.layer.pooling(
        input=inputs[0], pooling_type=paddle.pooling.Max())
    lstm_last = paddle.layer.pooling(
        input=inputs[1], pooling_type=paddle.pooling.Max())
    output = paddle.layer.fc(input=[fc_last, lstm_last],
                             size=class_dim,
                             act=paddle.activation.Softmax(),
                             bias_attr=bias_attr,
                             param_attr=para_attr)

    lbl = paddle.layer.data("label", paddle.data_type.integer_value(2))
    cost = paddle.layer.classification_cost(input=output, label=lbl)
    return cost


if __name__ == '__main__':
    # init
    paddle.init(use_gpu=False)

    #data
    print 'load dictionary...'
    word_dict = paddle.dataset.imdb.word_dict()
    dict_dim = len(word_dict)
    class_dim = 2
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            lambda: paddle.dataset.imdb.train(word_dict), buf_size=1000),
        batch_size=100)
    test_reader = paddle.batch(
        lambda: paddle.dataset.imdb.test(word_dict), batch_size=100)

    feeding = {'word': 0, 'label': 1}

    # network config
    # Please choose the way to build the network
    # by uncommenting the corresponding line.
    cost = convolution_net(dict_dim, class_dim=class_dim)
    # cost = stacked_lstm_net(dict_dim, class_dim=class_dim, stacked_num=3)

    # create parameters
    parameters = paddle.parameters.create(cost)

    # create optimizer
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=2e-3,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    # End batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_reader, feeding=feeding)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    # create trainer
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=adam_optimizer)

    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        feeding=feeding,
        num_passes=2)
