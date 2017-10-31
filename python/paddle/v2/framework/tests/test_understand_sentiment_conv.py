import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.nets as nets
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program, g_program
from paddle.v2.framework.executor import Executor

import numpy as np


def convolution_net(input_dim, class_dim=2, emb_dim=128, hid_dim=128):
    data = layers.data(name="data", shape=[1], data_type="int64")
    label = layers.data(name="label", shape=[1], data_type="int64")

    emb = layers.embedding(input=data, size=[input_dim, emb_dim])
    conv_3 = nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=3,
        pool_type="max",
        act="tanh")
    conv_4 = nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=4,
        pool_type="max",
        act="tanh")
    output = layers.fc(input=[conv_3, conv_4], size=class_dim, act="softmax")

    cost = layers.cross_entropy(input=output, label=label)
    return cost, output


def main():
    word_dict = paddle.dataset.imdb.word_dict()
    dict_dim = len(word_dict)
    class_dim = 2
