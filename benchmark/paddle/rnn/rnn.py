#!/usr/bin/env python

from paddle.trainer_config_helpers import *
import imdb

num_class = 2
vocab_size = 30000
fixedlen = 100
batch_size = get_config_arg('batch_size', int, 128)
lstm_num = get_config_arg('lstm_num', int, 1)
hidden_size = get_config_arg('hidden_size', int, 128)
# whether to pad sequence into fixed length
pad_seq = get_config_arg('pad_seq', bool, True)
imdb.create_data('imdb.pkl')

args = {'vocab_size': vocab_size, 'pad_seq': pad_seq, 'maxlen': fixedlen}
define_py_data_sources2(
    "train.list", None, module="provider", obj="process", args=args)

settings(
    batch_size=batch_size,
    learning_rate=2e-3,
    learning_method=AdamOptimizer(),
    regularization=L2Regularization(8e-4),
    gradient_clipping_threshold=25)

net = data_layer('data', size=vocab_size)
net = embedding_layer(input=net, size=128)

for i in xrange(lstm_num):
    net = simple_lstm(input=net, size=hidden_size)

net = last_seq(input=net)
net = fc_layer(input=net, size=2, act=SoftmaxActivation())

lab = data_layer('label', num_class)
loss = classification_cost(input=net, label=lab)
outputs(loss)
