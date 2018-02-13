#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import os

PASS_NUM = 100
EMBED_SIZE = 32
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 32
IS_SPARSE = True
TRAINERS = 2

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)

first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
forth_word = fluid.layers.data(name='forthw', shape=[1], dtype='int64')
next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

embed_first = fluid.layers.embedding(
    input=first_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=IS_SPARSE,
    param_attr='shared_w')
embed_second = fluid.layers.embedding(
    input=second_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=IS_SPARSE,
    param_attr='shared_w')
embed_third = fluid.layers.embedding(
    input=third_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=IS_SPARSE,
    param_attr='shared_w')
embed_forth = fluid.layers.embedding(
    input=forth_word,
    size=[dict_size, EMBED_SIZE],
    dtype='float32',
    is_sparse=IS_SPARSE,
    param_attr='shared_w')

concat_embed = fluid.layers.concat(
    input=[embed_first, embed_second, embed_third, embed_forth], axis=1)
hidden1 = fluid.layers.fc(input=concat_embed, size=HIDDEN_SIZE, act='sigmoid')
predict_word = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')
cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
avg_cost = fluid.layers.mean(x=cost)
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimize_ops, params_grads = sgd_optimizer.minimize(avg_cost)
train_reader = paddle.batch(
    paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)

place = fluid.CPUPlace()
exe = fluid.Executor(place)

t = fluid.DistributeTranspiler()
# all parameter server endpoints list for spliting parameters
pserver_endpoints = os.getenv("PSERVERS")
# server endpoint for current node
current_endpoint = os.getenv("SERVER_ENDPOINT")
# run as trainer or parameter server
training_role = os.getenv("TRAINING_ROLE",
                          "TRAINER")  # get the training role: trainer/pserver
t.transpile(
    optimize_ops, params_grads, pservers=pserver_endpoints, trainers=TRAINERS)
if training_role == "PSERVER":
    if not current_endpoint:
        print("need env SERVER_ENDPOINT")
        exit(1)
    pserver_prog = t.get_pserver_program(current_endpoint)
    pserver_startup = t.get_startup_program(current_endpoint, pserver_prog)
    exe.run(pserver_startup)
    exe.run(pserver_prog)
elif training_role == "TRAINER":
    feeder = fluid.DataFeeder(
        feed_list=[first_word, second_word, third_word, forth_word, next_word],
        place=place)
    exe.run(fluid.default_startup_program())
    for pass_id in range(PASS_NUM):
        for data in train_reader():
            avg_cost_np = exe.run(t.get_trainer_program(),
                                  feed=feeder.feed(data),
                                  fetch_list=[avg_cost])
            print("avg_cost_np", avg_cost_np)
            if avg_cost_np[0] < 5.0:
                exit(
                    0)  # if avg cost less than 10.0, we think our code is good.
else:
    print("environment var TRAINER_ROLE should be TRAINER os PSERVER")
exit(1)
