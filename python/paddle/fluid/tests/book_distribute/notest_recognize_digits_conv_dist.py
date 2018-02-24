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

images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
conv_pool_1 = fluid.nets.simple_img_conv_pool(
    input=images,
    filter_size=5,
    num_filters=20,
    pool_size=2,
    pool_stride=2,
    act="relu")
conv_pool_2 = fluid.nets.simple_img_conv_pool(
    input=conv_pool_1,
    filter_size=5,
    num_filters=50,
    pool_size=2,
    pool_stride=2,
    act="relu")

predict = fluid.layers.fc(input=conv_pool_2, size=10, act="softmax")
cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(x=cost)
optimizer = fluid.optimizer.Adam(learning_rate=0.01)
optimize_ops, params_grads = optimizer.minimize(avg_cost)

accuracy = fluid.evaluator.Accuracy(input=predict, label=label)

BATCH_SIZE = 50
PASS_NUM = 3
train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=500),
    batch_size=BATCH_SIZE)

place = fluid.CPUPlace()
exe = fluid.Executor(place)

pserver_endpoints = os.getenv("PSERVERS")  # all pserver endpoints
trainers = int(os.getenv("TRAINERS"))  # total trainer count
current_endpoint = os.getenv("SERVER_ENDPOINT")  # current pserver endpoint
training_role = os.getenv("TRAINING_ROLE",
                          "TRAINER")  # get the training role: trainer/pserver
if not current_endpoint:
    print("need env SERVER_ENDPOINT")
    exit(1)

t = fluid.DistributeTranspiler()
t.transpile(
    optimize_ops,
    params_grads,
    0,
    pservers=pserver_endpoints,
    trainers=trainers)

if training_role == "PSERVER":
    pserver_prog = t.get_pserver_program(current_endpoint)
    pserver_startup = t.get_startup_program(current_endpoint, pserver_prog)
    exe.run(pserver_startup)
    exe.run(pserver_prog)
elif training_role == "TRAINER":
    trainer_prog = t.get_trainer_program()
    feeder = fluid.DataFeeder(feed_list=[images, label], place=place)
    # TODO(typhoonzero): change trainer startup program to fetch parameters from pserver
    exe.run(fluid.default_startup_program())

    for pass_id in range(PASS_NUM):
        accuracy.reset(exe)
        batch_id = 0
        for data in train_reader():
            loss, acc = exe.run(trainer_prog,
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost] + accuracy.metrics)
            pass_acc = accuracy.eval(exe)
            if batch_id % 100 == 0:
                print("batch_id %d, loss: %f, acc: %f" %
                      (batch_id, loss, pass_acc))
            batch_id += 1

        pass_acc = accuracy.eval(exe)
        print("pass_id=" + str(pass_id) + " pass_acc=" + str(pass_acc))
else:
    print("environment var TRAINER_ROLE should be TRAINER os PSERVER")
