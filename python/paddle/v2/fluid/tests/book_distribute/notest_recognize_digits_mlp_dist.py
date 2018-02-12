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

BATCH_SIZE = 128
PASS_NUM = 100

images = fluid.layers.data(name='x', shape=[784], dtype='float32')

# TODO(aroraabhinav) Add regularization and error clipping after
# Issue 7432(https://github.com/PaddlePaddle/Paddle/issues/7432) is resolved.
hidden1 = fluid.layers.fc(input=images, size=128, act='relu')
hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')
predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')

label = fluid.layers.data(name='y', shape=[1], dtype='int64')

cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(x=cost)

optimizer = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
optimize_ops, params_grads = optimizer.minimize(avg_cost)

accuracy = fluid.evaluator.Accuracy(input=predict, label=label)

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=8192),
    batch_size=BATCH_SIZE)

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
t.transpile(optimize_ops, params_grads, pservers=pserver_endpoints, trainers=2)

if training_role == "PSERVER":
    if not current_endpoint:
        print("need env SERVER_ENDPOINT")
        exit(1)
    pserver_prog = t.get_pserver_program(current_endpoint)
    pserver_startup = t.get_startup_program(current_endpoint, pserver_prog)
    exe.run(pserver_startup)
    exe.run(pserver_prog)
elif training_role == "TRAINER":
    trainer_prog = t.get_trainer_program()
    feeder = fluid.DataFeeder(feed_list=[images, label], place=place)
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
