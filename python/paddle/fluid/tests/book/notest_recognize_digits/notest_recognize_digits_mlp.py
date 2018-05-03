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
import argparse
import paddle.fluid as fluid
import paddle
import sys
import numpy
import unittest
import math
import sys
import os
import paddle.v2.dataset as dataset

BATCH_SIZE = 64


def inference_program():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')

    hidden = fluid.layers.fc(input=img, size=200, act='tanh')
    hidden = fluid.layers.fc(input=hidden, size=200, act='tanh')
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction


def train_program():
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    predict = inference_program()
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predict, label=label)
    return avg_cost, acc


def train(use_cuda, save_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    trainer = fluid.Trainer(train_program, place=place, optimizer=optimizer)

    def event_handler(event):
        if isinstance(event, fluid.EndIteration):
            avg_cost, acc = event.values
            print("avg_cost: %s" % avg_cost)
            print("acc     : %s" % acc)

            if (event.batch_id + 1) % 10 == 0:
                test_metrics = trainer.test(reader=dataset.mnist.test())
                avg_cost_set = test_metrics[0]
                acc_set = test_metrics[1]

                # get test acc and loss
                acc = numpy.array(acc_set).mean()
                avg_cost = numpy.array(avg_cost_set).mean()
                if float(acc) > 0.2:  # Smaller value to increase CI speed
                    trainer.save_params(save_dirname)
                    return

                else:
                    print(
                        'BatchID {1:04}, Test Loss {2:2.2}, Acc {3:2.2}'.format(
                            event.batch_id + 1, float(avg_cost), float(acc)))
                    if math.isnan(float(avg_cost)):
                        sys.exit("got NaN loss, training failed.")

    trainer.train(
        reader=dataset.mnist.train(), num_pass=100, event_handler=event_handler)


def infer(use_cuda, save_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    inferencer = fluid.Inferencer(
        inference_program, param_path=save_dirname, place=place)

    batch_size = 1
    tensor_img = numpy.random.uniform(-1.0, 1.0,
                                      [batch_size, 1, 28, 28]).astype("float32")

    results = inferencer.infer({'img': tensor_img})

    print("infer results: ", results[0])


def main(use_cuda):
    save_dirname = "recognize_digits_mlp.inference.model"

    # call train() with is_local argument to run distributed train
    train(use_cuda=use_cuda, save_dirname=save_dirname)
    infer(use_cuda=use_cuda, save_dirname=save_dirname)


if __name__ == '__main__':
    for use_cuda in (False, True):
        main(use_cuda=use_cuda)
