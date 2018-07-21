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

import numpy as np
import argparse
import time
import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from paddle.fluid import core
import unittest
from multiprocessing import Process
import os
import sys
import signal

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class SE_ResNeXt():
    def __init__(self, layers=50):
        self.params = train_parameters
        self.layers = layers

    def net(self, input, class_dim=1000):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu')
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')
        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu')
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')
        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=3,
                stride=2,
                act='relu')
            conv = self.conv_bn_layer(
                input=conv, num_filters=64, filter_size=3, stride=1, act='relu')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=128,
                filter_size=3,
                stride=1,
                act='relu')
            conv = fluid.layers.pool2d(
                input=conv, pool_size=3, pool_stride=2, pool_padding=1, \
                pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    cardinality=cardinality,
                    reduction_ratio=reduction_ratio)

        pool = fluid.layers.pool2d(
            input=conv, pool_size=7, pool_type='avg', global_pooling=True)
        drop = fluid.layers.dropout(x=pool, dropout_prob=0.2)
        stdv = 1.0 / math.sqrt(drop.shape[1] * 1.0)
        out = fluid.layers.fc(input=drop, size=class_dim, act='softmax')
        return out

    def shortcut(self, input, ch_out, stride):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            filter_size = 1
            return self.conv_bn_layer(input, ch_out, filter_size, stride)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, cardinality,
                         reduction_ratio):
        conv0 = self.conv_bn_layer(
            input=input, num_filters=num_filters, filter_size=1, act='relu')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act='relu')
        conv2 = self.conv_bn_layer(
            input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
        scale = self.squeeze_excitation(
            input=conv2,
            num_channels=num_filters * 2,
            reduction_ratio=reduction_ratio)

        short = self.shortcut(input, num_filters * 2, stride)

        return fluid.layers.elementwise_add(x=short, y=scale, act='relu')

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) / 2,
            groups=groups,
            act=None,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)

    def squeeze_excitation(self, input, num_channels, reduction_ratio):
        pool = fluid.layers.pool2d(
            input=input, pool_size=0, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(input=pool,
                                  size=num_channels / reduction_ratio,
                                  act='relu')
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(input=squeeze,
                                     size=num_channels,
                                     act='sigmoid')
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale


def get_model(batch_size):
    # Input data
    image = fluid.layers.fill_constant(
        shape=[batch_size, 3, 224, 224], dtype='float32', value=0.0)
    label = fluid.layers.fill_constant(
        shape=[batch_size, 1], dtype='int64', value=0.0)

    # Train program
    model = SE_ResNeXt(layers=50)
    out = model.net(input=image, class_dim=102)
    cost = fluid.layers.cross_entropy(input=out, label=label)

    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    # Evaluator
    test_program = fluid.default_main_program().clone(for_test=True)

    # Optimization
    total_images = 6149  # flowers
    epochs = [30, 60, 90]
    step = int(total_images / batch_size + 1)

    bd = [step * e for e in epochs]
    base_lr = 0.1
    lr = []
    lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]

    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    optimizer.minimize(avg_cost)

    # Reader
    train_reader = paddle.batch(
        paddle.dataset.flowers.train(), batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.dataset.flowers.test(), batch_size=batch_size)

    return test_program, avg_cost, train_reader, test_reader, acc_top1, out


def get_transpiler(trainer_id, main_program, pserver_endpoints, trainers):
    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id=trainer_id,
        program=main_program,
        pservers=pserver_endpoints,
        trainers=trainers)
    return t


class DistSeResneXt2x2:
    def run_pserver(self, pserver_endpoints, trainers, current_endpoint,
                    trainer_id):
        get_model(batch_size=2)
        t = get_transpiler(trainer_id,
                           fluid.default_main_program(), pserver_endpoints,
                           trainers)
        pserver_prog = t.get_pserver_program(current_endpoint)
        startup_prog = t.get_startup_program(current_endpoint, pserver_prog)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        exe.run(pserver_prog)

    def _wait_ps_ready(self, pid):
        retry_times = 20
        while True:
            assert retry_times >= 0, "wait ps ready failed"
            time.sleep(3)
            print("waiting ps ready: ", pid)
            try:
                # the listen_and_serv_op would touch a file which contains the listen port
                # on the /tmp directory until it was ready to process all the RPC call.
                os.stat("/tmp/paddle.%d.port" % pid)
                return
            except os.error:
                retry_times -= 1

    def run_trainer(self, place, endpoints, trainer_id, trainers, is_dist=True):
        test_program, avg_cost, train_reader, test_reader, batch_acc, predict = get_model(
            batch_size=20)
        if is_dist:
            t = get_transpiler(trainer_id,
                               fluid.default_main_program(), endpoints,
                               trainers)
            trainer_prog = t.get_trainer_program()
        else:
            trainer_prog = fluid.default_main_program()

        startup_exe = fluid.Executor(place)
        startup_exe.run(fluid.default_startup_program())

        strategy = fluid.ExecutionStrategy()
        strategy.num_threads = 1
        strategy.allow_op_delay = False
        exe = fluid.ParallelExecutor(
            True,
            loss_name=avg_cost.name,
            exec_strategy=strategy,
            num_trainers=trainers,
            trainer_id=trainer_id)

        feed_var_list = [
            var for var in trainer_prog.global_block().vars.itervalues()
            if var.is_data
        ]

        feeder = fluid.DataFeeder(feed_var_list, place)
        reader_generator = train_reader()
        first_loss, = exe.run(fetch_list=[avg_cost.name])
        print(first_loss)
        for i in xrange(5):
            loss, = exe.run(fetch_list=[avg_cost.name])
        last_loss, = exe.run(fetch_list=[avg_cost.name])
        print(last_loss)


def main(role="pserver",
         endpoints="127.0.0.1:9123",
         trainer_id=0,
         current_endpoint="127.0.0.1:9123",
         trainers=1,
         is_dist=True):
    model = DistSeResneXt2x2()
    if role == "pserver":
        model.run_pserver(endpoints, trainers, current_endpoint, trainer_id)
    else:
        p = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        model.run_trainer(p, endpoints, trainer_id, trainers, is_dist)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(
            "Usage: python dist_se_resnext.py [pserver/trainer] [endpoints] [trainer_id] [current_endpoint] [trainers] [is_dist]"
        )
    role = sys.argv[1]
    endpoints = sys.argv[2]
    trainer_id = int(sys.argv[3])
    current_endpoint = sys.argv[4]
    trainers = int(sys.argv[5])
    is_dist = True if sys.argv[6] == "TRUE" else False
    main(
        role=role,
        endpoints=endpoints,
        trainer_id=trainer_id,
        current_endpoint=current_endpoint,
        trainers=trainers,
        is_dist=is_dist)
