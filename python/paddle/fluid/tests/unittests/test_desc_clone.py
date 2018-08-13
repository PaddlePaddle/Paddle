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
import signal
import collections

from paddle.fluid.transpiler.details import program_to_code

SEED = 1
DTYPE = "float32"
paddle.dataset.mnist.fetch()


# random seed must set before configuring the network.
# fluid.default_startup_program().random_seed = SEED
def cnn_model(data):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=data,
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

    # TODO(dzhwinter) : refine the initializer and random seed settting
    SIZE = 10
    input_shape = conv_pool_2.shape
    param_shape = [reduce(lambda a, b: a * b, input_shape[1:], 1)] + [SIZE]
    scale = (2.0 / (param_shape[0]**2 * SIZE))**0.5

    predict = fluid.layers.fc(
        input=conv_pool_2,
        size=SIZE,
        act="softmax",
        param_attr=fluid.param_attr.ParamAttr(
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=scale)))
    return predict


def get_model(batch_size):
    # Input data
    images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype=DTYPE)
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # Train program
    predict = cnn_model(images)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # Evaluator
    batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(
        input=predict, label=label, total=batch_size_tensor)

    inference_program = fluid.default_main_program().clone()
    # Optimization
    opt = fluid.optimizer.AdamOptimizer(
        learning_rate=0.001, beta1=0.9, beta2=0.999)

    # Reader
    train_reader = paddle.batch(
        paddle.dataset.mnist.train(), batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=batch_size)
    opt.minimize(avg_cost)
    return inference_program, avg_cost, train_reader, test_reader, batch_acc, predict


def get_transpiler(trainer_id, main_program, pserver_endpoints, trainers):
    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id=trainer_id,
        program=main_program,
        pservers=pserver_endpoints,
        trainers=trainers)
    return t


def operator_equal(a, b):
    for k, v in a.__dict__.iteritems():
        if isinstance(v, fluid.framework.Program) or \
                isinstance(v, fluid.framework.Block):
            continue

        elif isinstance(v, core.OpDesc):
            if v.serialize_to_string() != b.__dict__[k].serialize_to_string():
                raise ValueError("In operator_equal not equal:{0}\n".format(k))

        elif isinstance(v, collections.OrderedDict):
            v0 = sorted(v.iteritems(), key=lambda x: x[0])
            v1 = sorted(b.__dict__[k].iteritems(), key=lambda x: x[0])

            if v0 != v1:
                raise ValueError("In operator_equal not equal:{0}\n".format(k))

        elif (v != b.__dict__[k]):
            raise ValueError("In operator_equal not equal:{0}\n".format(k))

    return True


def block_equal(a, b):
    for k, v in a.__dict__.iteritems():
        if isinstance(v, core.ProgramDesc) or isinstance(
                v, fluid.framework.Program) or isinstance(v, core.BlockDesc):
            continue

        elif k == "ops":
            for i in range(0, len(a.ops)):
                if not operator_equal(a.ops[i], b.ops[i]):
                    raise ValueError("In block_equal not equal:{0}\n".format(k))
            assert (len(a.ops) == len(b.ops))

        elif isinstance(v, collections.OrderedDict):
            v0 = sorted(v.iteritems(), key=lambda x: x[0])
            v1 = sorted(b.__dict__[k].iteritems(), key=lambda x: x[0])

            if v0 != v1:
                raise ValueError("In block_equal not equal:{0}\n".format(k))

        elif (v != b.__dict__[k]):
            raise ValueError("In block_equal not equal:{0}\n".format(k))

    return True


def program_equal(a, b):
    for k, v in a.__dict__.iteritems():
        if isinstance(v, core.ProgramDesc):
            continue

        elif k == 'blocks':
            for i in range(0, len(a.blocks)):
                if not block_equal(a.blocks[i], b.blocks[i]):
                    raise ValueError("In operator_equal not equal:{0}\n".format(
                        k))
                    return False
            assert (len(a.blocks) == len(b.blocks))

        elif (v != b.__dict__[k]):
            raise ValueError("In program_equal not equal:{0}\n".format(k))

    return True


class TestDistMnist(unittest.TestCase):
    def test_desc_clone(self):
        get_model(batch_size=20)

        pserver_endpoints = "127.0.0.1:9123"
        trainers = 1
        current_endpoint = "127.0.0.1:9123"
        t = get_transpiler(0,
                           fluid.default_main_program(), pserver_endpoints,
                           trainers)

        pserver_prog = t.get_pserver_program(current_endpoint)
        startup_prog = t.get_startup_program(current_endpoint, pserver_prog)
        main = pserver_prog.clone()
        startup = startup_prog.clone()

        self.assertTrue(program_equal(main, pserver_prog))
        self.assertTrue(program_equal(startup, startup_prog))

        #program_to_code(startup)


if __name__ == "__main__":
    unittest.main()
