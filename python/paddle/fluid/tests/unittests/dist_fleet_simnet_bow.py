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
import argparse
import time
import math
import random
import shutil
import tempfile

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from paddle.fluid import core
import unittest
from multiprocessing import Process
import os
import signal
from functools import reduce
from test_dist_fleet_base import runtime_main, FleetDistRunnerBase

paddle.enable_static()

DTYPE = "int64"
DATA_URL = 'http://paddle-dist-ce-data.bj.bcebos.com/simnet.train.1000'
DATA_MD5 = '24e49366eb0611c552667989de2f57d5'

# For Net
base_lr = 0.2
emb_lr = base_lr * 3
dict_dim = 1500
emb_dim = 128
hid_dim = 128
margin = 0.1
sample_rate = 1

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


def fake_simnet_reader():
    def reader():
        for _ in range(1000):
            q = np.random.random_integers(0, 1500 - 1, size=1).tolist()
            label = np.random.random_integers(0, 1, size=1).tolist()
            pt = np.random.random_integers(0, 1500 - 1, size=1).tolist()
            nt = np.random.random_integers(0, 1500 - 1, size=1).tolist()
            yield [q, label, pt, nt]

    return reader


def get_acc(cos_q_nt, cos_q_pt, batch_size):
    cond = fluid.layers.less_than(cos_q_nt, cos_q_pt)
    cond = fluid.layers.cast(cond, dtype='float64')
    cond_3 = fluid.layers.reduce_sum(cond)
    acc = fluid.layers.elementwise_div(
        cond_3,
        fluid.layers.fill_constant(
            shape=[1], value=batch_size * 1.0, dtype='float64'),
        name="simnet_acc")
    return acc


def get_loss(cos_q_pt, cos_q_nt):
    loss_op1 = fluid.layers.elementwise_sub(
        fluid.layers.fill_constant_batch_size_like(
            input=cos_q_pt, shape=[-1, 1], value=margin, dtype='float32'),
        cos_q_pt)
    loss_op2 = fluid.layers.elementwise_add(loss_op1, cos_q_nt)
    loss_op3 = fluid.layers.elementwise_max(
        fluid.layers.fill_constant_batch_size_like(
            input=loss_op2, shape=[-1, 1], value=0.0, dtype='float32'),
        loss_op2)
    avg_cost = fluid.layers.mean(loss_op3)
    return avg_cost


def train_network(batch_size,
                  is_distributed=False,
                  is_sparse=False,
                  is_self_contained_lr=False,
                  is_pyreader=False):
    # query
    q = fluid.layers.data(
        name="query_ids", shape=[1], dtype="int64", lod_level=1)
    # label data
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    # pt
    pt = fluid.layers.data(
        name="pos_title_ids", shape=[1], dtype="int64", lod_level=1)
    # nt
    nt = fluid.layers.data(
        name="neg_title_ids", shape=[1], dtype="int64", lod_level=1)

    datas = [q, label, pt, nt]

    reader = None
    if is_pyreader:
        reader = fluid.io.PyReader(
            feed_list=datas,
            capacity=64,
            iterable=False,
            use_double_buffer=False)

    # embedding
    q_emb = fluid.embedding(
        input=q,
        is_distributed=is_distributed,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01), name="__emb__"),
        is_sparse=is_sparse)
    q_emb = fluid.layers.reshape(q_emb, [-1, emb_dim])
    # vsum
    q_sum = fluid.layers.sequence_pool(input=q_emb, pool_type='sum')
    q_ss = fluid.layers.softsign(q_sum)
    # fc layer after conv
    q_fc = fluid.layers.fc(
        input=q_ss,
        size=hid_dim,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01),
            name="__q_fc__",
            learning_rate=base_lr), )

    # embedding
    pt_emb = fluid.embedding(
        input=pt,
        is_distributed=is_distributed,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01),
            name="__emb__",
            learning_rate=emb_lr),
        is_sparse=is_sparse)
    pt_emb = fluid.layers.reshape(pt_emb, [-1, emb_dim])
    # vsum
    pt_sum = fluid.layers.sequence_pool(input=pt_emb, pool_type='sum')
    pt_ss = fluid.layers.softsign(pt_sum)
    # fc layer
    pt_fc = fluid.layers.fc(
        input=pt_ss,
        size=hid_dim,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01), name="__fc__"),
        bias_attr=fluid.ParamAttr(name="__fc_b__"))

    # embedding
    nt_emb = fluid.embedding(
        input=nt,
        is_distributed=is_distributed,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01), name="__emb__"),
        is_sparse=is_sparse)
    nt_emb = fluid.layers.reshape(nt_emb, [-1, emb_dim])
    # vsum
    nt_sum = fluid.layers.sequence_pool(input=nt_emb, pool_type='sum')
    nt_ss = fluid.layers.softsign(nt_sum)
    # fc layer
    nt_fc = fluid.layers.fc(
        input=nt_ss,
        size=hid_dim,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01), name="__fc__"),
        bias_attr=fluid.ParamAttr(name="__fc_b__"))
    cos_q_pt = fluid.layers.cos_sim(q_fc, pt_fc)
    cos_q_nt = fluid.layers.cos_sim(q_fc, nt_fc)
    # loss
    avg_cost = get_loss(cos_q_pt, cos_q_nt)
    # acc
    acc = get_acc(cos_q_nt, cos_q_pt, batch_size)
    return avg_cost, acc, cos_q_pt, reader


class TestDistSimnetBow2x2(FleetDistRunnerBase):
    """
    For test SimnetBow model, use Fleet api
    """

    def net(self, args, batch_size=4, lr=0.01):
        avg_cost, _, predict, self.reader = \
            train_network(batch_size=batch_size, is_distributed=False,
                          is_sparse=True, is_self_contained_lr=False, is_pyreader=(args.reader == "pyreader"))
        self.avg_cost = avg_cost
        self.predict = predict

        return avg_cost

    def check_model_right(self, dirname):
        model_filename = os.path.join(dirname, "__model__")

        with open(model_filename, "rb") as f:
            program_desc_str = f.read()

        program = fluid.Program.parse_from_string(program_desc_str)
        with open(os.path.join(dirname, "__model__.proto"), "w") as wn:
            wn.write(str(program))

    def do_pyreader_training(self, fleet):
        """
        do training using dataset, using fetch handler to catch variable
        Args:
            fleet(Fleet api): the fleet object of Parameter Server, define distribute training role
        """

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        fleet.init_worker()
        batch_size = 4
        # reader
        train_reader = paddle.batch(fake_simnet_reader(), batch_size=batch_size)
        self.reader.decorate_sample_list_generator(train_reader)
        for epoch_id in range(1):
            self.reader.start()
            try:
                pass_start = time.time()
                while True:
                    loss_val = exe.run(program=fluid.default_main_program(),
                                       fetch_list=[self.avg_cost.name])
                    loss_val = np.mean(loss_val)
                    message = "TRAIN ---> pass: {} loss: {}\n".format(epoch_id,
                                                                      loss_val)
                    fleet.util.print_on_rank(message, 0)

                pass_time = time.time() - pass_start
            except fluid.core.EOFException:
                self.reader.reset()

    def do_dataset_training(self, fleet):
        pass


if __name__ == "__main__":
    runtime_main(TestDistSimnetBow2x2)
