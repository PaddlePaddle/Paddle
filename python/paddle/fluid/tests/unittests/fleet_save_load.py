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

import os
import sys
import signal
import subprocess
import argparse
import time
import math
import random
from multiprocessing import Process
from functools import reduce

import numpy as np
import pickle
import unittest
import six

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid import io
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
from test_dist_fleet_base import FleetDistRunnerBase, runtime_main, RUN_STEP
from paddle.fluid.incubate.fleet.utils.hdfs import HDFSClient

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


def get_optimizer(op="sgd"):
    if op.upper() == "sgd".upper():
        optimizer = fluid.optimizer.SGD(learning_rate=base_lr)
    elif op.upper() == "adam".upper():
        optimizer = fluid.optimizer.Adam(learning_rate=base_lr)
    else:
        optimizer = fluid.optimizer.SGD(learning_rate=base_lr)
    return optimizer


def train_network(batch_size,
                  is_distributed=False,
                  is_sparse=False,
                  is_self_contained_lr=False):
    # query
    q = fluid.layers.data(
        name="query_ids", shape=[1], dtype="int64", lod_level=1)
    ## embedding
    q_emb = fluid.layers.embedding(
        input=q,
        is_distributed=is_distributed,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01),
            name="__emb__",
            learning_rate=emb_lr) if is_self_contained_lr else fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01),
                name="__emb__"),
        is_sparse=is_sparse)
    ## vsum
    q_sum = fluid.layers.sequence_pool(input=q_emb, pool_type='sum')
    q_ss = fluid.layers.softsign(q_sum)
    ## fc layer after conv
    q_fc = fluid.layers.fc(
        input=q_ss,
        size=hid_dim,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01),
            name="__q_fc__",
            learning_rate=base_lr))
    # label data
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    # pt
    pt = fluid.layers.data(
        name="pos_title_ids", shape=[1], dtype="int64", lod_level=1)
    ## embedding
    pt_emb = fluid.layers.embedding(
        input=pt,
        is_distributed=is_distributed,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01),
            name="__emb__",
            learning_rate=emb_lr) if is_self_contained_lr else fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01),
                name="__emb__"),
        is_sparse=is_sparse)
    ## vsum
    pt_sum = fluid.layers.sequence_pool(input=pt_emb, pool_type='sum')
    pt_ss = fluid.layers.softsign(pt_sum)
    ## fc layer
    pt_fc = fluid.layers.fc(
        input=pt_ss,
        size=hid_dim,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01),
            name="__fc__",
            learning_rate=base_lr),
        bias_attr=fluid.ParamAttr(name="__fc_b__"))
    # nt
    nt = fluid.layers.data(
        name="neg_title_ids", shape=[1], dtype="int64", lod_level=1)
    ## embedding
    nt_emb = fluid.layers.embedding(
        input=nt,
        is_distributed=is_distributed,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01),
            name="__emb__",
            learning_rate=emb_lr) if is_self_contained_lr else fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01),
                name="__emb__"),
        is_sparse=is_sparse)
    ## vsum
    nt_sum = fluid.layers.sequence_pool(input=nt_emb, pool_type='sum')
    nt_ss = fluid.layers.softsign(nt_sum)
    ## fc layer
    nt_fc = fluid.layers.fc(
        input=nt_ss,
        size=hid_dim,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01),
            name="__fc__",
            learning_rate=base_lr),
        bias_attr=fluid.ParamAttr(name="__fc_b__"))
    cos_q_pt = fluid.layers.cos_sim(q_fc, pt_fc)
    cos_q_nt = fluid.layers.cos_sim(q_fc, nt_fc)
    # loss
    avg_cost = get_loss(cos_q_pt, cos_q_nt)
    # acc
    acc = get_acc(cos_q_nt, cos_q_pt, batch_size)
    return [avg_cost, acc, cos_q_pt]


def combination(x, y):
    res = [[[xi, yi] for yi in y] for xi in x]
    return res[0]


def get_one_data(file_list):
    for file in file_list:
        contents = []
        with open(file, "r") as fin:
            for i in fin:
                contents.append(i.strip())
            for index, q in enumerate(contents):
                try:
                    one_data = [[int(j) for j in i.split(" ")]
                                for i in q.split(";")[:-1]]
                    if one_data[1][0] + one_data[1][1] != len(one_data) - 3:
                        q = fin.readline()
                        continue
                    tmp = combination(one_data[3:3 + one_data[1][0]],
                                      one_data[3 + one_data[1][0]:])
                except Exception as e:
                    continue

                for each in tmp:
                    yield [one_data[2], 0, each[0], each[1]]


def get_batch_reader(file_list, batch_size):
    def batch_reader():
        res = []
        for i in get_one_data(file_list):
            if random.random() <= sample_rate:
                res.append(i)
            if len(res) >= batch_size:
                yield res
                res = []

    return batch_reader


def get_train_reader(batch_size):
    # The training data set.
    train_file = os.path.join(paddle.dataset.common.DATA_HOME, "simnet",
                              "train")
    train_reader = get_batch_reader([train_file], batch_size)
    train_feed = ["query_ids", "pos_title_ids", "neg_title_ids"]
    return train_reader, train_feed


class TestFleetSaveLoad2x2(FleetDistRunnerBase):
    def net(self, batch_size=4, lr=0.01):
        # Train program
        avg_cost, acc, predict = \
            train_network(batch_size,
                          bool(int(os.environ["IS_DISTRIBUTED"])),
                          bool(int(os.environ["IS_SPARSE"])),
                          bool(int(os.environ["IS_SELF_CONTAINED_LR"])))

        self.batch_size = batch_size
        self.avg_cost = avg_cost
        self.predict = predict
        return avg_cost

    def run_pserver(self, args):
        if args.role.upper() != "PSERVER":
            raise ValueError("args role must be PSERVER")

        role = role_maker.UserDefinedRoleMaker(
            current_id=args.current_id,
            role=role_maker.Role.SERVER,
            worker_num=args.trainers,
            server_endpoints=args.endpoints.split(","))

        hdfs_trainer_client = None
        server_hadoop_configs = None
        trainer_hadoop_configs = dict()
        if args.hadoop_home != "":
            hdfs_trainer_client = HDFSClient(args.hadoop_home,
                                             trainer_hadoop_configs)
            server_hadoop_configs = {}
            server_hadoop_configs['127.0.0.1'] = {
                'HADOOP_HOME': args.hadoop_home,
                'configs': trainer_hadoop_configs
            }

        fleet.init(role, hdfs_trainer_client, server_hadoop_configs)

        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = args.sync_mode

        avg_cost = self.net()

        opt = os.getenv('OPTIMIZER', 'sgd')
        optimizer = get_optimizer(opt)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        need_load = bool(int(os.getenv("LOAD", "0")))
        model_dir = os.getenv("MODEL_DIR", "")

        if need_load and model_dir:
            fleet.init_server(model_dir)
        else:
            fleet.init_server()
        fleet.run_server()

    def run_trainer(self, args):
        if args.role.upper() != "TRAINER":
            raise ValueError("args role must be TRAINER")

        role = role_maker.UserDefinedRoleMaker(
            current_id=args.current_id,
            role=role_maker.Role.WORKER,
            worker_num=args.trainers,
            server_endpoints=args.endpoints.split(","))

        hdfs_trainer_client = None
        server_hadoop_configs = None
        trainer_hadoop_configs = dict()
        if args.hadoop_home != "":
            hdfs_trainer_client = HDFSClient(args.hadoop_home,
                                             trainer_hadoop_configs)
            server_hadoop_configs = {}
            server_hadoop_configs = {
                'HADOOP_HOME': args.hadoop_home,
                'configs': trainer_hadoop_configs
            }

        fleet.init(role, hdfs_trainer_client, server_hadoop_configs)

        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = args.sync_mode

        avg_cost = self.net()

        opt = os.getenv('OPTIMIZER', 'sgd')
        optimizer = get_optimizer(opt)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)
        out = self.do_training(fleet)

    def do_training(self, fleet):
        with open('/tmp/trainer_' + str(fleet.worker_index()) + '.train',
                  'w') as f:
            f.write(str(fleet.main_program))
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fleet.startup_program)

        strategy = fluid.ExecutionStrategy()
        strategy.num_threads = 1

        build_stra = fluid.BuildStrategy()
        build_stra.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

        compiled_prog = fluid.compiler.CompiledProgram(
            fleet.main_program).with_data_parallel(
                loss_name=self.avg_cost.name,
                build_strategy=build_stra,
                exec_strategy=strategy)

        feed_var_list = [
            var for var in fleet.main_program.global_block().vars.values()
            if var.is_data
        ]
        feeder = fluid.DataFeeder(feed_var_list, place)
        reader_generator, feed_var_names = get_train_reader(
            batch_size=self.batch_size)

        def get_data():
            origin_batch = next(reader_generator())
            new_batch = []
            for offset, item in enumerate(origin_batch):
                if offset % 2 == fleet.worker_index():
                    new_batch.append(item)
            return new_batch

        need_save = bool(int(os.getenv("SAVE", "0")))
        model_dir = os.getenv("MODEL_DIR", "")

        print("RUN_STEP = {}".format(RUN_STEP))
        for _ in six.moves.xrange(RUN_STEP):
            loss, = exe.run(program=compiled_prog,
                            fetch_list=[self.avg_cost.name],
                            feed=feeder.feed(get_data()))
            print("loss = {} ".format(loss))
        if need_save and model_dir and fleet.worker_index() == 0:
            fleet.save_inference_model(exe, model_dir, feed_var_names,
                                       self.predict)


if __name__ == "__main__":
    paddle.dataset.common.download(DATA_URL, 'simnet', DATA_MD5, "train")
    runtime_main(TestFleetSaveLoad2x2)
