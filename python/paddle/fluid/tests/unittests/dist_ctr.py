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
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from paddle.fluid import core
import unittest
from multiprocessing import Process
import os
import signal
from test_dist_base import TestDistRunnerBase, runtime_main
import dist_ctr_reader

IS_SPARSE = True

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


def prepare_data(dir):
    """
    download data and extract to certain dir
    """
    print("prepare_data")
    pass


class TestDistCTR2x2(TestDistRunnerBase):
    def get_model(self, batch_size=2):
        prepare_data("")
        dnn_input_dim, lr_input_dim = dist_ctr_reader.load_data_meta(
            "ctr_data/data.meta.txt")
        """ network definition """
        emb_lr = 5
        dnn_data = fluid.layers.data(
            name="data1",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
            append_batch_size=False)
        lr_data = fluid.layers.data(
            name="data2",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
            append_batch_size=False)
        label = fluid.layers.data(
            name="click",
            shape=[-1, 1],
            dtype="int64",
            lod_level=0,
            append_batch_size=False)

        # build dnn model
        dnn_layer_dims = [128, 64, 32, 1]
        dnn_embedding = fluid.layers.embedding(
            is_distributed=False,
            input=dnn_data,
            size=[dnn_input_dim, dnn_layer_dims[0]],
            param_attr=fluid.ParamAttr(name="deep_embedding"),
            is_sparse=IS_SPARSE)
        dnn_out = dnn_embedding
        for i, dim in enumerate(dnn_layer_dims[1:]):
            fc = fluid.layers.fc(input=dnn_out,
                                 size=dim,
                                 act="relu",
                                 name='dnn-fc-%d' % i)
            dnn_out = fc

        # build lr model
        lr_embbding = fluid.layers.embedding(
            is_distributed=True,
            input=lr_data,
            size=[lr_input_dim, 1],
            param_attr=fluid.ParamAttr(name="wide_embedding"),
            is_sparse=IS_SPARSE)

        merge_layer = fluid.layers.concat(input=[dnn_out, lr_embbding], aixs=1)

        predict = fluid.layers.fc(input=merge_layer, size=2, act='softmax')
        acc = fluid.layers.accuracy(input=predict, label=label)
        auc_var, auc_states = fluid.layers.auc(input=predict, label=label)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(x=cost)

        inference_program = paddle.fluid.default_main_program().clone()

        sgd_optimizer = fluid.optimizer.Adam(learning_rate=0.0001)
        sgd_optimizer.minimize(avg_cost)

        dataset = dist_ctr_reader.Dataset()
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                dataset.train("ctr_data/train.txt"), buf_size=500),
            batch_size=batch_size)
        test_reader = paddle.batch(
            dataset.test("ctr_data/test.txt"), batch_size=batch_size)

        return inference_program, avg_cost, train_reader, test_reader, None, predict


if __name__ == "__main__":
    runtime_main(TestDistCTR2x2)
    # dataset = dist_ctr_reader.Dataset()
    # for data in dataset.train("ctr_data/train.txt")():
    #     print(data)
