#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
Distribute CTR model for test fleet api
"""

from __future__ import print_function

import shutil
import tempfile
import time

import paddle
import paddle.fluid as fluid
import os
import numpy as np

import ctr_dataset_reader
from test_dist_fleet_base import runtime_main, FleetDistRunnerBase
from dist_fleet_ctr import TestDistCTR2x2, fake_ctr_reader
from paddle.distributed.fleet.base.util_factory import fleet_util

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


class TestHeterPsCTR2x2(TestDistCTR2x2):
    def net(self, args, batch_size=4, lr=0.01):
        """
        network definition

        Args:
            batch_size(int): the size of mini-batch for training
            lr(float): learning rate of training
        Returns:
            avg_cost: LoDTensor of cost.
        """
        dnn_input_dim, lr_input_dim = int(1e5), int(1e5)

        dnn_data = fluid.layers.data(
            name="dnn_data",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
            append_batch_size=False)
        lr_data = fluid.layers.data(
            name="lr_data",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
            append_batch_size=False)
        label = fluid.layers.data(
            name="click",
            shape=[-1, 1],
            dtype="float32",
            lod_level=0,
            append_batch_size=False)

        datas = [dnn_data, lr_data, label]

        if args.reader == "pyreader":
            self.reader = fluid.io.PyReader(
                feed_list=datas,
                capacity=64,
                iterable=False,
                use_double_buffer=False)

        # build dnn model
        dnn_layer_dims = [128, 64, 32, 1]
        dnn_embedding = fluid.layers.embedding(
            is_distributed=False,
            input=dnn_data,
            size=[dnn_input_dim, dnn_layer_dims[0]],
            param_attr=fluid.ParamAttr(
                name="deep_embedding",
                initializer=fluid.initializer.Constant(value=0.01)),
            is_sparse=True)
        dnn_pool = fluid.layers.sequence_pool(
            input=dnn_embedding, pool_type="sum")
        dnn_out = dnn_pool

        with fluid.device_guard("gpu"):
            for i, dim in enumerate(dnn_layer_dims[1:]):
                fc = fluid.layers.fc(
                    input=dnn_out,
                    size=dim,
                    act="relu",
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Constant(value=0.01)),
                    name='dnn-fc-%d' % i)
                dnn_out = fc

            # build lr model
            lr_embbding = fluid.layers.embedding(
                is_distributed=False,
                input=lr_data,
                size=[lr_input_dim, 1],
                param_attr=fluid.ParamAttr(
                    name="wide_embedding",
                    initializer=fluid.initializer.Constant(value=0.01)),
                is_sparse=True)
            lr_pool = fluid.layers.sequence_pool(
                input=lr_embbding, pool_type="sum")

            merge_layer = fluid.layers.concat(input=[dnn_out, lr_pool], axis=1)
            label = fluid.layers.cast(label, dtype="int64")
            predict = fluid.layers.fc(input=merge_layer, size=2, act='softmax')
            acc = fluid.layers.accuracy(input=predict, label=label)

            auc_var, batch_auc_var, auc_states = fluid.layers.auc(input=predict,
                                                                  label=label)

            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(x=cost)

        self.feeds = datas
        self.train_file_path = ["fake1", "fake2"]
        self.avg_cost = avg_cost
        self.predict = predict

        return avg_cost


if __name__ == "__main__":
    runtime_main(TestHeterPsCTR2x2)
