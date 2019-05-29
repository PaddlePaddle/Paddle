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

import traceback
import math
import collections

import six
import unittest
import numpy as np

import gc

gc.set_debug(gc.DEBUG_COLLECTABLE)

import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distributed_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig


class FleetTest(unittest.TestCase):
    def setUp(self):
        self.trainer_id = 0
        self.trainers = 2
        self.pservers = 2
        # NOTE: we do not actually bind this port
        self.pserver_eps = ["127.0.0.1:6174", "127.0.0.1:6175"]
        self.pserver1_ep = "127.0.0.1:6174"
        self.pserver2_ep = "127.0.0.1:6175"
        self.sync_mode = True
        self.transpiler = None

    def get_current_role(self, current_id, training_role):
        if training_role == "TRAINER":
            role = role_maker.UserDefinedRoleMaker(
                current_id=current_id,
                role=role_maker.Role.WORKER,
                worker_num=self.trainers,
                server_endpoints=self.pserver_eps)
        else:
            role = role_maker.UserDefinedRoleMaker(
                current_id=current_id,
                role=role_maker.Role.WORKER,
                worker_num=self.trainers,
                server_endpoints=self.pserver_eps)
        return role

    def get_reader(self):
        x = np.random.randint(1000, size=np.random.randint(20))
        y = np.random.random_sample()
        return x, y

    def net_conf(self):
        x = fluid.layers.data(name='x', shape=[1], lod_level=1, dtype='int64')
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')

        embed = fluid.layers.embedding(input=x, is_sparse=True, size=[1000, 8])
        cnn = fluid.layers.sequence_pool(input=embed, pool_type='sum')

        y_predict = fluid.layers.fc(input=cnn,
                                    size=1000,
                                    act=None,
                                    param_attr=fluid.ParamAttr(name='fc_w'),
                                    bias_attr=fluid.ParamAttr(name='fc_b'))

        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)
        sgd_optimizer.minimize(avg_cost)

    def _run_instance(self):
        main = fluid.Program()
        main.random_seed = 1
        with fluid.program_guard(main):
            self.net_conf()
        self.origin_prog = main.clone()
        return main

    def transpiler_test_impl(self):
        pass

    def test_transpiler(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                self.transpiler_test_impl()
        # NOTE: run gc.collect to eliminate pybind side objects to
        # prevent random double-deallocate when inherited in python.
        del self.transpiler
        del main
        del startup
        gc.collect()
