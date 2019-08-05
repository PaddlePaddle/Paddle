#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import shutil
import tempfile
import time

import paddle.fluid as fluid
import os
import sys

import numpy as np
from test_dist_fleet_base import runtime_main, FleetDistRunnerBase

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


class TestDistPServer(FleetDistRunnerBase):
    def net(self, batch_size=4, lr=0.01):
        hid_dim = 128
        label_dim = 2
        input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
        input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
        fc_1 = fluid.layers.fc(input=input_x, size=hid_dim, act='tanh')
        fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim, act='tanh')
        prediction = fluid.layers.fc(input=[fc_2],
                                     size=label_dim,
                                     act='softmax')
        cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
        avg_cost = fluid.layers.mean(x=cost)
        self.avg_cost = avg_cost
        return avg_cost

    def gen_data(self):
        return {
            "x": np.random.random(size=(128, 32)).astype('float32'),
            "y": np.random.randint(
                2, size=(128, 1)).astype('int64')
        }

    def do_training(self, fleet):
        fleet.init_worker()
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fleet.startup_program)
        step = 10
        cost_list = []
        for i in range(step):
            cost_val = exe.run(program=fleet.main_program,
                               feed=self.gen_data(),
                               fetch_list=[self.avg_cost.name])
            cost_list.append(cost_val)
        fleet.stop_worker()


if __name__ == "__main__":
    runtime_main(TestDistPServer)
