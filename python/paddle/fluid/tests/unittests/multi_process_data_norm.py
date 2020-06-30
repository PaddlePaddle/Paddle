# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import time
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.base import role_maker


def train():
    gpu_id = int(os.getenv("FLAGS_selected_gpus"))
    if os.name == 'nt':
        print(
            'Skip TestDataNormOpWithSyncStats because nccl is not supported on windows'
        )
        return
    x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=0)
    emb = layers.embedding(
        input=x,
        param_attr=fluid.ParamAttr(name="embx"),
        size=[10, 2],
        is_sparse=False)
    dn = layers.data_norm(
        input=emb,
        name="hehe",
        epsilon=1e-4,
        param_attr={"batch_size": 1e4,
                    "batch_sum": 1e5,
                    "batch_square": 1e4},
        summary_decay_rate=1,
        sync_stats=True)  #[-1,3]
    loss = layers.mean(dn)

    all_p = fluid.default_main_program().global_block().all_parameters()
    parameter_without_datanorm = []
    for e in all_p:
        if e.name.find("batch_size") != -1 or e.name.find(
                "batch_sq") != -1 or e.name.find("batch_sum") != -1:
            continue
        parameter_without_datanorm.append(e.name)

    dist_strategy = DistributedStrategy()
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)
    optimizer = fluid.optimizer.SGD(learning_rate=0.5)
    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
    optimizer.minimize(loss, parameter_list=parameter_without_datanorm)
    train_prog = fleet.main_program
    place = fluid.CUDAPlace(gpu_id)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    emb_t = fluid.global_scope().find_var("embx").get_tensor()
    para = np.ones((10, 2)).astype("float32")
    emb_t.set(para, place)

    batch_size = 1

    def reader():
        batch1 = np.ones((batch_size, 1)).astype("int64")
        #(batch_size, 1)).astype("int64").reshape(batch_size, 1, 1)
        yield batch1

    for data in reader():
        cost_val = exe.run(program=train_prog,
                           feed={'x': data},
                           fetch_list=[loss.name])
        batch_size = np.array(fluid.global_scope().find_var("hehe.batch_size")
                              .get_tensor())
        assert batch_size[0] == 10002
        b = np.array(fluid.global_scope().find_var("hehe.batch_sum").get_tensor(
        ))
        assert b[0] == 100002
        c = np.array(fluid.global_scope().find_var("hehe.batch_square_sum")
                     .get_tensor())
        assert c[0] == 10162


if __name__ == '__main__':
    train()
