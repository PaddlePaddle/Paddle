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

from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.trainer_factory import TrainerFactory
from paddle.fluid.trainer_desc import MultiTrainer, DistMultiTrainer

input_x = fluid.layers.data(name="input_x", shape=[1], lod_level=1)
label = fluid.layers.data(name="label", shape=[1], lod_level=0)
emb = fluid.layers.embedding(input=input_x, size=[10000, 10], is_sparse=True)
bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
bow_tanh = fluid.layers.tanh(bow)
fc_1 = fluid.layers.fc(input=bow_tanh, size=100, act='tanh')
cost = fluid.layers.cross_entropy(input=fc_1, label=label)
avg_cost = fluid.layers.mean(x=cost)
program = fluid.default_main_program()
trainer = TrainerFactory()._create_trainer(program._fleet_opt)
trainer._set_program(program)
assert isinstance(trainer, MultiTrainer)

from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
from paddle.fluid.incubate.fleet.parameter_server.pslib import DownpourOptimizer

exe = fluid.Executor(place=fluid.CPUPlace())

sgd_optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
sgd_optimizer = fleet.distributed_optimizer(
    sgd_optimizer, strategy={"use_cvm": True})
sgd_optimizer.minimize(avg_cost)
fleet.init(exe)
dist_trainer = TrainerFactory()._create_trainer(program._fleet_opt)
dist_trainer._set_program(fluid.default_main_program())
assert isinstance(dist_trainer, DistMultiTrainer)
