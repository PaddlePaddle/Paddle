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
import types
import numpy as np

from plsc import Entry
from plsc.models import BaseModel
from plsc.version import plsc_version
import paddle
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy

seed = 100
DATA_DIM = 28
BATCH_SIZE = 1


class SimpleModel(BaseModel):
    def __init__(self, emb_dim=2):
        super(SimpleModel, self).__init__()
        self.emb_dim = emb_dim

    def build_network(self, input, label, is_train=True):
        return fluid.layers.reshape(input, [-1, DATA_DIM * DATA_DIM])


def build_program(self, is_train=True, use_parallel_test=False):
    model_name = self.model_name
    assert not (is_train and use_parallel_test), \
         "is_train and use_parallel_test cannot be set simultaneously."
    trainer_id = self.trainer_id
    num_trainers = self.num_trainers

    image_shape = [28, 28]
    main_program = self.train_program if is_train else self.test_program
    startup_program = self.startup_program
    model = self.model
    version = plsc_version.split('.')
    version = [int(v) for v in version]
    use_rank_info = version[0] > 0 or version[1] > 1 or version[2] > 0
    np.random.seed(seed)
    arr = np.random.normal(size=(self.emb_dim, self.num_classes))
    if 'dist' in self.loss_type:
        per_trainer = self.num_classes // 2
        start = trainer_id * per_trainer
        end = (trainer_id + 1) * per_trainer
        arr = arr[:, start:end]
    param_attr = fluid.param_attr.ParamAttr(
        initializer=fluid.initializer.NumpyArrayInitializer(arr))
    with fluid.program_guard(main_program, startup_program):
        with fluid.unique_name.guard():
            image = fluid.layers.data(
                name='image', shape=image_shape, dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')

            if use_rank_info:
                emb, loss, prob = model.get_output(
                    input=image,
                    label=label,
                    is_train=is_train,
                    num_ranks=self.num_trainers,
                    rank_id=self.trainer_id,
                    num_classes=self.num_classes,
                    loss_type=self.loss_type,
                    param_attr=param_attr,
                    margin=self.margin,
                    scale=self.scale)
            else:
                emb, loss, prob = model.get_output(
                    input=image,
                    label=label,
                    is_train=is_train,
                    num_classes=self.num_classes,
                    loss_type=self.loss_type,
                    param_attr=param_attr,
                    margin=self.margin,
                    scale=self.scale)

            optimizer = None
            if is_train:
                # initialize optimizer
                optimizer = self._get_optimizer()
                dist_optimizer = self.fleet.distributed_optimizer(
                    optimizer, strategy=self.strategy)
                dist_optimizer.minimize(loss)
                if "dist" in self.loss_type:
                    optimizer = optimizer._optimizer
            else:
                raise ValueError("Only support training now.")
    return loss


def train(self):
    self._check()

    trainer_id = self.trainer_id
    num_trainers = self.num_trainers
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)
    strategy = DistributedStrategy()
    strategy.mode = "collective"
    strategy.collective_mode = "grad_allreduce"
    self.fleet = fleet
    self.strategy = strategy

    loss = self.build_program(True, False)
    origin_prog = fleet._origin_program
    train_prog = fleet.main_program

    gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
    place = fluid.CUDAPlace(gpu_id)
    exe = fluid.Executor(place)
    exe.run(self.startup_program)

    train_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=self.train_batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=['image', 'label'], program=origin_prog)
    fetch_list = [loss.name]

    loss = []
    for batch_id, data in enumerate(train_reader()):
        if batch_id >= 1:
            break
        local_loss, = exe.run(train_prog,
                              feed=feeder.feed(data),
                              fetch_list=fetch_list,
                              use_program_cache=True)
        loss.append(local_loss)
    return loss


if __name__ == "__main__":
    entry = Entry()
    entry.emb_dim = DATA_DIM * DATA_DIM
    entry.build_program = types.MethodType(build_program, entry)
    entry.train = types.MethodType(train, entry)
    entry.set_model(SimpleModel())
    entry.set_loss_type('arcface')
    entry.set_train_batch_size(BATCH_SIZE)
    entry.set_class_num(10)
    loss_arcface = entry.train()

    entry = Entry()
    entry.emb_dim = DATA_DIM * DATA_DIM
    entry.build_program = types.MethodType(build_program, entry)
    entry.train = types.MethodType(train, entry)
    entry.set_model(SimpleModel())
    entry.set_loss_type('dist_arcface')
    entry.set_train_batch_size(BATCH_SIZE)
    entry.set_class_num(10)
    loss_distarcface = entry.train()

    entry = Entry()
    entry.emb_dim = DATA_DIM * DATA_DIM
    entry.build_program = types.MethodType(build_program, entry)
    entry.train = types.MethodType(train, entry)
    entry.set_model(SimpleModel())
    entry.set_loss_type('softmax')
    entry.set_train_batch_size(BATCH_SIZE)
    entry.set_class_num(10)
    loss_softmax = entry.train()

    entry = Entry()
    entry.emb_dim = DATA_DIM * DATA_DIM
    entry.build_program = types.MethodType(build_program, entry)
    entry.train = types.MethodType(train, entry)
    entry.set_model(SimpleModel())
    entry.set_loss_type('dist_softmax')
    entry.set_train_batch_size(BATCH_SIZE)
    entry.set_class_num(10)
    loss_distsoftmax = entry.train()

    print("loss_arcface: ", loss_arcface)
    print("loss_distarcface: ", loss_distarcface)
    print("loss_softmax: ", loss_softmax)
    print("loss_distsoftmax: ", loss_distsoftmax)
