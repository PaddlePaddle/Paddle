# -*- coding: UTF-8 -*-

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
from paddle.distributed import fleet
from paddle.fluid.framework import _test_eager_guard

from paddle.distributed.fleet.utils.internal_storage import GradStorage
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.sharding_optimizer_stage2 import ShardingOptimizerStage2

base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4

epoch = 100
batch_size = 32
class_dim = 102


class MLP(fluid.Layer):

    def __init__(self, param_attr=None, bias_attr=None):
        super(MLP, self).__init__()

        self._linear1 = Linear(10, 10)
        self._linear2 = Linear(10, 10)

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        return y


def reader_decorator():

    def __reader__():
        for _ in range(100):
            img = np.random.rand(10).astype('float32')
            label = np.ones(1).astype('int64')
            yield img, label

    return __reader__


def optimizer_setting(parameter_list=None):
    optimizer = paddle.optimizer.Momentum(
        learning_rate=base_lr,
        momentum=momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(l2_decay),
        parameters=parameter_list)
    return optimizer


def train_mlp():
    fleet.init(is_collective=True)
    group = paddle.distributed.new_group([0, 1])

    mlp = MLP()

    optimizer = optimizer_setting(parameter_list=mlp.parameters())
    oss_optimizer = ShardingOptimizerStage2(params=mlp.parameters(),
                                            optim=optimizer,
                                            group=group)
    # cover grad_storage code
    trainable_param2align = dict()
    for p in mlp.parameters():
        trainable_param2align[p.name] = 0
    grad_storage = GradStorage(10000,
                               dtype=paddle.float32,
                               device="gpu",
                               destination=0,
                               parm2align=trainable_param2align)
    for p in mlp.parameters():
        grad_storage.can_add_grad_view(p, trainable_param2align[p.name])
        grad_storage.add_grad(p, trainable_param2align[p.name])
    grad_storage.manumal_relase()
    grad_storage.rebuild()
    grad_storage.reset_checked_in()

    train_reader = paddle.batch(reader_decorator(),
                                batch_size=batch_size,
                                drop_last=True)

    train_loader = paddle.io.DataLoader.from_generator(capacity=32,
                                                       use_double_buffer=True,
                                                       iterable=True,
                                                       return_list=True,
                                                       use_multiprocess=True)
    train_loader.set_sample_list_generator(train_reader)

    for eop in range(epoch):
        mlp.train()

        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True

            out = mlp(img)
            loss = paddle.nn.functional.cross_entropy(input=out, label=label)
            avg_loss = paddle.mean(x=loss)
            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

            dy_out = avg_loss.numpy()

            avg_loss.backward()
            oss_optimizer.step()

            # oss_optimizer clear cache
            oss_optimizer._clear_cache()

            # check optimizer.minimize() error
            try:
                oss_optimizer.minimize()
            except:
                print(
                    "====== Find sharding_stage2_optimizer.minimize() error ======"
                )
            return


if __name__ == '__main__':
    with _test_eager_guard():
        pass
    train_mlp()
