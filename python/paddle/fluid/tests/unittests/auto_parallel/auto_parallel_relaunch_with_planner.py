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

import paddle
import paddle.static as static
from paddle.distributed import fleet


def train():
    from auto_parallel_relaunch_model import mlp_pretrain_forward
    from auto_parallel_relaunch_model import batch_generator_creator
    dist_strategy = fleet.DistributedStrategy()
    # init parallel optimizer
    dist_strategy.auto_search = True
    fleet.init(is_collective=True, strategy=dist_strategy)
    train_program = static.Program()
    start_program = static.Program()
    loss, train_program, start_program, loader = mlp_pretrain_forward(
        train_program, start_program)

    optimizer = paddle.fluid.optimizer.AdamOptimizer(
        learning_rate=0.00001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        grad_clip=None)

    optimizer = fleet.distributed_optimizer(optimizer)
    _, _, distributed_startup_program, distributed_main_program = optimizer.minimize(
        loss, start_program)

    places = static.cuda_places()
    loader.set_batch_generator(batch_generator_creator(), places=places)
    exe = paddle.static.Executor(places[0])
    exe.run(distributed_startup_program)

    for data in loader():
        exe.run(distributed_main_program, feed=data)


if __name__ == "__main__":
    train()
