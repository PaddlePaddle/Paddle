# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import random

import numpy as np
from custom_setup_op_relu_model_static_multidevices import custom_relu

import paddle
import paddle.vision.transforms as T
from paddle import nn
from paddle.distributed import fleet

batch_size = 32


class Net(nn.Layer):
    """
    A simple exmaple for Regression Model.
    """

    def __init__(self, in_dim, out_dim, use_custom_op=False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.relu_act = custom_relu if use_custom_op else nn.functional.relu

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu_act(out)
        out = self.fc2(out)
        out = self.relu_act(out)

        out = paddle.mean(out, axis=-1)

        return out


def get_program(args):
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        x = paddle.static.data(
            shape=[batch_size, 1, 28, 28], name='x', dtype='float32'
        )
        x = paddle.flatten(x, start_axis=1)
        y = paddle.static.data(shape=[batch_size, 1], name='y', dtype='int64')
        y = paddle.cast(y, dtype='float32')

        net = Net(784, 10, args.use_custom_op)
        out = net(x)

        loss = nn.functional.mse_loss(out, y)
        if args.train_mode:
            sgd = paddle.optimizer.SGD(learning_rate=0.01)
            opt = fleet.distributed_optimizer(sgd)
            opt.minimize(loss)
    return main_program, startup_program, loss


def get_dataloader(mode='train'):
    transform = T.Compose(
        [
            T.Normalize(
                mean=[127.5],
                std=[127.5],
            ),
        ]
    )
    train_dataset = paddle.vision.datasets.MNIST(mode=mode, transform=transform)
    sampler = paddle.io.DistributedBatchSampler(
        train_dataset, shuffle=False, drop_last=True, batch_size=batch_size
    )
    train_loader = paddle.io.DataLoader(train_dataset, batch_sampler=sampler)
    return train_loader


def train(args):
    main_program, startup_program, loss = get_program(args)
    exe = paddle.static.Executor()
    exe.run(startup_program)

    losses = []
    for x_data, y_data in get_dataloader():
        res = exe.run(
            main_program,
            feed={'x': x_data, 'y': y_data},
            fetch_list=[loss],
        )
        losses.append(res)
    losses = np.array(losses)
    rank = paddle.distributed.get_rank()
    np.save(
        os.path.join(
            args.output_dir, 'train_{}_{}.npy'.format(rank, args.use_custom_op)
        ),
        losses,
    )
    if rank != 0:
        model_path = os.path.join(args.model_dir, str(args.use_custom_op))
        paddle.static.save(main_program, model_path)


def eval(args):
    main_program, startup_program, loss = get_program(args)
    exe = paddle.static.Executor()
    exe.run(startup_program)
    model_path = os.path.join(args.model_dir, str(args.use_custom_op))
    paddle.static.load(main_program, model_path, exe)

    losses = []
    for x_data, y_data in get_dataloader(mode='test'):
        res = exe.run(
            main_program,
            feed={'x': x_data, 'y': y_data},
            fetch_list=[loss],
        )
        losses.append(res)
    losses = np.array(losses)
    rank = paddle.distributed.get_rank()
    np.save(
        os.path.join(
            args.output_dir, 'eval_{}_{}.npy'.format(rank, args.use_custom_op)
        ),
        losses,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--use_custom_op', action='store_true')
    parser.add_argument('--train_mode', action='store_true')

    args = parser.parse_args()
    paddle.enable_static()
    paddle.seed(0)
    np.random.seed(0)
    random.seed(0)

    fleet.init()
    if args.train_mode:
        train(args)
    else:
        eval(args)
