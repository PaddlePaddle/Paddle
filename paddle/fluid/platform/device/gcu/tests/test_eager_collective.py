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

import copy
import random
import time

import numpy as np

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F

BATCH_SIZE = 64
BATCH_NUM = 4
EPOCH_NUM = 4

CLASS_NUM = 10

paddle.set_device('gcu')
paddle.seed(666)
np.random.seed(666)
random.seed(666)


# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        # np.random.seed(2023 + idx)
        # shape: image:[BATCH_SIZE, 1, 28, 28], label:[BATCH_SIZE, 1]
        image = np.random.random([1, 28, 28]).astype('float32')
        label = np.random.randint(0, CLASS_NUM, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class LeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2
        )
        # self.bn = paddle.nn.BatchNorm(num_channels=6,
        #                               moving_mean_name="moving_mean",
        #                               moving_variance_name="moving_var")
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=6, out_channels=16, kernel_size=5, stride=1
        )
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(
            in_features=16 * 5 * 5, out_features=120
        )
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


def build_data_loader():
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    sampler = paddle.io.BatchSampler(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
    )
    data_loader = paddle.io.DataLoader(dataset, batch_sampler=sampler)
    return data_loader


def get_dist_data(data, rank, dist_mini_batch):
    x = data[0][(rank * dist_mini_batch) : ((rank + 1) * dist_mini_batch)]
    y = data[1][(rank * dist_mini_batch) : ((rank + 1) * dist_mini_batch)]
    return [x, y]


def check_state_dict(state_dict, dist_state_dict):
    atol = 1e-05
    for key in sorted(state_dict):
        tensor = state_dict[key]
        dist_tensor = dist_state_dict[key]
        assert np.allclose(
            tensor, dist_tensor, atol=atol
        ), 'state_dict check failed, tensor name:{}, tensor:{}, vs dist tensor:{}'.format(
            key, tensor, dist_tensor
        )


def train_one_step(model, optim, epoch, batch_id, data, is_dist=False):
    start = time.time()
    x_data = data[0]
    y_data = data[1]
    predicts = model(x_data)
    loss = F.cross_entropy(predicts, y_data)
    acc = paddle.metric.accuracy(predicts, y_data)
    loss.backward()
    optim.step()
    optim.clear_grad()
    end = time.time()
    if dist.get_rank() == 0:
        print(
            '{} train epoch:{}, batch_id:{}, loss:{}, acc:{}, time:{}s'.format(
                ('Distributed' if is_dist else '  Single   '),
                epoch,
                batch_id,
                loss.numpy(),
                acc.numpy(),
                (end - start),
            )
        )
    state_dict = model.state_dict()
    return state_dict


def train(model, dist_model, data_loader, world_size):
    optim = paddle.optimizer.Adam(
        learning_rate=0.001,
        parameters=model.parameters(),
        use_multi_tensor=True,
    )
    dist_optim = paddle.optimizer.Adam(
        learning_rate=0.001,
        parameters=dist_model.parameters(),
        use_multi_tensor=True,
    )
    rank = dist.get_rank()
    dist_mini_batch = BATCH_SIZE // world_size
    for epoch in range(EPOCH_NUM):
        model.train()
        dist_model.train()
        for batch_id, data in enumerate(data_loader()):
            state_dict = train_one_step(model, optim, epoch, batch_id, data)
            dist_state_dict = train_one_step(
                dist_model,
                dist_optim,
                epoch,
                batch_id,
                get_dist_data(data, rank, dist_mini_batch),
                True,
            )
            check_state_dict(state_dict, dist_state_dict)
            if rank == 0:
                print(
                    'Train successfully, epoch:{}, step:{}'.format(
                        epoch, batch_id
                    )
                )


if __name__ == '__main__':
    start = time.time()
    dist.init_parallel_env()
    model = LeNet()
    model.to('gcu')

    dist_model = copy.deepcopy(model)
    dist_model.to('gcu')

    model = paddle.jit.to_static(model)
    dist_model = paddle.jit.to_static(dist_model)

    dist_model = paddle.DataParallel(dist_model)
    dist_model.load_dict(model.state_dict())
    check_state_dict(model.state_dict(), dist_model.state_dict())

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert (
        BATCH_SIZE % world_size == 0
    ), 'batch_size check failed, batch_size:{}, world_size:{}'.format(
        BATCH_SIZE, world_size
    )
    data_loader = build_data_loader()
    train(model, dist_model, data_loader, world_size=world_size)
    print(
        'Test eager collective successfully, world_size:{}, rank:{}, '
        'total time: {}s'.format(world_size, rank, (time.time() - start))
    )
