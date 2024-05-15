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

import random

import numpy as np
import pytest

import paddle
import paddle.optimizer as opt
from paddle import nn

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 1

IMAGE_SIZE = 784
CLASS_NUM = 48

paddle.seed(333)
np.random.seed(333)
random.seed(333)


# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
        self._conv = nn.Conv2D(3, 3, (4, 4))

    @paddle.jit.to_static
    def forward(self, x):
        x = self._linear(x)
        x = paddle.reshape(x, [16, 3, 4, 4])
        x = self._conv(x)
        x = paddle.mean(x, axis=[1, 2, 3])
        x = paddle.reshape(x, [16, 1])
        x = paddle.cast(x, 'int64')
        return x


def train(layer, loader, loss_fn, opt):
    loss = -1
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            # loss = loss_fn(out, label)
            print(" batch id ", batch_id)
            print(label)
            print(out)
            loss = paddle.add(out, label)
            loss = paddle.cast(loss, 'float32')
            loss = paddle.mean(loss)
            loss.backward()
            opt.step()
            opt.clear_grad()
            print(
                "Epoch {} batch {}: loss = {}".format(
                    epoch_id, batch_id, np.mean(loss.numpy())
                )
            )
            loss = np.mean(loss.numpy())
    return loss


@pytest.mark.bp_fp_no_trans
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_bp_fp_no_trans():
    # create network
    layer = LinearNet()
    loss_fn = nn.CrossEntropyLoss()
    adam = opt.Adam(
        learning_rate=0.001,
        parameters=layer.parameters(),
        use_multi_tensor=True,
    )

    # create data loader
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    loader = paddle.io.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=1,
    )
    loss_cpu = 23.4375
    paddle.set_device('gcu')
    # # train gcu
    loss_gcu = train(layer, loader, loss_fn, adam)
    assert np.allclose(loss_gcu, loss_cpu, atol=1e-4, rtol=1e-5)
