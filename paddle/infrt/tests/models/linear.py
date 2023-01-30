# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
<<<<<<< HEAD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
=======
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# example 1: save layer
import numpy as np
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle
import paddle.nn as nn
import paddle.optimizer as opt

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10


# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
<<<<<<< HEAD
        label = np.random.randint(0, CLASS_NUM - 1, (1,)).astype('int64')
=======
        label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return image, label

    def __len__(self):
        return self.num_samples


class LinearNet(nn.Layer):
    def __init__(self):
<<<<<<< HEAD
        super().__init__()
=======
        super(LinearNet, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    @paddle.jit.to_static
    def forward(self, x):
        return self._linear(x)


def train(layer, loader, loss_fn, opt):
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()


# 1. train & save model.

# create network
layer = LinearNet()
loss_fn = nn.CrossEntropyLoss()
adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

# create data loader
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
loader = paddle.io.DataLoader(
<<<<<<< HEAD
    dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2
)
=======
    dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

# train
train(layer, loader, loss_fn, adam)

# save
path = "linear/linear"
paddle.jit.save(layer, path)
