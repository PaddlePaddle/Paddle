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
from test_dist_base import TestParallelDyGraphRunnerBase, runtime_main

import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F

paddle.seed(123)
np.random.seed(2021)


class SimpleNet(fluid.Layer):
    def __init__(self, hidden_size, vocab_size, is_sparse=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = paddle.nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            sparse=is_sparse,
        )

        self.lin_a = paddle.nn.Linear(self.hidden_size, self.vocab_size)
        self.lin_b = paddle.nn.Linear(self.vocab_size, 1)

        self.unused_net = paddle.nn.Linear(5, 3)
        self.phony = self.create_parameter(shape=[1], dtype="float32")

    def forward(self, input, label, conf):
        x_emb = self.embedding(input)
        fc = self.lin_a(x_emb)
        mask = conf > 0
        mask = paddle.cast(mask, dtype="int64")
        mask.stop_gradient = True
        emb_mask = mask.max(1).flatten()
        emb_mask_inds = paddle.nonzero(emb_mask > 0).flatten()
        emb_mask_inds.stop_gradient = True

        if emb_mask_inds.numel() == 0:
            loss_box = self.phony * 0
        else:
            projection = self.lin_b(fc)
            projection = paddle.reshape(projection, shape=[-1, 1])
            output = paddle.gather(projection, emb_mask_inds)
            target = paddle.gather(label, emb_mask_inds)
            loss_box = F.smooth_l1_loss(
                output, target, reduction='sum', delta=1.0
            )
            loss_box = loss_box / len(conf)

        return loss_box


# global configs
batch_size = 4
batch_num = 2000
hidden_size = 5
vocab_size = 100

conf_dataset = [
    [0],
    [0],
    [0],
    [0],
    [1],
    [0],
    [1],
    [0],
    [0],
    [1],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [0],
    [0],
    [1],
]


def fake_sample_reader():
    def __reader__():
        for i in range(batch_num):
            x_data = np.random.randint(0, vocab_size)
            y_data = np.random.random_sample((1,)).astype('float32')
            conf_data = np.array(conf_dataset[i % len(conf_dataset)]).astype(
                'int64'
            )
            yield x_data, y_data, conf_data

    return __reader__


class TestSimpleNet(TestParallelDyGraphRunnerBase):
    def get_model(self):
        model = SimpleNet(
            hidden_size=hidden_size, vocab_size=vocab_size, is_sparse=False
        )

        train_reader = paddle.batch(
            fake_sample_reader(), batch_size=batch_size, drop_last=True
        )

        optimizer = paddle.optimizer.SGD(
            learning_rate=0.001, parameters=model.parameters()
        )

        return model, train_reader, optimizer

    def run_one_loop(self, model, optimizer, batch):
        x_data = np.array([x[0] for x in batch]).astype('int64')
        y_data = np.array([x[1] for x in batch]).astype('float32')
        conf_data = np.array([x[2] for x in batch]).astype('int64')
        x_data = x_data.reshape((-1, 1))
        y_data = y_data.reshape((-1, 1))
        conf_data = conf_data.reshape((-1, 1))

        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        conf = paddle.to_tensor(conf_data)

        loss = model(x, y, conf)
        return loss


if __name__ == "__main__":
    runtime_main(TestSimpleNet)
