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

from __future__ import print_function

import numpy as np
import paddle.distributed as dist

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Embedding
from paddle.fluid.dygraph.base import to_variable
from paddle.distributed import fleet
import paddle.nn.functional as F
from test_dist_base import runtime_main, TestParallelDyGraphRunnerBase

paddle.seed(123)


class SimpleNet(fluid.Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 num_steps=20,
                 init_scale=0.1,
                 is_sparse=False):
        super(SimpleNet, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_steps = num_steps
        self.embedding = Embedding(
            size=[self.vocab_size, self.hidden_size],
            dtype='float32',
            is_sparse=is_sparse,
            param_attr=fluid.ParamAttr(
                name='embedding_param',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))

        self.softmax_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.hidden_size, self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

        self.fc_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.vocab_size, 1],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

        self.fc_weight_2 = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.vocab_size, 1],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

    def forward(self, input, label, conf):
        x_emb = self.embedding(input)
        fc = fluid.layers.matmul(x_emb, self.softmax_weight)

        fc = fluid.layers.elementwise_add(fc, self.softmax_bias)

        mask = conf > 0
        mask = paddle.cast(mask, dtype="int64")
        mask.stop_gradient = True
        emb_mask = mask.max(1).flatten()
        emb_mask_inds = paddle.nonzero(emb_mask > 0).flatten()
        emb_mask_inds.stop_gradient = True

        weight = paddle.to_tensor([1], dtype='float32')
        if emb_mask_inds.numel() == 0:
            weight = paddle.to_tensor([0], dtype='float32')

            projection = fluid.layers.reshape(fc, shape=[-1, self.vocab_size])
            projection = paddle.matmul(projection, self.fc_weight_2)
            projection = paddle.reshape(projection, shape=[-1, 1])

            output = projection[0]
            target = label[0]
            loss_box = F.smooth_l1_loss(
                output, target, reduction='sum', delta=1.0)
            loss_box = loss_box / len(conf)

        else:
            fc = fluid.layers.elementwise_add(fc, self.softmax_bias)
            projection = fluid.layers.reshape(fc, shape=[-1, self.vocab_size])
            projection = paddle.matmul(projection, self.fc_weight)
            projection = paddle.reshape(projection, shape=[-1, 1])

            output = paddle.gather(projection, emb_mask_inds)
            target = paddle.gather(label, emb_mask_inds)
            loss_box = F.smooth_l1_loss(
                output, target, reduction='sum', delta=1.0)
            loss_box = loss_box / len(conf)

        return loss_box * weight


# global configs
batch_size = 4
batch_num = 20000
hidden_size = 10
vocab_size = 100
num_steps = 3
init_scale = 0.5

conf_dataset = [
    [0, 0, 1],  # 0
    [0, 0, 0],  # 1 
    [1, 1, 0],  # 0
    [0, 0, 0],  # 1
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]


def fake_sample_reader():
    def __reader__():
        for i in range(batch_num):
            x_data = np.arange(num_steps).astype('int64')
            y_data = np.arange(1, 1 + num_steps).astype('float32')
            conf_data = np.array(conf_dataset[i % len(conf_dataset)]).astype(
                'int64')
            yield x_data, y_data, conf_data

    return __reader__


class TestControlFlow(TestParallelDyGraphRunnerBase):
    def get_model(self):
        model = SimpleNet(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_steps=num_steps,
            init_scale=init_scale,
            is_sparse=False)

        train_reader = paddle.batch(
            fake_sample_reader(), batch_size=batch_size, drop_last=True)

        optimizer = paddle.optimizer.SGD(learning_rate=0.001,
                                         parameters=model.parameters())

        return model, train_reader, optimizer

    def run_one_loop(self, model, optimizer, batch):
        x_data = np.array([x[0].reshape(3) for x in batch]).astype('int64')
        y_data = np.array([x[1].reshape(3) for x in batch]).astype('float32')
        conf_data = np.array([x[2].reshape(3) for x in batch]).astype('int64')
        x_data = x_data.reshape((-1, 1))
        y_data = y_data.reshape((-1, 1))
        conf_data = conf_data.reshape((-1, 1))

        x = to_variable(x_data)
        y = to_variable(y_data)
        conf = to_variable(conf_data)

        dy_loss = model(x, y, conf)

        return dy_loss


if __name__ == "__main__":
    runtime_main(TestControlFlow)
