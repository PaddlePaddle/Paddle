# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys

import numpy as np

sys.path.append("..")

import paddle
from legacy_test.test_dist_base import (
    TestParallelDyGraphRunnerBase,
    runtime_main,
)
from paddle.nn import Embedding, Layer


class SimpleNet(Layer):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        num_steps=20,
        init_scale=0.1,
        is_sparse=False,
        dtype="float32",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_steps = num_steps
        self.embedding = Embedding(
            self.vocab_size,
            self.hidden_size,
            sparse=is_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(
                    low=-init_scale, high=init_scale
                )
            ),
        )
        self.softmax_weight = self.create_parameter(
            attr=paddle.ParamAttr(),
            shape=[self.hidden_size, self.vocab_size],
            dtype=dtype,
            default_initializer=paddle.nn.initializer.Uniform(
                low=-self.init_scale, high=self.init_scale
            ),
        )
        self.softmax_bias = self.create_parameter(
            attr=paddle.ParamAttr(),
            shape=[self.vocab_size],
            dtype=dtype,
            default_initializer=paddle.nn.initializer.Uniform(
                low=-self.init_scale, high=self.init_scale
            ),
        )
        # add tmp var
        self.tmp = self.create_parameter(
            attr=paddle.ParamAttr(),
            shape=[self.vocab_size],
            dtype=dtype,
            default_initializer=paddle.nn.initializer.Uniform(
                low=-self.init_scale, high=self.init_scale
            ),
        )

    def forward(self, input, label):
        x_emb = self.embedding(input)
        fc = paddle.matmul(x_emb, self.softmax_weight)

        # it use stop gradient to block gradient return
        fc.stop_gradient = True
        fc = paddle.add(fc, self.softmax_bias)
        projection = paddle.reshape(fc, shape=[-1, self.vocab_size])
        loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False
        )
        loss = paddle.reshape(loss, shape=[-1, self.num_steps])
        loss = paddle.mean(loss, axis=[0])
        loss = paddle.sum(loss)

        return {"loss": loss}


# global configs
batch_size = 4
batch_num = 200
hidden_size = 10
vocab_size = 1000
num_steps = 3
init_scale = 0.1


def fake_sample_reader():
    def __reader__():
        for i in range(batch_num):
            x_data = np.arange(num_steps).astype('int64')
            y_data = np.arange(1, 1 + num_steps).astype('int64')
            yield x_data, y_data

    return __reader__


class TestSparseEmbeddingUnusedVars(TestParallelDyGraphRunnerBase):
    def get_model(self):
        model = SimpleNet(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_steps=num_steps,
            init_scale=init_scale,
            is_sparse=False,
        )

        train_reader = paddle.batch(
            fake_sample_reader(), batch_size=batch_size, drop_last=True
        )

        optimizer = paddle.optimizer.SGD(
            learning_rate=0.001, parameters=model.parameters()
        )

        return model, train_reader, optimizer

    def run_one_loop(self, model, optimizer, batch):
        x_data = np.array([x[0].reshape(3) for x in batch]).astype('int64')
        y_data = np.array([x[1].reshape(3) for x in batch]).astype('int64')
        x_data = x_data.reshape((-1, num_steps, 1))
        y_data = y_data.reshape((-1, 1))

        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)

        dy_loss = model(x, y)

        return dy_loss["loss"]


if __name__ == "__main__":
    runtime_main(TestSparseEmbeddingUnusedVars)
