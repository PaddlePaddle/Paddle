# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import time
import tempfile
import copy
import os
import numpy as np
import subprocess
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
import paddle.static as static
import paddle.nn.functional as F
import paddle.utils as utils
from paddle.fluid import layers
from paddle.io import Dataset, IterableDataset, DataLoader

from paddle.distributed.fleet import auto
from paddle.distributed.auto_parallel.interface import get_collection, CollectionNames
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.fluid.dataloader.collate import default_collate_fn

paddle.enable_static()
global_process_mesh = auto.ProcessMesh(mesh=[0, 1])
PP_MESH_0 = auto.ProcessMesh([0])
PP_MESH_1 = auto.ProcessMesh([1])
batch_size = 1
batch_num = 10
hidden_size = 1024
sequence_len = 512
image_size = hidden_size
class_num = 10

paddle.seed(44)

is_fetch = True
is_feed = True
my_feed_vars = []


class MyDataset(Dataset):

    def __init__(self, num_samples):
        super(MyDataset, self).__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=image_size).astype("float32")
        label = np.random.randint(0, class_num - 1, dtype="int64")
        return input, label

    def __len__(self):
        return self.num_samples


def get_random_inputs_and_labels(image_shape, label_shape):
    input = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return input, label


def batch_generator_creator():

    def __reader__():
        for _ in range(batch_size):
            batch_input, batch_label = get_random_inputs_and_labels(
                [batch_size, image_size], [batch_size, 1])
            yield batch_input, batch_label

    return __reader__


class MLPLayer(nn.Layer):

    def __init__(self,
                 hidden_size=1024,
                 intermediate_size=4 * 1024,
                 dropout_ratio=0.1,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range))
        bias_attr = None

        self.linear0 = nn.Linear(d_model,
                                 dim_feedforward,
                                 weight_attr,
                                 bias_attr=bias_attr)
        self.linear1 = nn.Linear(dim_feedforward,
                                 d_model,
                                 weight_attr,
                                 bias_attr=bias_attr)
        self.linear2 = nn.Linear(d_model, 1, weight_attr, bias_attr=bias_attr)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout = nn.Dropout(dropout_ratio, mode="upscale_in_train")

    def forward(self, input):
        out = auto.shard_op(self.norm, PP_MESH_0)(input)
        out = self.linear0(out)
        if is_feed:
            my_feed_vars.append((out, out.shape))
        out = F.gelu(out, approximate=True)
        out = auto.shard_op(self.linear1, PP_MESH_1)(out)
        out = self.dropout(out)
        out = self.linear2(out)
        if is_feed:
            my_feed_vars.append((out, out.shape))
        if is_fetch:
            auto.fetch(out, "my_out", logging=True)
        return out


def train_high_level(fetch):
    global is_fetch
    is_fetch = fetch
    mlp = MLPLayer(hidden_size=hidden_size,
                   intermediate_size=4 * hidden_size,
                   dropout_ratio=0.1,
                   initializer_range=0.02)
    loss = paddle.nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(learning_rate=0.00001,
                                      beta1=0.9,
                                      beta2=0.999,
                                      epsilon=1e-08,
                                      grad_clip=None)
    metric = paddle.metric.Accuracy()

    strategy = auto.Strategy()
    strategy.auto_mode = "semi"

    engine = auto.Engine(mlp, loss, optimizer, metric, strategy=strategy)

    # train
    train_dataset = MyDataset(batch_num * batch_size)
    eval_dataset1 = MyDataset(5 * batch_size)
    engine.fit(train_data=train_dataset,
               epochs=2,
               batch_size=batch_size,
               valid_data=eval_dataset1)

    # eval
    eval_dataset2 = MyDataset(batch_size)
    engine.evaluate(eval_dataset2, batch_size=batch_size)

    # predict
    test_dataset = MyDataset(batch_size)
    engine.predict(test_dataset, batch_size=batch_size)

    # save
    temp_dir = tempfile.TemporaryDirectory()
    model_filename = os.path.join(temp_dir.name, 'mlp')
    engine.save(model_filename, training=True)
    engine.load(model_filename)
    temp_dir.cleanup()


def train_low_level():
    mlp = MLPLayer(hidden_size=hidden_size,
                   intermediate_size=4 * hidden_size,
                   dropout_ratio=0.1,
                   initializer_range=0.02)
    loss = paddle.nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(learning_rate=0.00001,
                                      beta1=0.9,
                                      beta2=0.999,
                                      epsilon=1e-08,
                                      grad_clip=None)
    metric = paddle.metric.Accuracy()

    strategy = auto.Strategy()
    strategy.auto_mode = "semi"

    engine = auto.Engine(mlp, loss, optimizer, metric, strategy=strategy)

    feed_dict = {}
    for feed_var, shape in my_feed_vars:
        feed_dict[feed_var.name] = np.zeros(shape, dtype="float32")

    # Build normal dataloader
    # train
    train_dataset = MyDataset(batch_num * batch_size)
    train_dataloader = engine.dataloader(train_dataset,
                                         return_list=False,
                                         batch_size=batch_size,
                                         mode="train")
    engine.prepare(mode="train")
    for data in train_dataloader:
        outs = engine.run(data, feeds=feed_dict, mode="train")

    # eval
    eval_dataset2 = MyDataset(batch_size)
    eval_dataloader = engine.dataloader(eval_dataset2,
                                        batch_size=batch_size,
                                        mode="eval")
    engine.prepare(mode="eval")
    for data in eval_dataloader:
        outs = engine.run(data, feeds=feed_dict, mode="eval")

    # predict
    engine.to_mode("predict")
    test_dataset = MyDataset(batch_size)
    predict_dataloader = engine.dataloader(test_dataset, batch_size=batch_size)
    engine.prepare()
    for data in predict_dataloader:
        outs = engine.run(data, feeds=feed_dict)

    # save
    temp_dir = tempfile.TemporaryDirectory()
    model_filename = os.path.join(temp_dir.name, 'mlp')
    engine.save(model_filename, training=True)
    engine.load(model_filename)
    temp_dir.cleanup()

    # Build dataloader from generator
    # train
    train_dataset = MyDataset(batch_num * batch_size)
    train_dataloader = engine.dataloader_from_generator(train_dataset,
                                                        batch_size=batch_size,
                                                        mode="train")
    engine.prepare(mode="train")
    for data in train_dataloader:
        outs = engine.run(data, feeds=feed_dict, mode="train")

    # eval
    eval_dataset2 = MyDataset(batch_size)
    engine.to_mode(mode="eval")
    eval_dataloader = engine.dataloader_from_generator(eval_dataset2,
                                                       batch_size=batch_size)
    engine.prepare()
    for data in eval_dataloader:
        outs = engine.run(data, feeds=feed_dict, mode="eval")

    # predict
    test_dataset = MyDataset(batch_size)
    predict_dataloader = engine.dataloader_from_generator(test_dataset,
                                                          batch_size=batch_size,
                                                          mode="predict")
    engine.prepare(mode="predict")
    for data in predict_dataloader:
        outs = engine.run(data, feeds=feed_dict, mode="predict")

    # save
    temp_dir = tempfile.TemporaryDirectory()
    model_filename = os.path.join(temp_dir.name, 'mlp')
    engine.save(model_filename, training=True)
    engine.load(model_filename)
    temp_dir.cleanup()


def train_within_static():
    main_program = static.Program()
    startup_program = static.Program()
    with static.program_guard(main_program,
                              startup_program), utils.unique_name.guard():
        input = static.data(name="input",
                            shape=[batch_size, image_size],
                            dtype='float32')
        label = static.data(name="label", shape=[batch_size, 1], dtype='int64')

        mlp = MLPLayer(hidden_size=hidden_size,
                       intermediate_size=4 * hidden_size,
                       dropout_ratio=0.1,
                       initializer_range=0.02)
        loss = paddle.nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.Adam(learning_rate=0.00001,
                                          beta1=0.9,
                                          beta2=0.999,
                                          epsilon=1e-08,
                                          grad_clip=None)
        metric = paddle.metric.Accuracy()
        predict = mlp(input)
        loss_var = loss(predict, label)

    loader = paddle.io.DataLoader.from_generator(feed_list=[input, label],
                                                 capacity=4 * batch_size,
                                                 iterable=True)
    places = static.cuda_places()
    loader.set_batch_generator(batch_generator_creator(), places=places)
    print(main_program)

    strategy = auto.Strategy()
    strategy.auto_mode = "semi"

    engine = auto.Engine(loss=loss_var, optimizer=optimizer, strategy=strategy)

    # train
    engine.to_mode("train")
    engine.prepare(main_program=main_program, startup_program=startup_program)
    for data in loader:
        print(data)
        outs = engine.run(feed=data)


if __name__ == "__main__":
    train_high_level(fetch=True)
    train_high_level(fetch=False)
    train_low_level()
    train_within_static()
