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
from paddle.static import InputSpec
from paddle.distributed import fleet
import paddle.distributed.auto_parallel as auto
from paddle.distributed.auto_parallel.engine import Engine
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.fluid.dataloader.collate import default_collate_fn

paddle.enable_static()
global_process_mesh = auto.ProcessMesh(mesh=[0, 1])
PP_MESH_0 = auto.ProcessMesh([0])
PP_MESH_1 = auto.ProcessMesh([1])
batch_size = 2
batch_num = 10
hidden_size = 1024
sequence_len = 512
image_size = hidden_size
class_num = 10

paddle.seed(44)


class MyDataset(IterableDataset):

    def __init__(self, num_samples):
        super(MyDataset, self).__init__()
        self.num_samples = num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            input = np.random.uniform(size=image_size).astype("float32")
            label = np.random.randint(0, class_num - 1, dtype="int64")
            yield input, label


class MyDataset1(Dataset):

    def __init__(self, num_samples):
        super(MyDataset1, self).__init__()
        self.num_samples = num_samples
        self.data = []
        for i in range(self.num_samples):
            input1 = np.random.uniform(size=image_size).astype("float32")
            label1 = np.array(np.random.randint(0, class_num - 1,
                                                dtype="int64"))
            input2 = np.random.uniform(size=image_size).astype("float32")
            label2 = np.array(np.random.randint(0, class_num - 1,
                                                dtype="int64"))
            input = np.stack((input1, input2))
            label = np.stack((label1, label2))
            self.data.append((input, label))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


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
        out = auto.shard_op(self.norm, dist_attr={"process_mesh":
                                                  PP_MESH_0})(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = auto.shard_op(self.linear1, dist_attr={"process_mesh":
                                                     PP_MESH_1})(out)
        out = self.dropout(out)
        out = self.linear2(out)
        self.out = out
        return out


def train(fetch):
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

    inputs_spec = InputSpec([batch_size, hidden_size], 'float32', 'x')
    labels_spec = InputSpec([batch_size], 'int64', 'label')

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    dist_strategy.split_data = True
    fleet.init(is_collective=True, strategy=dist_strategy)

    # init engine
    engine = Engine(mlp,
                    inputs_spec=inputs_spec,
                    labels_spec=labels_spec,
                    strategy=dist_strategy)
    engine.prepare(optimizer, loss, metrics=paddle.metric.Accuracy())

    # fetch
    if fetch:
        fetches = {'out': mlp.out}
    else:
        fetches = None

    # train
    train_dataset = MyDataset(batch_num * batch_size)
    train_dataset1 = MyDataset1(batch_num)
    engine.fit(train_dataset,
               epochs=2,
               batch_size=batch_size,
               steps_per_epoch=batch_num,
               fetches=fetches)

    engine.fit(train_dataset1,
               epochs=2,
               batch_size=None,
               steps_per_epoch=batch_num,
               fetches=fetches)

    # eval
    eval_dataset = MyDataset(batch_size)
    engine.evaluate(eval_dataset, batch_size, fetches=fetches)

    # predict
    test_dataset = MyDataset(batch_size)
    engine.predict(test_dataset, batch_size, fetches=fetches)

    # save
    temp_dir = tempfile.TemporaryDirectory()
    model_filename = os.path.join(temp_dir.name, 'mlp_inf')
    engine.save(model_filename, training=False, mode='predict')
    temp_dir.cleanup()


if __name__ == "__main__":
    train(fetch=True)
