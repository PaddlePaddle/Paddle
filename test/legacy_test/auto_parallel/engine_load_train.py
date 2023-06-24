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

import tempfile

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet import auto
from paddle.io import Dataset

paddle.enable_static()

global_process_mesh = auto.ProcessMesh(mesh=[0, 1])
PP_MESH_0 = auto.ProcessMesh([0])
PP_MESH_1 = auto.ProcessMesh([1])
epoch_num = 2
batch_size = 10
batch_num = 60
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
        super().__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=image_size).astype("float32")
        label = np.random.randint(0, class_num - 1, dtype="int64")
        return input, label

    def __len__(self):
        return self.num_samples


class MLPLayer(nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        dropout_ratio=0.1,
        initializer_range=0.02,
    ):
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
        )
        bias_attr = None

        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr
        )
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr
        )
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
            auto.fetch(out, "my_fetch", logging=True)
        return out


def train(save_dir=None, load_dir=None):
    global is_fetch
    is_fetch = True
    mlp = MLPLayer(
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        dropout_ratio=0.1,
        initializer_range=0.02,
    )
    loss = paddle.nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(
        learning_rate=0.00001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        grad_clip=None,
    )
    metric = paddle.metric.Accuracy()

    strategy = auto.Strategy()
    strategy.auto_mode = "semi"

    engine = auto.Engine(mlp, loss, optimizer, metric, strategy=strategy)

    # train
    train_dataset = MyDataset(batch_num * batch_size)
    eval_dataset1 = MyDataset(5 * batch_size)

    history = engine.fit(
        train_data=train_dataset,
        epochs=epoch_num,
        batch_size=batch_size,
        save_dir=save_dir,
        load_dir=load_dir,
        valid_data=eval_dataset1,
        log_freq=20,
        save_checkpoint_every_n_step=20,
        keep_checkpoint_max_num=3,
    )


def load_train():
    temp_dir = tempfile.TemporaryDirectory()
    train(save_dir=temp_dir.name, load_dir=temp_dir.name)
    temp_dir.cleanup()


if __name__ == "__main__":
    load_train()
