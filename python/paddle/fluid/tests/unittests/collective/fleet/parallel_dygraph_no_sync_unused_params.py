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

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
from test_dist_base import runtime_main
from parallel_dygraph_no_sync import TestNoSync

seed = 90
RUN_STEP = 20
batch_size = 4
batch_num = 1000


class SimpleNetUnusedParam(fluid.Layer):

    def __init__(self):
        super(SimpleNetUnusedParam, self).__init__()
        self.net_a = Linear(input_dim=10, output_dim=20)
        self.net_b = Linear(input_dim=20, output_dim=5)
        self.net_c = Linear(input_dim=5, output_dim=10)

        self.net_d = Linear(input_dim=20, output_dim=10)

    def forward(self, x):
        x = self.net_a(x)
        x.stop_gradient = True
        x = self.net_b(x)
        x = self.net_c(x)
        return x


class TestNoSyncUnusedParam(TestNoSync):

    def get_model(self):
        model = SimpleNetUnusedParam()
        train_reader = paddle.batch(fake_sample_reader(),
                                    batch_size=batch_size,
                                    drop_last=True)
        optimizer = paddle.optimizer.SGD(learning_rate=0.001,
                                         parameters=model.parameters())
        return model, train_reader, optimizer

    def run_one_loop(self, model, optimizer, batch):
        x_data = np.array([x for x in batch])
        x_data = x_data.reshape((-1, 10))
        x = paddle.to_tensor(x_data)
        out = model(x)
        loss = out.sum() / len(batch)
        return loss


def fake_sample_reader():

    def __reader__():
        for i in range(batch_num):
            x_data = np.random.random_sample((10, )).astype('float32')
            yield x_data

    return __reader__


if __name__ == "__main__":
    runtime_main(TestNoSyncUnusedParam)
