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

from __future__ import print_function

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Embedding
from paddle.fluid.dygraph.base import to_variable

from test_dist_base import runtime_main, TestParallelDyGraphRunnerBase
from parallel_dygraph_sparse_embedding import SimpleNet, fake_sample_reader, TestSparseEmbedding

# global configs
batch_size = 4
batch_num = 200
hidden_size = 10
vocab_size = 1000
num_steps = 3
init_scale = 0.1


class TestSparseEmbeddingFP64(TestSparseEmbedding):
    def get_model(self):
        model = SimpleNet(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_steps=num_steps,
            init_scale=init_scale,
            is_sparse=True,
            dtype="float64")

        train_reader = paddle.batch(
            fake_sample_reader(), batch_size=batch_size, drop_last=True)

        optimizer = fluid.optimizer.SGD(learning_rate=0.001,
                                        parameter_list=model.parameters())

        return model, train_reader, optimizer


if __name__ == "__main__":
    runtime_main(TestSparseEmbeddingFP64)
