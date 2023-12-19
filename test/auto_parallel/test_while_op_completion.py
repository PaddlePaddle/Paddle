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

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn, static
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.completion import Completer
from paddle.distributed.auto_parallel.static.dist_context import (
    DistributedContext,
)
from paddle.distributed.fleet import auto

paddle.enable_static()

batch_size = 4
epoch_num = 10
hidden_size = 1024
sequence_len = 512
_g_process_mesh = auto.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])


def get_random_inputs_and_labels(input_shape, label_shape):
    input = np.random.random(size=input_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('float32')
    return input, label


def batch_generator_creator():
    def __reader__():
        for _ in range(batch_size):
            batch_input, batch_label = get_random_inputs_and_labels(
                [batch_size, sequence_len, hidden_size],
                [batch_size, sequence_len, 1],
            )
            yield batch_input, batch_label

    return __reader__


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
        param_initializer = nn.initializer.Normal(
            mean=0.0, std=initializer_range
        )

        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
        self.linear0 = nn.Linear(
            d_model,
            dim_feedforward,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None,
        )
        self.linear1 = nn.Linear(
            dim_feedforward,
            d_model,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None,
        )

    def forward(self, input):
        out = self.norm(input)
        auto.shard_tensor(
            self.linear0.weight, _g_process_mesh[:, 0], [None, 'x']
        )
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        auto.shard_tensor(
            self.linear1.weight, _g_process_mesh[:, 1], ['x', None]
        )
        out = self.linear1(out)

        return out


def loop_cond(i, loop_len, input_array):
    return i < loop_len


def loop_body(i, loop_len, input_array):
    pre_input = paddle.tensor.array_read(array=input_array, i=i)
    mlp_while0 = MLPLayer(
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        dropout_ratio=0.1,
        initializer_range=0.02,
    )

    mlp_while1 = MLPLayer(
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        dropout_ratio=0.1,
        initializer_range=0.02,
    )

    output = mlp_while0(pre_input)
    cur_pred = mlp_while1(output)
    # 更新循环条件
    i = paddle.increment(x=i, value=1)
    paddle.tensor.array_write(cur_pred, array=input_array, i=i)
    return i, loop_len, input_array


def get_program():
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    # fleet.init(is_collective=True, strategy=dist_strategy)

    train_program = static.Program()
    start_program = static.Program()
    with static.program_guard(train_program, start_program):
        # 循环计数器
        i = paddle.full(shape=[1], fill_value=0, dtype='int64')
        # 循环次数
        loop_len = paddle.full(shape=[1], fill_value=epoch_num, dtype='int64')

        # input
        input = static.data(
            name="input",
            shape=[batch_size, sequence_len, hidden_size],
            dtype='float32',
        )
        label = static.data(
            name="label", shape=[batch_size, sequence_len, 1], dtype='float32'
        )
        data_holder = [input, label]
        # dataloader
        dataloader = paddle.base.io.DataLoader.from_generator(
            feed_list=data_holder, capacity=4 * batch_size, iterable=False
        )
        dataloader.set_batch_generator(
            batch_generator_creator(), places=paddle.static.cuda_places()
        )
        # data dist_attr
        auto.shard_tensor(input, _g_process_mesh[:, 0], [None, None, None])
        auto.shard_tensor(label, _g_process_mesh[:, 0], [None, None, None])

        mlp_start = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02,
        )
        pred = mlp_start(input)

        input_array = paddle.tensor.array_write(pred, i)
        i, loop_len, input_array = static.nn.while_loop(
            cond=loop_cond, body=loop_body, loop_vars=[i, loop_len, input_array]
        )
        end_pred = paddle.tensor.array_read(array=input_array, i=i)

        mlp_end = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02,
        )
        pred = mlp_end(end_pred)

        error_cost = paddle.nn.functional.square_error_cost(pred, label)
        loss = paddle.mean(error_cost)

    return train_program, start_program, dataloader, i, loss


class TestMLP(unittest.TestCase):
    def test_completer(self):
        train_program, start_program, dataloader, i, loss = get_program()
        dist_context = DistributedContext()
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(
            train_program
        )
        # print_program_with_dist_attr(complete_train_program, dist_context)

    def test_completer_by_dist_op(self):
        train_program, start_program, dataloader, i, loss = get_program()
        dist_context = DistributedContext()
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(
            train_program
        )
        complete_train_program = completer._complete_tensor_dist_attr_by_op()


if __name__ == "__main__":
    unittest.main()
