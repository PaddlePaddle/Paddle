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
# limitations under the License

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn, static
from paddle.base.core import TensorDistAttr
from paddle.base.framework import Program
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.dist_context import (
    DistributedContext,
    set_default_distributed_context,
)
from paddle.distributed.auto_parallel.static.process_mesh_v2 import ProcessMesh
from paddle.distributed.fleet import auto

paddle.enable_static()

batch_size = 4
epoch_num = 10
hidden_size = 1024
sequence_len = 512
_g_process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]], dim_names=['x', 'y'])


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
            self.linear0.weight,
            process_mesh=_g_process_mesh[0],
            shard_spec=[None, 'y'],
        )
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        auto.shard_tensor(
            self.linear1.weight,
            process_mesh=_g_process_mesh[1],
            shard_spec=['y', None],
        )
        out = auto.shard_op(self.linear1, process_mesh=_g_process_mesh)(out)

        return out


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


def get_program():
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    # fleet.init(is_collective=True, strategy=dist_strategy)

    train_program = static.Program()
    start_program = static.Program()
    with static.program_guard(train_program, start_program):
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
        auto.shard_tensor(
            input, process_mesh=_g_process_mesh[0], shard_spec=['y', None, None]
        )
        auto.shard_tensor(
            label, process_mesh=_g_process_mesh[0], shard_spec=['y', None, None]
        )

        mlp_start = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02,
        )
        pred = mlp_start(input)

        mlp_mid = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02,
        )
        pred = mlp_mid(pred)

        mlp_end = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02,
        )
        pred = mlp_end(pred)

        error_cost = paddle.nn.functional.square_error_cost(pred, label)
        loss = paddle.mean(error_cost)

        optimizer = paddle.optimizer.Adam(
            learning_rate=0.00001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            grad_clip=None,
        )

        feed_vars = {"inputs": [input], "labels": [label]}
        fetch_vars = {"loss": [loss]}

    return (
        train_program,
        start_program,
        dataloader,
        loss,
        optimizer,
        feed_vars,
        fetch_vars,
    )


class TestDistAttrSerialization(unittest.TestCase):
    def test_serialization_tensor(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input = static.data(name="input", shape=[2, 3], dtype='float32')
        dist_attr = input.dist_attr
        dist_attr.process_mesh = ProcessMesh([[0, 1, 2], [3, 4, 5]])
        dist_attr.dims_mapping = [0, -1]
        dist_attr.batch_dim = 1
        dist_attr.dynamic_dims = [1, 1]
        dist_attr_data = dist_attr.serialize_to_string()

    def test_serialization_operator(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input = static.data(name="input", shape=[2, 3], dtype='float32')
            input1 = static.data(name="input1", shape=[3, 4], dtype='float32')
            output = paddle.matmul(input, input1)
        op = train_program.current_block().ops[0]
        process_mesh = ProcessMesh([[0, 1, 2], [3, 4, 5]])
        op_dist_attr = op.dist_attr

        op_dist_attr.process_mesh = process_mesh
        # Set the distributed attribute of input
        input_dist_attr = TensorDistAttr(input.desc)
        input_dist_attr.dims_mapping = [0, -1]
        op_dist_attr.set_input_dist_attr(input.name, input_dist_attr)
        # Set the distributed attribute of input1
        input1_dist_attr = TensorDistAttr(input1.desc)
        input1_dist_attr.dims_mapping = [-1, 1]
        op_dist_attr.set_input_dist_attr(input1.name, input1_dist_attr)
        # Set the distributed attribute of output
        output_dist_attr = TensorDistAttr(output.desc)
        output_dist_attr.dims_mapping = [0, 1]
        op_dist_attr.set_output_dist_attr(output.name, output_dist_attr)

    def test_serialization_program(self):
        set_default_distributed_context(DistributedContext())
        (
            train_program,
            start_program,
            dataloader,
            loss,
            optimizer,
            feed_vars,
            fetch_vars,
        ) = get_program()
        dist_context = DistributedContext(
            train_program, start_program, optimizer, loss, feed_vars, fetch_vars
        )
        dist_context.initialize(with_cpp=True)

        # Distribute context will clone the original train program to serial_main_program
        original_program = dist_context.serial_main_program
        for block in original_program.blocks:
            for tensor in block.vars.values():
                dist_attr_data = tensor.dist_attr.serialize_to_string()
                tensor._set_attr("dist_attr", dist_attr_data)
            for op in block.ops:
                dist_attr_data = op.dist_attr.serialize_to_string()
                op._set_attr("dist_attr", dist_attr_data)

        program_data = original_program.desc.serialize_to_string()
        program = Program.parse_from_string(program_data)

        for block in program.blocks:
            for tensor in block.vars.values():
                dist_attr_data = tensor.attr("dist_attr")
                tensor._remove_attr("dist_attr")
                tensor.dist_attr.parse_from_string(dist_attr_data)
            for op in block.ops:
                dist_attr_data = op.attr("dist_attr")
                op._remove_attr("dist_attr")
                op.dist_attr.parse_from_string(dist_attr_data)

        self.assertEqual(len(original_program.blocks), len(program.blocks))
        for original_block, block in zip(
            original_program.blocks, program.blocks
        ):
            self.assertEqual(
                len(original_block.vars.values()), len(block.vars.values())
            )
            for original_tensor in original_block.vars.values():
                self.assertEqual(
                    original_tensor.dist_attr,
                    block.vars[original_tensor.name].dist_attr,
                )
            self.assertEqual(len(original_block.ops), len(block.ops))
            for original_op, op in zip(original_block.ops, block.ops):
                self.assertEqual(original_op.dist_attr, op.dist_attr)


if __name__ == "__main__":
    unittest.main()
