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
import paddle
import numpy as np
import paddle.nn as nn
import paddle.utils as utils
import paddle.fluid as fluid
import paddle.static as static
import paddle.nn.functional as F
import paddle.distributed.auto_parallel as auto

from paddle.distributed import fleet

from paddle.distributed.auto_parallel.partitioner import Partitioner
from paddle.distributed.auto_parallel.utils import make_data_unshard
from paddle.distributed.auto_parallel.dist_attribute import OperatorDistributedAttribute, TensorDistributedAttribute
from paddle.distributed.auto_parallel.dist_context import DistributedContext, get_default_distributed_context
from paddle.distributed.auto_parallel.operators import find_best_compatible_distributed_operator_impl

paddle.enable_static()

batch_size = 4
epoch_num = 10
hidden_size = 1024
sequence_len = 512
_g_process_mesh = auto.ProcessMesh([0, 1])


def get_random_inputs_and_labels(input_shape, label_shape):
    input = np.random.random(size=input_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('float32')
    return input, label


def batch_generator_creator():
    def __reader__():
        for _ in range(batch_size):
            batch_input, batch_label = get_random_inputs_and_labels(
                [batch_size, sequence_len, hidden_size],
                [batch_size, sequence_len, 1])
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
        param_initializer = nn.initializer.Normal(
            mean=0.0, std=initializer_range)

        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
        self.linear0 = nn.Linear(
            d_model,
            dim_feedforward,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None)
        self.linear1 = nn.Linear(
            dim_feedforward,
            d_model,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None)

    def forward(self, input):

        auto.shard_tensor(
            self.norm.weight,
            dist_attr={"process_mesh": _g_process_mesh,
                       "dims_mapping": [-1]})
        auto.shard_tensor(
            self.norm.bias,
            dist_attr={"process_mesh": _g_process_mesh,
                       "dims_mapping": [-1]})
        auto.shard_tensor(
            self.linear0.weight,
            dist_attr={
                "process_mesh": _g_process_mesh,
                "dims_mapping": [-1, 0]
            })
        auto.shard_tensor(
            self.linear0.bias,
            dist_attr={"process_mesh": _g_process_mesh,
                       "dims_mapping": [0]})
        auto.shard_tensor(
            self.linear1.weight,
            dist_attr={
                "process_mesh": _g_process_mesh,
                "dims_mapping": [0, -1]
            })
        auto.shard_tensor(
            self.linear1.bias,
            dist_attr={"process_mesh": _g_process_mesh,
                       "dims_mapping": [-1]})

        out = self.norm(input)
        auto.shard_tensor(
            out,
            dist_attr={
                "process_mesh": _g_process_mesh,
                "dims_mapping": [-1, -1, -1]
            })
        out = self.linear0(out)
        auto.shard_tensor(
            out,
            dist_attr={
                "process_mesh": _g_process_mesh,
                "dims_mapping": [-1, -1, 0]
            })
        out = F.gelu(out, approximate=True)
        auto.shard_tensor(
            out,
            dist_attr={
                "process_mesh": _g_process_mesh,
                "dims_mapping": [-1, -1, 0]
            })
        out = self.linear1(out)
        auto.shard_tensor(
            out,
            dist_attr={
                "process_mesh": _g_process_mesh,
                "dims_mapping": [-1, -1, -1]
            })

        return out


def get_program():
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    # fleet.init(is_collective=True, strategy=dist_strategy)

    train_program = static.Program()
    start_program = static.Program()
    with fluid.program_guard(train_program, start_program):

        # 循环计数器
        i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
        auto.shard_tensor(
            i,
            dist_attr={"process_mesh": _g_process_mesh,
                       "dims_mapping": [-1]})

        # 循环次数
        loop_len = fluid.layers.fill_constant(
            shape=[1], dtype='int64', value=epoch_num)
        auto.shard_tensor(
            loop_len,
            dist_attr={"process_mesh": _g_process_mesh,
                       "dims_mapping": [-1]})

        # input
        input = static.data(
            name="input",
            shape=[batch_size, sequence_len, hidden_size],
            dtype='float32')
        label = static.data(
            name="label", shape=[batch_size, sequence_len, 1], dtype='float32')

        data_holder = [input, label]
        # dataloader
        dataloader = paddle.io.DataLoader.from_generator(
            feed_list=data_holder, capacity=4 * batch_size, iterable=False)
        dataloader.set_batch_generator(
            batch_generator_creator(), places=paddle.static.cuda_places())
        # data dist_attr
        auto.shard_tensor(
            input,
            dist_attr={
                "process_mesh": _g_process_mesh,
                "dims_mapping": [-1, -1, -1]
            })
        auto.shard_tensor(
            label,
            dist_attr={
                "process_mesh": _g_process_mesh,
                "dims_mapping": [-1, -1, -1]
            })

        # fill constant bsz like
        tmp = paddle.fluid.layers.fill_constant_batch_size_like(
            input=input, shape=[-1, 16, 0, 48], dtype='float32', value=0)
        auto.shard_tensor(
            tmp,
            dist_attr={
                "process_mesh": _g_process_mesh,
                "dims_mapping": [-1, 0, -1, -1]
            })

        # model
        mlp_start = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02)
        pred = mlp_start(input)

        input_array = fluid.layers.array_write(pred, i)
        auto.shard_tensor(
            input_array,
            dist_attr={
                "process_mesh": _g_process_mesh,
                "dims_mapping": [-1, -1, -1]
            })

        cond = fluid.layers.less_than(x=i, y=loop_len)
        auto.shard_tensor(
            cond,
            dist_attr={"process_mesh": _g_process_mesh,
                       "dims_mapping": [-1]})

        while_op = fluid.layers.While(cond=cond)
        with while_op.block():

            pre_input = fluid.layers.array_read(array=input_array, i=i)
            auto.shard_tensor(
                pre_input,
                dist_attr={
                    "process_mesh": _g_process_mesh,
                    "dims_mapping": [-1, -1, -1]
                })

            mlp_while = MLPLayer(
                hidden_size=hidden_size,
                intermediate_size=4 * hidden_size,
                dropout_ratio=0.1,
                initializer_range=0.02)
            cur_pred = mlp_while(pre_input)

            # 更新循环条件
            i = fluid.layers.increment(x=i, value=1, in_place=True)
            fluid.layers.array_write(cur_pred, array=input_array, i=i)
            fluid.layers.less_than(x=i, y=loop_len, cond=cond)

        end_pred = fluid.layers.array_read(array=input_array, i=i)
        auto.shard_tensor(
            end_pred,
            dist_attr={
                "process_mesh": _g_process_mesh,
                "dims_mapping": [-1, -1, -1]
            })

        mlp_end = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02)
        pred = mlp_end(end_pred)

        error_cost = paddle.nn.functional.square_error_cost(pred, label)
        auto.shard_tensor(
            error_cost,
            dist_attr={
                "process_mesh": _g_process_mesh,
                "dims_mapping": [-1, -1, -1]
            })

        loss = paddle.mean(error_cost)
        auto.shard_tensor(
            loss,
            dist_attr={"process_mesh": _g_process_mesh,
                       "dims_mapping": [-1]})

    return train_program, start_program, dataloader, i, loss


def completion(train_program, start_program, dist_context):
    blocks = train_program.blocks
    # completion tensors
    for block in blocks:
        for op in block.ops:
            if op.type == "layer_norm":
                for out_name in op.output_arg_names:
                    out_var = block.vars[out_name]
                    tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                        out_var)
                    if tensor_dist_attr:
                        continue
                    tensor_dist_attr = TensorDistributedAttribute()
                    tensor_dist_attr.process_mesh = _g_process_mesh
                    tensor_dist_attr.dims_mapping = [-1]
                    dist_context.set_tensor_dist_attr_for_program(
                        out_var, tensor_dist_attr)

            elif op.type == "elementwise_sub":
                for out_name in op.output_arg_names:
                    out_var = block.vars[out_name]
                    tensor_dist_attr = TensorDistributedAttribute()
                    tensor_dist_attr.process_mesh = _g_process_mesh
                    tensor_dist_attr.dims_mapping = [-1, -1, -1]
                    dist_context.set_tensor_dist_attr_for_program(
                        out_var, tensor_dist_attr)

            elif op.type == "matmul_v2":
                col = False
                for in_name in op.input_arg_names:
                    if ".w_" not in in_name:
                        continue
                    if in_name not in block.vars:
                        in_var = blocks[0].vars[in_name]
                    else:
                        in_var = block.vars[in_name]
                    tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                        in_var)
                    assert tensor_dist_attr is not None
                    if tensor_dist_attr.dims_mapping == [-1, 0]:
                        col = True
                for out_name in op.output_arg_names:
                    out_var = block.vars[out_name]
                    tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                        out_var)
                    if tensor_dist_attr:
                        continue
                    tensor_dist_attr = TensorDistributedAttribute()
                    tensor_dist_attr.process_mesh = _g_process_mesh
                    if col:
                        tensor_dist_attr.dims_mapping = [-1, -1, 0]
                    else:
                        tensor_dist_attr.dims_mapping = [-1, -1, -1]
                    dist_context.set_tensor_dist_attr_for_program(
                        out_var, tensor_dist_attr)
            elif op.type == "while":
                out_name = op.desc.output("StepScopes")[0]
                out_var = block.vars[out_name]
                tensor_dist_attr = TensorDistributedAttribute()
                tensor_dist_attr.process_mesh = _g_process_mesh
                tensor_dist_attr.dims_mapping = [-1]
                dist_context.set_tensor_dist_attr_for_program(out_var,
                                                              tensor_dist_attr)

    # completion ops
    for block in blocks:
        for op in block.ops:
            op_dist_attr = OperatorDistributedAttribute()
            op_dist_attr.process_mesh = _g_process_mesh
            if op.type == "create_by_read" or op.type == "create_double_buffer_reader":
                for in_name in op.input_arg_names:
                    op_dist_attr.set_input_dims_mapping(in_name, [])
                for out_name in op.output_arg_names:
                    op_dist_attr.set_output_dims_mapping(out_name, [])
            elif op.type == "read":
                for in_name in op.input_arg_names:
                    op_dist_attr.set_output_dims_mapping(in_name, [])
                for out_name in op.output_arg_names:
                    out_var = block.vars[out_name]
                    out_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                        out_var)
                    op_dist_attr.set_output_dist_attr(out_name, out_dist_attr)
            elif op.type == "while":
                for in_name in op.input_arg_names:
                    in_var = block.vars[in_name]
                    in_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                        in_var)
                    op_dist_attr.set_input_dist_attr(in_name, in_dist_attr)
                for out_name in op.output_arg_names:
                    if out_name == op.desc.output("StepScopes")[0]:
                        op_dist_attr.set_output_dims_mapping(out_name, [])
                    else:
                        out_var = block.vars[out_name]
                        out_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                            out_var)
                        op_dist_attr.set_output_dist_attr(out_name,
                                                          out_dist_attr)
            else:
                for in_name in op.input_arg_names:
                    if in_name == "lod_tensor_blocking_queue_0":
                        continue
                    if in_name not in block.vars:
                        in_var = blocks[0].vars[in_name]
                    else:
                        in_var = block.vars[in_name]
                    in_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                        in_var)
                    op_dist_attr.set_input_dist_attr(in_name, in_dist_attr)
                for out_name in op.output_arg_names:
                    if out_name not in block.vars:
                        out_var = blocks[0].vars[out_name]
                    else:
                        out_var = block.vars[out_name]
                    out_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                        out_var)
                    op_dist_attr.set_output_dist_attr(out_name, out_dist_attr)

            if op.type == "matmul_v2":
                op_dist_attr.impl_type = "matmul_v2"
                for in_name in op_dist_attr.inputs_dist_attrs.keys():
                    in_dist_attr = op_dist_attr.inputs_dist_attrs[in_name]
                    if ".w_" in in_name and in_dist_attr.dims_mapping[-1] == 0:
                        op_dist_attr.impl_idx = 0
                    else:
                        op_dist_attr.impl_idx = 1
            elif op.type == "fill_constant_batch_size_like":
                op_dist_attr.impl_type = "fill_constant_batch_size_like"
                op_dist_attr.impl_idx = 0
            else:
                op_dist_attr.impl_type = "default"
                op_dist_attr.impl_idx = 0

            dist_context.set_op_dist_attr_for_program(op, op_dist_attr)
            make_data_unshard(train_program, start_program, dist_context)

    return train_program, start_program


def partition(train_program, start_program, dist_context):

    # optimizer = paddle.optimizer.SGD(learning_rate=0.00001)
    rank = paddle.distributed.get_rank()
    partitioner = Partitioner(dist_context, rank)
    dist_main_prog, dist_startup_prog, _ = partitioner.partition(
        train_program, start_program, [])

    return dist_main_prog, dist_startup_prog


class TestMLP(unittest.TestCase):
    def test_partitioner(self):

        train_program, start_program, dataloader, i, loss = get_program()
        dist_context = get_default_distributed_context()
        train_program, start_program = completion(train_program, start_program,
                                                  dist_context)
        dist_context.block_state.parse_forward_blocks(train_program)

        dist_main_prog, dist_startup_prog = partition(
            train_program, start_program, dist_context)
        global_block_ops = dist_main_prog.blocks[0].ops

        fill_op = None
        for op in global_block_ops:
            if op.type == "fill_constant_batch_size_like":
                fill_op = op

        global_block_ops = [op.type for op in global_block_ops]
        sub_block_ops = dist_main_prog.blocks[1].ops
        sub_block_ops = [op.type for op in sub_block_ops]

        self.assertTrue("c_allreduce_sum" in global_block_ops)
        self.assertTrue("c_allreduce_sum" in sub_block_ops)

        # test fill_constant_batch_size_like

        self.assertTrue(fill_op is not None)
        ref_shape = [-1, 8, 0, 48]
        shape = fill_op.attr("shape")
        self.assertTrue(ref_shape == shape)


if __name__ == "__main__":
    unittest.main()
