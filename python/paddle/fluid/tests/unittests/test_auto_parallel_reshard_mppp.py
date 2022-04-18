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

import unittest

import paddle
import paddle.nn as nn
import paddle.static as static
import paddle.nn.functional as F
import paddle.utils as utils
import paddle.distributed.auto_parallel as auto
from paddle.distributed.auto_parallel.completion import Completer
from paddle.distributed.auto_parallel.dist_context import DistributedContext
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.parallelizer import AutoParallelizer
from paddle.distributed.auto_parallel.partitioner import Partitioner
from paddle.distributed.auto_parallel.reshard import Resharder
from paddle.distributed.auto_parallel.utils import print_program_with_dist_attr

paddle.enable_static()
_global_parallel_strategy = "mp_pp"
_global_process_mesh = auto.ProcessMesh([[0, 1], [2, 3]])
PP_MESH_0 = auto.ProcessMesh([0, 1])
PP_MESH_1 = auto.ProcessMesh([2, 3])


class MLPLayer(nn.Layer):
    def __init__(self,
                 hidden_size=1024,
                 intermediate_size=4 * 1024,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
            mean=0.0, std=initializer_range))
        bias_attr = None

        self.word_embeddings = nn.Embedding(
            hidden_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                name="word_embeddings",
                initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range)))

        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr)
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr)
        self.linear2 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr)

    def forward(self, input):
        auto.shard_tensor(
            self.word_embeddings.weight,
            dist_attr={"process_mesh": PP_MESH_0,
                       "dims_mapping": [0, -1]})
        auto.shard_tensor(
            self.linear0.weight,
            dist_attr={"process_mesh": PP_MESH_0,
                       "dims_mapping": [-1, 0]})
        auto.shard_tensor(
            self.linear1.weight,
            dist_attr={"process_mesh": PP_MESH_1,
                       "dims_mapping": [0, -1]})
        auto.shard_tensor(
            self.linear2.weight,
            dist_attr={"process_mesh": PP_MESH_1,
                       "dims_mapping": [0, -1]})
        w_out = self.word_embeddings(input)
        out = self.linear0(w_out)
        gelu_out = F.gelu(out, approximate=True)
        out = self.linear1(gelu_out)
        out1 = self.linear2(gelu_out)
        out = out + out1

        return out


def mlp_forward(train_program, start_program):
    with static.program_guard(train_program,
                              start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
        input = static.data(name="input", shape=[batch_size], dtype='int32')
        label = static.data(
            name="label", shape=[batch_size, 1], dtype='float32')

        auto.shard_tensor(
            input, dist_attr={"process_mesh": PP_MESH_0,
                              "dims_mapping": [-1]})
        auto.shard_tensor(
            label,
            dist_attr={"process_mesh": PP_MESH_1,
                       "dims_mapping": [-1, -1]})

        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            initializer_range=0.02)

        predict = mlp(input)
        error_cost = paddle.nn.functional.square_error_cost(predict, label)
        loss = paddle.mean(error_cost)

    return loss, train_program, start_program


def get_dist_prog(train_program, startup_program, dist_context, rank_id):
    global _global_process_mesh
    dist_context.process_mesh = _global_process_mesh
    loss, train_program, startup_program = mlp_forward(train_program,
                                                       startup_program)

    fleet._user_defined_strategy = fleet.DistributedStrategy()
    fleet.user_defined_optimizer = paddle.fluid.optimizer.AdamOptimizer()
    parallelizer = AutoParallelizer(fleet)
    parallelizer._dist_context = dist_context

    # serial forward & backward completion
    completer = Completer(dist_context)
    complete_train_program = completer.complete_forward_annotation(
        train_program)
    dist_context.block_state.parse_forward_blocks(complete_train_program)
    params_grads = parallelizer._generate_backward(
        complete_train_program,
        startup_program,
        loss,
        parameter_list=None,
        no_grad_set=None,
        callbacks=None)

    # logical partition
    partitioner = Partitioner(dist_context, rank_id)
    auto_parallel_main_prog, auto_parallel_startup_prog, dist_params_grads = partitioner.partition(
        complete_train_program, startup_program, params_grads)

    partitioned_optimize_ops = parallelizer._apply_optimize(
        auto_parallel_main_prog, auto_parallel_startup_prog, dist_params_grads)
    return auto_parallel_main_prog, auto_parallel_startup_prog, dist_params_grads


def check_send_recv_result(dist_main_prog, rank_id):
    send_result = False
    recv_result = False
    ops = dist_main_prog.global_block().ops
    if rank_id in [0, 1]:
        for idx, op in enumerate(ops):
            if op.type == "send_v2" and "gelu_0.tmp_0" in op.input_arg_names:
                send_result = True
            if op.type == "recv_v2" and "gelu_0.tmp_0@GRAD" in op.output_arg_names[
                    0]:
                recv_result = True
    else:
        for idx, op in enumerate(ops):
            if op.type == "send_v2" and "gelu_0.tmp_0@GRAD" in op.input_arg_names[
                    0]:
                send_result = True
            if op.type == "recv_v2" and "gelu_0.tmp_0" in op.output_arg_names[
                    0]:
                recv_result = True

    return send_result and recv_result


def check_initialization_for_mppp(dist_startup_prog, rank_id):
    if rank_id in [0, 1]:
        need_check_params = []
    else:
        need_check_params = ["linear_1.b_0", "linear_2.b_0"]
    broadcast_varnames = []
    for op in dist_startup_prog.global_block().ops:
        if op.type == "c_broadcast":
            broadcast_varnames.append(op.output_arg_names[0])

    return need_check_params == broadcast_varnames


def check_allgather(dist_main_program):
    allgather_out = "x@RESHARD_0"
    var_result = False
    op_result = False
    vars = dist_main_program.global_block().vars
    if allgather_out in vars and vars[allgather_out].shape == (4, 4):
        var_result = True
    for op in dist_main_program.global_block().ops:
        if op.type == "matmul_v2":
            if allgather_out in op.input_arg_names:
                op_result = True
    return var_result and op_result


class TestMLPReshard(unittest.TestCase):
    def test_mlp_mppp(self):
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        dist_context = DistributedContext()
        rank_id = 2
        dist_main_prog, dist_startup_prog, dist_params_grads = get_dist_prog(
            train_program, startup_program, dist_context, rank_id)
        resharder = Resharder(dist_main_prog, dist_startup_prog, rank_id,
                              dist_context, dist_params_grads)
        resharder.reshard()

        # check send and recv result
        self.assertTrue(check_send_recv_result(dist_main_prog, rank_id))

        # parameter which not been sliced should be the same in the mp scene
        self.assertTrue(
            check_initialization_for_mppp(dist_startup_prog, rank_id))

    def test_allgather(self):
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        process_mesh = auto.ProcessMesh(mesh=[0, 3])
        with static.program_guard(train_program, startup_program):
            x = paddle.static.data(name="x", shape=[4, 4], dtype='float32')
            x = auto.shard_tensor(
                x,
                dist_attr={
                    "process_mesh": process_mesh,
                    "dims_mapping": [0, -1]
                })

            w = paddle.static.data(name="w", shape=[4, 4], dtype='float32')
            w = auto.shard_tensor(
                w,
                dist_attr={
                    "process_mesh": process_mesh,
                    "dims_mapping": [-1, -1]
                })

            # y = paddle.distributed.shard_op(paddle.matmul, process_mesh, {
            #     x.name: [-1, -1],
            #     w.name: [-1, -1]
            # }, **{"x": x,
            #       "y": w})[0]

            y = paddle.distributed.shard_op(
                paddle.matmul,
                dist_attr={
                    "process_mesh": process_mesh,
                    x: {
                        "dims_mapping": [-1, -1]
                    },
                    w: {
                        "dims_mapping": [-1, -1]
                    }
                })(x, w)[0]

        rank_id = 0
        dist_context = DistributedContext()
        dist_strategy = fleet.DistributedStrategy()
        partitioner = Partitioner(dist_context, rank_id)
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(
            train_program)
        dist_context.block_state.parse_forward_blocks(complete_train_program)
        partitioned_main_prog, partitioned_startup_prog, partitioned_params_grads = partitioner.partition(
            complete_train_program, startup_program, [])
        resharder = Resharder(partitioned_main_prog, partitioned_startup_prog,
                              rank_id, dist_context, partitioned_params_grads)
        resharder.reshard()
        # the x should not be slice
        self.assertTrue(check_allgather(partitioned_main_prog))


if __name__ == "__main__":
    unittest.main()
