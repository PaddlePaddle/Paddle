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
import os
import json
import tempfile

import paddle
import paddle.nn as nn
import paddle.static as static
import paddle.nn.functional as F
import paddle.utils as utils
from paddle.distributed.fleet import auto
from paddle.distributed.auto_parallel.completion import Completer
from paddle.distributed.auto_parallel.dist_context import DistributedContext
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.parallelizer import AutoParallelizer
from paddle.distributed.auto_parallel.partitioner import Partitioner
from paddle.distributed.auto_parallel.reshard import Resharder
from paddle.distributed.auto_parallel.utils import print_program_with_dist_attr
from paddle.distributed.auto_parallel.cluster import Cluster
from paddle.distributed.auto_parallel.cost import CommContext
from paddle.distributed.auto_parallel.cost.base_cost import build_comp_desc_from_dist_op
from paddle.distributed.auto_parallel.cost.base_cost import build_comm_desc_from_dist_op
from paddle.distributed.auto_parallel.cost.base_cost import build_comm_costs_from_descs
from paddle.distributed.auto_parallel.cost.base_cost import build_comp_costs_from_descs
from paddle.distributed.auto_parallel.cost.base_cost import build_dp_costs
from paddle.distributed.auto_parallel.cost import AllreduceSumOpCost
from paddle.distributed.auto_parallel.cost import _g_op_cost_factory
from test_cluster import cluster_json

paddle.enable_static()
_global_parallel_strategy = "dp_mp_pp"
_global_process_mesh = auto.ProcessMesh([[[0, 1], [4, 5]], [[2, 3], [6, 7]]],
                                        dim_names=["x", "y", "z"])
PP_MESH_0 = auto.ProcessMesh([[0, 1], [4, 5]], dim_names=["x", "y"])
PP_MESH_1 = auto.ProcessMesh([[2, 3], [6, 7]], dim_names=["x", "y"])


class MLPLayer(nn.Layer):

    def __init__(self,
                 hidden_size=1024,
                 intermediate_size=4 * 1024,
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
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)

    def forward(self, input):
        auto.shard_tensor(self.linear0.weight, PP_MESH_0, [None, "y"])
        auto.shard_tensor(self.linear1.weight, PP_MESH_1, ["y", None])

        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)

        return out


def mlp_forward(train_program, start_program):
    with static.program_guard(train_program,
                              start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
        input = static.data(name="input",
                            shape=[batch_size, hidden_size],
                            dtype='float32')
        label = static.data(name="label",
                            shape=[batch_size, 1],
                            dtype='float32')

        fill_constant_out = paddle.fluid.layers.fill_constant_batch_size_like(
            input=input, shape=[batch_size], value=1, dtype="int32")
        embedding = paddle.nn.Embedding(10, hidden_size, sparse=True)
        embedding_out = embedding(fill_constant_out)

        auto.shard_tensor(input, PP_MESH_0, ["x", None])
        auto.shard_tensor(label, PP_MESH_1, ["x", None])

        mlp = MLPLayer(hidden_size=hidden_size,
                       intermediate_size=4 * hidden_size,
                       initializer_range=0.02)

        predict = mlp(embedding_out)
        error_cost = paddle.nn.functional.square_error_cost(predict, label)
        loss = paddle.mean(error_cost)

    return loss, train_program, start_program


def get_prog(train_program, startup_program, dist_context, rank_id):
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
    params_grads = parallelizer._generate_backward(complete_train_program,
                                                   startup_program,
                                                   loss,
                                                   parameter_list=None,
                                                   no_grad_set=None,
                                                   callbacks=None)
    return train_program, startup_program, params_grads


class TestBaseCost(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_base_cost(self):
        # Build cluster
        cluster_json_path = os.path.join(self.temp_dir.name,
                                         "auto_parallel_cluster.json")
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)

        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        dist_context = DistributedContext()
        rank_id = 2
        train_program, startup_program, params_grads = get_prog(
            train_program, startup_program, dist_context, rank_id)

        for op in train_program.global_block().ops:
            dist_op = dist_context.get_dist_op_for_program(op)
            if dist_op:
                processes = dist_op.dist_attr.process_mesh.processes
                comp_descs = build_comp_desc_from_dist_op(dist_op, dist_context)
                self.assertTrue(isinstance(comp_descs, dict) and comp_descs)
                var_names = None
                if op.input_arg_names:
                    var_names = op.input_arg_names[0]
                    comm_descs = build_comm_desc_from_dist_op("c_allreduce_sum",
                                                              dist_op,
                                                              dist_context,
                                                              var_names,
                                                              attrs=None,
                                                              parallel_axis=0,
                                                              group_ranks=None)
                    self.assertTrue(isinstance(comm_descs, dict) and comm_descs)
                    comm_descs = build_comm_desc_from_dist_op(
                        "c_allreduce_sum",
                        dist_op,
                        dist_context,
                        var_names,
                        attrs=None,
                        parallel_axis=None,
                        group_ranks=processes)
                    self.assertTrue(isinstance(comm_descs, dict) and comm_descs)

                    comm_costs = build_comm_costs_from_descs(
                        AllreduceSumOpCost, dist_context, processes, comm_descs,
                        cluster)
                    self.assertTrue(comm_costs)

                    comp_costs = build_comp_costs_from_descs(
                        _g_op_cost_factory[op.type], dist_context, processes,
                        comp_descs, cluster)
                    self.assertTrue(comp_costs)

                    result = []
                    build_dp_costs(result, dist_op, dist_context, var_names[0],
                                   None, 0, cluster)
                    self.assertTrue(result)

        # Remove unnecessary files
        if os.path.exists(cluster_json_path):
            os.remove(cluster_json_path)


if __name__ == "__main__":
    unittest.main()
