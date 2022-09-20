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

import copy
import unittest

import paddle
from paddle.fluid import core
from paddle.distributed.fleet import auto
from paddle.distributed.auto_parallel.completion import Completer
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.parallelizer import AutoParallelizer
from paddle.distributed.auto_parallel.partitioner import Partitioner
from paddle.distributed.auto_parallel.dist_context import DistributedContext
from paddle.distributed.auto_parallel.dist_tensor import DistributedTensor
from paddle.distributed.auto_parallel.dist_attribute import TensorDistributedAttribute
import test_auto_parallel_reshard
from test_auto_parallel_reshard import mlp_forward


def get_dist_prog(train_program,
                  startup_program,
                  dist_context,
                  rank_id,
                  complete_train_program=None):
    loss, train_program, startup_program = mlp_forward(train_program,
                                                       startup_program)

    fleet._user_defined_strategy = fleet.DistributedStrategy()
    fleet.user_defined_optimizer = paddle.fluid.optimizer.AdamOptimizer()
    parallelizer = AutoParallelizer(fleet)
    parallelizer._dist_context = dist_context

    # serial forward & backward completion
    completer = Completer(dist_context)
    complete_train_program = completer.complete_forward_annotation(
        train_program
    ) if complete_train_program is None else complete_train_program
    dist_context.block_state.parse_forward_blocks(complete_train_program)

    params_grads = parallelizer._generate_backward(complete_train_program,
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

    return auto_parallel_main_prog, auto_parallel_startup_prog, complete_train_program


class TestDistributedTensor(unittest.TestCase):

    def test_new_local_tensor(self):
        test_auto_parallel_reshard._global_process_mesh = auto.ProcessMesh(
            mesh=[0, 1], dim_names=["x"])
        test_auto_parallel_reshard._global_parallel_strategy = "dp"
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        dist_context = DistributedContext()
        rank_id = 0
        dist_main_prog, dist_startup_prog, complete_train_program = get_dist_prog(
            train_program, startup_program, dist_context, rank_id)
        dist_context.dist_main_programs[rank_id] = dist_main_prog
        dist_context.dist_startup_programs[rank_id] = dist_startup_prog
        name = "layer_norm_1.tmp_2"
        dist_tensor = dist_context.get_dist_tensor_for_program(
            complete_train_program.global_block().vars[name])
        dist_tensor._dist_context = dist_context
        intermediate_var_0 = dist_tensor.new_local_tensor(
            name="intermediate_var_0")
        self.assertEqual(intermediate_var_0.shape, (2, 1024))
        self.assertEqual(intermediate_var_0.name, "intermediate_var_0")

        rank_id = 1
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        dist_context = DistributedContext()
        dist_main_prog, dist_startup_prog, complete_train_program = get_dist_prog(
            train_program, startup_program, dist_context, rank_id, None)
        dist_context.dist_main_programs[rank_id] = dist_main_prog
        dist_context.dist_startup_programs[rank_id] = dist_startup_prog
        name = "layer_norm_1.tmp_2"
        dist_tensor = dist_context.get_dist_tensor_for_program(
            complete_train_program.global_block().vars[name])
        dist_tensor._dist_context = dist_context
        intermediate_var_1 = dist_tensor.new_local_tensor(
            rank=rank_id, name="intermediate_var_1")
        self.assertEqual(intermediate_var_0.shape, (2, 1024))
        self.assertEqual(intermediate_var_1.name, "intermediate_var_1")

        name = "linear_0.w_0"
        dist_tensor = dist_context.get_dist_tensor_for_program(
            complete_train_program.global_block().vars[name])
        dist_tensor._dist_context = dist_context
        intermediate_var_1 = dist_tensor.new_local_tensor(
            rank=rank_id, name="linear_0.w_0_intermediate")
        self.assertEqual(intermediate_var_1.shape, (1024, 4096))
        self.assertEqual(intermediate_var_1.name, "linear_0.w_0_intermediate")

        copied_dist_context = copy.deepcopy(dist_context)
        self.assertIsNotNone(copied_dist_context)
        self.assertEqual(
            id(copied_dist_context),
            id(
                copied_dist_context.get_dist_tensor_for_program(
                    dist_tensor.serial_tensor).dist_context))

    def test_static_method(self):
        dims_mapping = [1, 0]
        processes = [0, 1, 2, 3, 4, 5, 6]
        topology = [2, 3]
        global_sizes = [6, 6]

        # rank 0 [(0, 2), (0, 3)]
        # rank 1 [(2, 4), (0, 3)]
        # rank 4 [(2, 4), (3, 6)]
        rank = 0
        local_sizes = DistributedTensor.get_local_sizes(global_sizes,
                                                        dims_mapping, topology,
                                                        processes)
        self.assertEqual(local_sizes, [2, 3])
        local_offsets = DistributedTensor.get_local_offsets(
            global_sizes, dims_mapping, topology, processes, rank)
        self.assertEqual(local_offsets, [0, 0])
        local_shard = DistributedTensor.get_local_shard(global_sizes,
                                                        dims_mapping, topology,
                                                        processes, rank)
        self.assertEqual(local_shard, [(0, 2), (0, 3)])

        rank = 1
        local_sizes = DistributedTensor.get_local_sizes(global_sizes,
                                                        dims_mapping, topology,
                                                        processes)
        self.assertEqual(local_sizes, [2, 3])
        local_offsets = DistributedTensor.get_local_offsets(
            global_sizes, dims_mapping, topology, processes, rank)
        self.assertEqual(local_offsets, [2, 0])
        local_shard = DistributedTensor.get_local_shard(global_sizes,
                                                        dims_mapping, topology,
                                                        processes, rank)
        self.assertEqual(local_shard, [(2, 4), (0, 3)])

        rank = 4
        local_sizes = DistributedTensor.get_local_sizes(global_sizes,
                                                        dims_mapping, topology,
                                                        processes)
        self.assertEqual(local_sizes, [2, 3])
        local_offsets = DistributedTensor.get_local_offsets(
            global_sizes, dims_mapping, topology, processes, rank)
        self.assertEqual(local_offsets, [2, 3])
        local_shard = DistributedTensor.get_local_shard(global_sizes,
                                                        dims_mapping, topology,
                                                        processes, rank)
        self.assertEqual(local_shard, [(2, 4), (3, 6)])

        # global sizes
        local_sizes = [2, 3]
        global_sizes = DistributedTensor.get_global_sizes(
            local_sizes, dims_mapping, topology, processes)
        self.assertEqual(global_sizes, [6, 6])

    def test_instance_method(self):
        tensor_dist_attr = TensorDistributedAttribute()
        tensor_dist_attr.dims_mapping = [1, 0]
        tensor_dist_attr.process_mesh = auto.ProcessMesh(
            mesh=[[0, 1, 2], [3, 4, 5]])
        serial_tensor = paddle.static.data(name="data",
                                           shape=[6, 6],
                                           dtype='float32')
        dist_tensor = DistributedTensor(serial_tensor, tensor_dist_attr)

        # rank 0 [(0, 2), (0, 3)]
        # rank 1 [(2, 4), (0, 3)]
        # rank 4 [(2, 4), (3, 6)]
        rank = 0
        local_sizes = dist_tensor.local_sizes(rank)
        self.assertEqual(local_sizes, [2, 3])
        local_offsets = dist_tensor.local_offsets(rank)
        self.assertEqual(local_offsets, [0, 0])
        local_shard = dist_tensor.local_shard(rank)
        self.assertEqual(local_shard, [(0, 2), (0, 3)])
        self.assertEqual(local_sizes, dist_tensor.local_sizes(rank))
        self.assertEqual(local_offsets, dist_tensor.local_offsets(rank))
        self.assertEqual(local_shard, dist_tensor.local_shard(rank))
        self.assertEqual(local_sizes, dist_tensor.local_sizes())
        self.assertEqual(local_offsets, dist_tensor.local_offsets())
        self.assertEqual(local_shard, dist_tensor.local_shard())

        rank = 1
        local_sizes = dist_tensor.local_sizes(rank)
        self.assertEqual(local_sizes, [2, 3])
        local_offsets = dist_tensor.local_offsets(rank)
        self.assertEqual(local_offsets, [2, 0])
        local_shard = dist_tensor.local_shard(rank)
        self.assertEqual(local_shard, [(2, 4), (0, 3)])

        rank = 4
        local_sizes = dist_tensor.local_sizes(rank)
        self.assertEqual(local_sizes, [2, 3])
        local_offsets = dist_tensor.local_offsets(rank)
        self.assertEqual(local_offsets, [2, 3])
        local_shard = dist_tensor.local_shard(rank)
        self.assertEqual(local_shard, [(2, 4), (3, 6)])

        global_sizes = dist_tensor.global_sizes()
        self.assertEqual(global_sizes, (6, 6))


if __name__ == "__main__":
    unittest.main()
