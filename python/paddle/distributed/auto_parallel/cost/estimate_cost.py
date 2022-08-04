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

from collections import OrderedDict
from functools import reduce
import time

import paddle
import paddle.fluid.core as core
from paddle.distributed.fleet.meta_optimizers.common import OpRole

from .base_cost import Cost, CompOpCost, CommContext
from ..operators.common import get_distributed_operator_impl_container
from ..utils import print_program_with_dist_attr
from ..dist_tensor import DistributedTensor

cached_reshard_cost = {}
cached_dist_op_cost = {}
var_memory = {}


class CostEstimator:
    _sepical_op_type = ["fused_attention", "fused_feedforward"]

    def __init__(self,
                 program,
                 cluster,
                 mode="modeling",
                 rank=None,
                 loop_count=10):
        self._program = program
        self._cluster = cluster
        self._check_mode(mode)
        self._mode = mode
        self._rank = rank if rank is not None else paddle.distributed.get_rank()
        self._loop_count = loop_count
        self._global_cost = Cost()
        self._local_cost_mapping = {}
        # delta cost will be implemented in the future
        self._detailed_cost = OrderedDict(
        )  # {`op_id`: {"reshard": [], "dist_op": [], "local_cost": local_cost}}}
        self._bubble_time_mapping = {}
        self._ordered_ops = []

    @property
    def loop_count(self):
        return self._loop_count

    @property
    def detailed_cost(self):
        return self._detailed_cost

    @property
    def program(self):
        return self._program

    @property
    def rank(self):
        return self._rank

    @property
    def dist_context(self):
        return self._dist_context

    @property
    def cluster(self):
        return self._cluster

    @property
    def mode(self):
        return self._mode

    @property
    def global_cost(self):
        max_time = 0
        memory = 0
        flops = 0
        for rank in self._local_cost_mapping:
            cost = self._local_cost_mapping[rank]
            if cost.time > max_time:
                max_time = cost.time
            memory += cost.memory
            flops += cost.flops
        self._global_cost.time = max_time
        self._global_cost.memory = memory
        self._global_cost.flops = flops
        return self._global_cost

    def get_op_cost(self):
        return 0

    def local_cost(self, rank=None):
        rank = self.rank if rank is None else rank
        if rank not in self._local_cost_mapping:
            self._local_cost_mapping[rank] = Cost()

        return self._local_cost_mapping[rank]

    def local_bubble_time(self, rank=None):
        rank = self.rank if rank is None else rank
        return self._bubble_time_mapping[rank]

    def get_tensor_cost(self):
        return 0

    def _check_mode(self, mode):
        if mode not in ["modeling", "profiling"]:
            raise ValueError(
                "Just support modeling and profiling, but got {}".format(mode))

    def _is_special_var_name(self, var_name):
        special_var_name = ["lod_tensor_blocking_queue_0"]
        if var_name in special_var_name:
            return True
        return False

    def _estimate_core(self, dist_context, resharder, block):
        global cached_reshard_cost
        global cached_dist_op_cost
        from ..reshard import get_var_with_recursion
        ops = block.ops
        loop_count = None
        if block.desc.id != self.program.global_block().desc.id:
            loop_count = self.loop_count
        else:
            loop_count = 1
        for i in range(loop_count):
            for idx, op in enumerate(ops):
                if idx not in cached_reshard_cost:
                    cached_reshard_cost[idx] = {}
                if idx not in cached_dist_op_cost:
                    cached_dist_op_cost[idx] = {}
                self._detailed_cost[idx] = OrderedDict()
                # if in the while sub block, the detail of cost is the last cost
                detail = self._detailed_cost[idx]
                detail["reshard_cost"] = OrderedDict()  #
                detail["dist_op_cost"] = []
                if int(op.attr('op_role')) == int(OpRole.Optimize):
                    continue
                if op.type in [
                        "create_py_reader", "create_double_buffer_reader",
                        "read"
                ]:
                    continue

                # NOTE: It does not support nested loop and just supports while op when op has sub block now.
                if op.type == "while":
                    while_block = self.program.blocks[op.attr("sub_block").id]
                    self._estimate_core(dist_context, resharder, while_block)
                    continue

                dist_op = dist_context.get_dist_op_for_program(op)
                op_dist_attr = dist_op.dist_attr
                processes = op_dist_attr.process_mesh.processes

                dist_op_cost_key = ""
                for var_name in op.output_arg_names:
                    output_dims_mapping = op_dist_attr.get_output_dims_mapping(
                        var_name)
                    dist_op_cost_key += self._convert_pm_and_dm_to_str(
                        op_dist_attr.process_mesh, output_dims_mapping)

                for var_name in op.input_arg_names:
                    if self._is_special_var_name(var_name):
                        continue
                    var = get_var_with_recursion(var_name, block, self.program)
                    dist_tensor = dist_context.get_dist_tensor_for_program(var)
                    tensor_dist_attr = dist_tensor.dist_attr

                    op_key = self._convert_pm_and_dm_to_str(
                        op_dist_attr.process_mesh,
                        op_dist_attr.get_input_dims_mapping(var_name))
                    tensor_key = self._convert_pm_and_dm_to_str(
                        tensor_dist_attr.process_mesh,
                        tensor_dist_attr.dims_mapping)
                    key = var_name + op_key + tensor_key
                    dist_op_cost_key += op_key
                    if key not in cached_reshard_cost[idx]:
                        reshard_cost = resharder.get_cost(op, var, self.cluster)
                        cached_reshard_cost[idx][key] = reshard_cost
                    else:
                        reshard_cost = cached_reshard_cost[idx][key]

                    # calc reshard cost
                    if reshard_cost is not None:
                        detail["reshard_cost"][var_name] = reshard_cost

                        comm_costs = reshard_cost[0]
                        local_comp_cost = reshard_cost[1]
                        for comm_cost in comm_costs:
                            # time is cumulative in global cost and local cost, but memory and flops just are cumulative in global cost.
                            # comm sync
                            for item in comm_cost:
                                group_ranks, cost = item
                                max_time = None
                                cost_time = {}
                                for rank in group_ranks:
                                    rank_cost = self.local_cost(rank)
                                    cost_time[rank] = rank_cost.time
                                    if max_time is None:
                                        max_time = rank_cost.time
                                    else:
                                        if max_time < rank_cost.time:
                                            max_time = rank_cost.time

                                for rank in group_ranks:
                                    self.local_cost(
                                        rank).time = max_time + cost.time
                                    if rank not in self._bubble_time_mapping:
                                        self._bubble_time_mapping[rank] = 0

                                    self._bubble_time_mapping[rank] += (
                                        max_time - cost_time[rank])

                        for rank in local_comp_cost:
                            for comp_cost in local_comp_cost[rank]:
                                self.local_cost(rank).time += comp_cost.time

                # calc dist op cost
                if dist_op_cost_key not in cached_dist_op_cost[idx]:
                    container = get_distributed_operator_impl_container(
                        op_dist_attr.impl_type)
                    dist_impl = container.impls[op_dist_attr.impl_idx]
                    dist_op_cost = dist_impl.calc_cost(op.attr('op_role'),
                                                       dist_op, dist_context,
                                                       self.cluster)
                    cached_dist_op_cost[idx][dist_op_cost_key] = dist_op_cost
                else:
                    dist_op_cost = cached_dist_op_cost[idx][dist_op_cost_key]

                if dist_op_cost is None:
                    detail["dist_op_cost"] = dist_op_cost
                    assert dist_op.serial_op.type in CostEstimator._sepical_op_type
                    continue
                for item in dist_op_cost:
                    if isinstance(item, list):
                        # comm sync
                        for comm_op_cost in item:
                            max_time = None
                            cost_time = {}
                            group_ranks = comm_op_cost.group_ranks
                            for rank in comm_op_cost.group_ranks:
                                rank_cost = self.local_cost(rank)
                                cost_time[rank] = rank_cost.time
                                if max_time is None:
                                    max_time = rank_cost.time
                                else:
                                    if max_time < rank_cost.time:
                                        max_time = rank_cost.time
                            for rank in group_ranks:
                                self.local_cost(
                                    rank).time = max_time + comm_op_cost.time
                                if rank not in self._bubble_time_mapping:
                                    self._bubble_time_mapping[rank] = 0
                                self._bubble_time_mapping[rank] += (
                                    max_time - cost_time[rank])
                    elif isinstance(item, dict):
                        # op just one
                        for rank in processes:
                            # dp+pp+mp
                            if rank not in item:
                                continue
                            self.local_cost(rank).time += item[rank].time

    def prepare(self):
        self._global_cost = Cost()
        self._local_cost_mapping = {}
        self._detailed_cost = OrderedDict()
        self._bubble_time_mapping = {}

    def _calculate_bytes(self, sizes, dtype):
        if sizes:
            total_count = reduce(lambda x, y: x * y, sizes)
        else:
            total_count = 0

        if dtype == paddle.float64 or dtype == paddle.int64:
            dtype_factor = 8
        elif dtype == paddle.float32 or dtype == paddle.int32:
            dtype_factor = 4
        elif dtype == paddle.float16 or dtype == paddle.bfloat16 \
            or dtype == paddle.int16:
            dtype_factor = 2
        elif dtype == paddle.int8 or dtype == paddle.uint8:
            dtype_factor = 1
        else:
            dtype_factor = 8

        memory = total_count * dtype_factor
        return memory

    def _estimate_max_memory(self, dist_context):
        # This estimation will be improved
        memories = {}
        for block in self.program.blocks:
            for tensor in block.vars.values():
                dist_tensor = dist_context.get_dist_tensor_for_program(tensor)
                if dist_tensor is None:
                    raise ValueError("Cannot find the dist tensor of {}".format(
                        tensor.name))
                serial_tensor = dist_tensor.serial_tensor
                if serial_tensor.type == core.VarDesc.VarType.READER \
                    or serial_tensor.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY \
                    or serial_tensor.type == core.VarDesc.VarType.STEP_SCOPES:
                    continue
                else:
                    dist_attr = dist_tensor.dist_attr
                    if dist_attr.process_mesh is None:
                        continue
                    processes = dist_attr.process_mesh.processes
                    for process in processes:
                        sizes = dist_tensor.local_sizes()
                        dtype = dist_tensor.serial_tensor.dtype
                        if sizes and not all(size >= 0 for size in sizes):
                            print(dist_tensor.serial_tensor.name,
                                  sizes,
                                  dtype,
                                  flush=True)
                        if process in memories:
                            memories[process] += self._calculate_bytes(
                                sizes, dtype)
                        else:
                            memories[process] = self._calculate_bytes(
                                sizes, dtype)
        # Calculate the max memory in all ranks
        max_memory = max(memories.values())
        return max_memory

    @property
    def ordered_ops(self):
        if not self._ordered_ops:
            for block in self.program.blocks:
                for idx, op in enumerate(block.ops):
                    self._ordered_ops.append([op.desc.id(), op])
            self._ordered_ops.sort(key=lambda x: x[0])
        return self._ordered_ops

    def _convert_pm_and_dm_to_str(self, process_mesh, dims_mapping):

        processes = ",".join([str(x) for x in process_mesh.processes])
        topology = ",".join([str(x) for x in process_mesh.topology])
        dims_mapping = ",".join([str(x) for x in dims_mapping])
        result = processes + "." + topology + "." + dims_mapping
        return result

    def _estimate_max_memory_by_dist_op(self, dist_context):
        # This estimation will be improved, now reshard and inplace are not considered.
        # Persist var is not free.
        global var_memory
        memories = {}
        max_memories = {}
        var_info = {
        }  # var_name: [[process_mesh, dims_mapping], [id]], [[process_mesh, dims_mapping], [id]]}
        for op_id, op in self.ordered_ops:
            dist_op = dist_context.get_dist_op_for_program(op)
            process_mesh = dist_op.dist_attr.process_mesh
            for var_name in op.input_arg_names:
                input_dims_mapping = dist_op.dist_attr.get_input_dims_mapping(
                    var_name)
                if var_name not in var_info:
                    var_info[var_name] = {}
                if var_name not in var_memory:
                    var_memory[var_name] = {}
                key = self._convert_pm_and_dm_to_str(process_mesh,
                                                     input_dims_mapping)
                if key not in var_info[var_name]:
                    var_info[var_name][key] = {}
                if key not in var_memory[var_name]:
                    var_memory[var_name][key] = {}
                if "memory" not in var_info[var_name][key]:
                    if "memory" not in var_memory[var_name][key]:
                        var = dist_op.get_serial_input(var_name)
                        global_sizes = var.shape
                        dtype = var.dtype
                        sizes = DistributedTensor.get_local_sizes(
                            global_sizes, input_dims_mapping,
                            process_mesh.topology, process_mesh.processes,
                            False)
                        var_memory[var_name][key][
                            "memory"] = self._calculate_bytes(sizes, dtype)
                    var_info[var_name][key]["memory"] = var_memory[var_name][
                        key]["memory"]
                if "position" not in var_info[var_name][key]:
                    var_info[var_name][key]["position"] = []
                var_info[var_name][key]["position"].append(op_id)

            for var_name in op.output_arg_names:
                output_dims_mapping = dist_op.dist_attr.get_output_dims_mapping(
                    var_name)
                if var_name not in var_info:
                    var_info[var_name] = {}
                if var_name not in var_memory:
                    var_memory[var_name] = {}
                key = self._convert_pm_and_dm_to_str(process_mesh,
                                                     output_dims_mapping)
                if key not in var_info[var_name]:
                    var_info[var_name][key] = {}
                if key not in var_memory[var_name]:
                    var_memory[var_name][key] = {}
                if "memory" not in var_info[var_name][key]:
                    if "memory" not in var_memory[var_name][key]:
                        var = dist_op.get_serial_output(var_name)
                        global_sizes = var.shape
                        dtype = var.dtype
                        sizes = DistributedTensor.get_local_sizes(
                            global_sizes, output_dims_mapping,
                            process_mesh.topology, process_mesh.processes,
                            False)
                        var_memory[var_name][key][
                            "memory"] = self._calculate_bytes(sizes, dtype)
                    var_info[var_name][key]["memory"] = var_memory[var_name][
                        key]["memory"]
                if "position" not in var_info[var_name][key]:
                    var_info[var_name][key]["position"] = []
                var_info[var_name][key]["position"].append(op_id)

        has_used_vars = set()
        for op_id, op in self.ordered_ops:
            can_free_memories = {}
            can_free_vars = set()
            dist_op = dist_context.get_dist_op_for_program(op)
            process_mesh = dist_op.dist_attr.process_mesh
            for var_name in op.input_arg_names:
                input_dims_mapping = dist_op.dist_attr.get_input_dims_mapping(
                    var_name)
                key = self._convert_pm_and_dm_to_str(process_mesh,
                                                     input_dims_mapping)
                has_used_var = var_name + key
                var = dist_op.get_serial_input(var_name)
                # not used
                if var_name + key not in has_used_vars:
                    has_used_vars.add(has_used_var)
                    for process in process_mesh.processes:
                        if process not in memories:
                            memories[process] = 0
                        memories[process] += var_info[var_name][key]["memory"]
                # used
                else:
                    if op_id == var_info[var_name][key]["position"][-1]:
                        if has_used_var not in can_free_vars:
                            can_free_vars.add(has_used_var)
                            if not var.persistable:
                                for process in process_mesh.processes:
                                    if process not in can_free_memories:
                                        can_free_memories[process] = 0
                                    can_free_memories[process] += var_info[
                                        var_name][key]["memory"]

            for var_name in op.output_arg_names:
                output_dims_mapping = dist_op.dist_attr.get_output_dims_mapping(
                    var_name)
                key = self._convert_pm_and_dm_to_str(process_mesh,
                                                     output_dims_mapping)
                has_used_var = var_name + key
                var = dist_op.get_serial_output(var_name)
                # not used
                if var_name + key not in has_used_vars:
                    has_used_vars.add(has_used_var)
                    for process in process_mesh.processes:
                        if process not in memories:
                            memories[process] = 0
                        memories[process] += var_info[var_name][key]["memory"]
                # used
                else:
                    if op_id == var_info[var_name][key]["position"][-1]:
                        if has_used_var not in can_free_vars:
                            can_free_vars.add(has_used_var)
                            if not var.persistable:
                                for process in process_mesh.processes:
                                    if process not in can_free_memories:
                                        can_free_memories[process] = 0
                                    can_free_memories[process] += var_info[
                                        var_name][key]["memory"]

            # calc peak memory
            for process in memories:
                if process not in max_memories:
                    max_memories[process] = memories[process]
                else:
                    if memories[process] > max_memories[process]:
                        max_memories[process] = memories[process]

            # free memory
            for process in can_free_memories:
                if process in memories:
                    memories[process] -= can_free_memories[process]
                    assert memories[process] >= 0
        # Calculate the max memory in all ranks
        max_memory = max(max_memories.values())
        return max_memory

    def estimate(self, dist_context, resharder=None):
        self.prepare()

        from ..reshard import Resharder
        resharder = Resharder(self.program, None, self.rank, dist_context,
                              []) if resharder is None else resharder

        block = self.program.global_block()
        self._estimate_core(dist_context, resharder, block)
        return self.global_cost
