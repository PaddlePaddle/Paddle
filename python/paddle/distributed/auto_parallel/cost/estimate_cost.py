#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid.core as core
from paddle.distributed.fleet.meta_optimizers.common import OpRole

from .base_cost import Cost, CompOpCost, CommContext
from ..operators.common import get_distributed_operator_impl_container
from ..utils import print_program_with_dist_attr


class CostEstimator:

    def __init__(self,
                 program,
                 cluster=None,
                 dist_context=None,
                 mode="modeling"):
        self._program = program
        self._cluster = cluster
        self._dist_context = dist_context
        self._check_mode(mode)
        self._mode = mode
        self._global_cost = None
        self._local_cost = {}

    @property
    def program(self):
        return self._program

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
        return self._global_cost

    @property
    def local_cost(self):
        return self._local_cost

    def get_op_cost(self):
        return 0

    def get_tensor_cost(self):
        return 0

    def get_global_cost(self):
        return 0

    def get_local_cost(self, rank=None):
        return 0

    def _check_mode(self, mode):
        if mode not in ["modeling", "profiling"]:
            raise ValueError(
                "Just support modeling and profiling, but got {}".format(mode))

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

    def _estimate_max_memory_by_dist_tensor(self, dist_context):
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
                        print(tensor.name, tensor.type, dist_attr, flush=True)
                        continue
                    processes = dist_attr.process_mesh.processes
                    for process in processes:
                        sizes = dist_tensor.local_sizes()
                        dtype = dist_tensor.serial_tensor.dtype
                        if process in memories:
                            memories[process] += self._calculate_bytes(
                                sizes, dtype)
                        else:
                            memories[process] = self._calculate_bytes(
                                sizes, dtype)
        # Calculate the max memory in all ranks
        max_memory = max(memories.values())
        return max_memory

    def _estimate_max_memory_by_dist_op(self, dist_context):
        # This estimation will be improved, now reshard and inplace are not considered.
        # Persist var is not free. 
        def _convert_pm_and_dm_to_str(process_mesh, dims_mapping):
            processes = ",".join([str(x) for x in process_mesh.processes])
            topology = ",".join([str(x) for x in process_mesh.topology])
            dims_mapping = ",".join([str(x) for x in dims_mapping])
            result = processes + topology + dims_mapping
            return result

        memories = {}
        max_memories = {}
        var_as_input_info = {} # var_name: [[process_mesh, dims_mapping], [id]], [[process_mesh, dims_mapping], [id]]}
        for op_id, op in self._ordered_ops:
            dist_op = dist_context.get_dist_op_for_program(op)
            process_mesh = dist_op.dist_attr.process_mesh
            for var_name in op.input_arg_names:
                input_dims_mapping = dist_op.dist_attr.get_input_dims_mapping(var_name)
                if var_name not in var_as_input_info:
                    var_as_input_info[var_name] = {}
                key = _convert_pm_and_dm_to_str(process_mesh, input_dims_mapping)
                if key not in var_as_input_info[var_name]:
                    var_as_input_info[var_name][key] = {}
                # it is even partition now
                if "memory" not in var_as_input_info[var_name][key]:
                    var = dist_op.get_serial_input(var_name)
                    global_sizes = var.shape
                    dtype = var.dtype
                    sizes = DistributedTensor.get_local_sizes(global_sizes, dims_mapping, process_mesh.topology, process_mesh.processes)
                    var_as_input_info[var_name][key]["memory"] = self._calculate_bytes(sizes, dtype)
                if "position" not in var_as_input_info[var_name][key]:
                    var_as_input_info[var_name][key]["position"] = []
                var_as_input_info[var_name][key]["position"].append(op_id)

            for var_name in op.out_arg_names:
                output_dims_mapping = dist_op.dist_attr.get_output_dims_mapping(var_name)
                if var_name not in var_as_input_info:
                    var_as_input_info[var_name] = {}
                key = _convert_pm_and_dm_to_str(process_mesh, output_dims_mapping)
                if key not in var_as_input_info[var_name]:
                    var_as_input_info[var_name][key] = {}
                if "memory" not in var_as_input_info[var_name][key]:
                    var = dist_op.get_serial_output(var_name)
                    global_sizes = var.shape
                    sizes = DistributedTensor.get_local_sizes(global_sizes, dims_mapping, process_mesh.topology, process_mesh.processes)
                    dtype = var.dtype
                    var_as_input_info[var_name][key]["memory"] = self._calculate_bytes(sizes, dtype)
                if "position" not in var_as_input_info[var_name][key]:
                    var_as_input_info[var_name][key]["position"] = []
                var_as_input_info[var_name][key]["position"].append(op_id)

        has_uesd_vars = set()
        for op_id, op in self.ordered_ops:
            can_free_memories = {}
            can_free_vars = set()
            dist_op = dist_context.get_dist_op_for_program(op)
            process_mesh = dist_op.dist_attr.process_mesh
            for var_name in op.input_arg_names:
                input_dims_mapping = dist_op.dist_attr.get_input_dims_mapping(var_name)
                key = _convert_pm_and_dm_to_str(process_mesh, input_dims_mapping)
                has_used_var = var_name+key
                var = dist_op.get_serial_input(var_name)
                # not used
                if var_name+key not in has_used_vars:
                    has_used_vars.add(has_used_var)
                    for process in process_mesh.processes:
                        if process not in memories:
                            memories[process] = 0
                        memories[process] += var_as_input_info[var_name][key]["memory"]
                # used
                else:
                    if op_id == var_as_input_info[var_name][key]["position"][-1]:
                        if has_used_var not in can_free_vars:
                            can_free_vars.add(has_used_var)
                            if not var.persistable:
                                for process in process_mesh.processes:
                                    if process not in can_free_memories:
                                        can_free_memories[process] = 0
                                    can_free_memories[process] += var_as_input_info[var_name][key]["memory"]

            for var_name in op.output_arg_names:
                output_dims_mapping = dist_op.dist_attr.get_output_dims_mapping(var_name)
                key = _convert_pm_and_dm_to_str(process_mesh, output_dims_mapping)
                has_used_var = var_name+key
                var = dist_op.get_serial_output(var_name)
                # not used
                if var_name+key not in has_used_vars:
                    has_used_vars.add(has_used_var)
                    for process in process_mesh.processes:
                        if process not in memories:
                            memories[process] = 0
                        memories[process] += var_as_input_info[var_name][key]["memory"]
                # used
                else:
                    if op_id == var_as_input_info[var_name][key]["position"][-1]:
                        if has_used_var not in can_free_vars:
                            can_free_vars.add(has_used_var)
                            if not var.persistable:
                                for process in process_mesh.processes:
                                    if process not in can_free_memories:
                                        can_free_memories[process] = 0
                                    can_free_memories[process] += var_as_input_info[var_name][key]["memory"]
            
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
