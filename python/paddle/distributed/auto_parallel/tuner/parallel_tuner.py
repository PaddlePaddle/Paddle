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
# limitations under the License.

import time
import math
import time
import datetime
import copy
import hashlib
import itertools
from collections import defaultdict
import numpy as np
import paddle
from ..process_mesh import ProcessMesh
from ..process_mesh import _flatten_nested_list
from ..completion import Completer
from ..parallelizer_v2 import Parallelizer
from ..dist_context import _node_id
from ..dist_op import DistributedOperator
from ..dist_attribute import TensorDistributedAttribute
from ..dist_attribute import OperatorDistributedAttribute
from ..operators.common import find_compatible_distributed_operator_impls
from ..operators.common import is_elementwise_op
from ..utils import print_program_with_dist_attr
from .trial import Trial, TrialStatus
from .tuner import Tuner
from .tunable_space import TunableSpace
from .tunable_variable import Boolean, IntRange
from ..cost import CostEstimator
from .tunable_variable import Boolean, IntRange
from ..process_group import get_all_process_groups


class ParallelTuner:

    def __init__(self,
                 dist_context,
                 mode="main",
                 max_trials=50,
                 tuner_id=None,
                 seed=None,
                 logger=None,
                 loop_count=10):
        self._loop_count = loop_count
        self._estimator = None
        self._dist_context = dist_context
        assert self._dist_context._is_initialized
        self._mode = mode
        self._cluster = self._dist_context.cluster
        self._num_machines = self._cluster.get_num_machines()
        self._num_devices_per_machine = self._cluster.get_num_devices_per_machine(
        )
        self._space = TunableSpace()
        self._objective = "time"
        self._direction = "min"
        self._max_trials = max_trials
        self._tuner_id = tuner_id
        self._seed = seed if seed is not None else 9999
        # for pipeline random
        # np.random.seed(999)

        print("seed",
              self._seed,
              "mode",
              self._mode,
              "num_machies",
              self._num_machines,
              "num_devices_per_machine",
              self._num_devices_per_machine,
              flush=True)
        self._seed_state = self._seed
        self._logger = logger
        self._max_collisions = 3
        self._tried_values = set()
        self._num_trials = 0
        self._rng = np.random.default_rng(self._seed)
        self._materialized_for_all_ranks = False

        # Search the op types in the include_op_types,
        # and will search all op types if it is empty.
        # self._include_op_types = ["matmul_v2", "matmul", "mul", "embedding"]
        self._include_op_types = []
        # Exclude the op types in the exclude_op_types
        # from the search list.
        self._exclude_op_types = []
        # The final dist ops will be searched after considering
        # the include_op_types and exclude_op_types.
        self._concerned_dist_ops = {}

        self._op_id_to_dist_attr_candidates = defaultdict(list)
        self._cached_dims_mapping_candidates = {}
        self._cached_candidates_info = defaultdict(list)

        self._special_ops = [
            "create_py_reader", "create_double_buffer_reader", "read", "while",
            "read_from_array", "write_to_array"
        ]
        # self._special_tensors = [
        #     "lod_tensor_blocking_queue_0", "create_py_reader_0",
        #     "double_buffer_0"
        # ]

        # Each parallel strategy has two elements. The First one is for distributed tensors,
        # the second element is for distributed tensors, the third element is for process meshes.
        self._init_parallel_strategy = [None, None, None]
        self._best_parallel_strategy = [None, None, None]

        self._completer = Completer(self._dist_context)

        self._parallelizer = Parallelizer(self._mode, self._completer,
                                          self._dist_context)

        self._is_ernie = False

    def _generate_combination(self,
                              elements,
                              target,
                              idx,
                              partial_candidate,
                              candidates,
                              num_candidates=None):
        if target == 0:
            candidates.append(copy.deepcopy(partial_candidate))
            return

        if target < 0 or idx == len(elements) \
            or len(candidates) > num_candidates:
            return

        # Use
        partial_candidate.append(elements[idx])
        self._generate_combination(elements, target - elements[idx], idx,
                                   partial_candidate, candidates,
                                   num_candidates)
        # Not use
        partial_candidate.pop()
        self._generate_combination(elements, target, idx + 1, partial_candidate,
                                   candidates, num_candidates)

    def _permute_combination(self,
                             combination,
                             target,
                             check,
                             partial_candidate,
                             candidates,
                             num_candidates=None,
                             skip_prob=None):
        if num_candidates is not None \
            and len(candidates) == num_candidates:
            return

        if len(partial_candidate) == len(combination):
            candidates.append(partial_candidate)
            return

        for i in range(len(combination)):
            if check[i] == 1:
                continue
            if self._rng.choice([True, False], p=[skip_prob, 1 - skip_prob]):
                continue
            if i > 0 and combination[i] == combination[i - 1] \
                and check[i -1] == 0:
                continue
            check[i] = 1
            self._permute_combination(combination, target, check,
                                      partial_candidate + [combination[i]],
                                      candidates, num_candidates, skip_prob)
            check[i] = 0

    def _partition_number(self, target):
        log2_target = int(math.log2(target))
        elements = [pow(2, i) for i in range(log2_target)]
        if pow(2, log2_target) == target:
            elements.append(target)
        seed_candidates = []
        num_seed_candidates = 1000
        partial_results = []
        self._generate_combination(elements, target, 0, partial_results,
                                   seed_candidates, num_seed_candidates)

        candidates = []
        for seed_candidate in seed_candidates:
            cur_candidates = []
            num_cur_candidates = 16
            seed_candidate.sort()
            check = [0 for i in range(len(seed_candidate))]
            if target <= 8:
                skip_prob = 0.0
            else:
                skip_prob = (len(seed_candidate) / target)
            self._permute_combination(seed_candidate, target, check, [],
                                      cur_candidates, num_cur_candidates,
                                      skip_prob)
            candidates.extend(cur_candidates)
        return candidates

    def _partition_devices(self, num_machines, num_devices_per_machine):
        inter_node_partitions = self._partition_number(num_machines)
        intra_node_partitions = self._partition_number(num_devices_per_machine)
        return inter_node_partitions, intra_node_partitions

    def _generate_process_mesh_list(self, inter_node_partition,
                                    intra_node_partition):
        process_mesh_list = []
        start_row = 0
        start_col = 0
        for m in inter_node_partition:
            start_col = 0
            for n in intra_node_partition:
                process_mesh = []
                for p in range(m):
                    start = (start_row +
                             p) * self._num_devices_per_machine + start_col
                    tmp = []
                    for q in range(n):
                        tmp.append(start + q)
                    process_mesh.append(tmp)
                process_mesh_list.append(copy.deepcopy(process_mesh))
                start_col += n
            start_row += m
        return process_mesh_list

    def _generate_dims_mapping_candidates_helper(self, dims_mapping, dims_list,
                                                 start, visited, candidates):
        if start == len(dims_mapping) or all(visited):
            candidates.append(copy.deepcopy(dims_mapping))
            return

        for idx, dim in enumerate(dims_list):
            if visited[idx] == False:
                dims_mapping[start] = dim
                visited[idx] = True
                self._generate_dims_mapping_candidates_helper(
                    dims_mapping, dims_list, start + 1, visited, candidates)
                visited[idx] = False
        dims_mapping[start] = -1
        self._generate_dims_mapping_candidates_helper(dims_mapping, dims_list,
                                                      start + 1, visited,
                                                      candidates)

    def _generate_dims_mapping_candidates(self, dims_mapping_len,
                                          process_mesh_len):
        assert dims_mapping_len >= 1 and process_mesh_len >= 1
        key = (dims_mapping_len, process_mesh_len)
        if key in self._cached_dims_mapping_candidates:
            return self._cached_dims_mapping_candidates[key]
        candidates = []
        dims_mapping = [-1 for i in range(dims_mapping_len)]
        dims_list = [i for i in range(process_mesh_len)]
        visited = [False for i in range(process_mesh_len)]
        self._generate_dims_mapping_candidates_helper(dims_mapping, dims_list,
                                                      0, visited, candidates)
        self._cached_dims_mapping_candidates[key] = candidates
        return candidates

    def _generate_dist_attr_candidates(self, op_id, dist_op):
        # For now, only allow the process meshes have two dimensions
        process_mesh_len = 2
        serial_op = dist_op.serial_op
        op_dist_attr = dist_op.dist_attr
        if serial_op.type in self._special_ops:
            return [copy.deepcopy(op_dist_attr)]
        key = []
        key.append(serial_op.type)
        for input_name in serial_op.input_names:
            key.append(input_name)
            for input_arg_name in serial_op.input(input_name):
                key.append(
                    len(op_dist_attr.get_input_dims_mapping(input_arg_name)))
        for output_name in serial_op.output_names:
            key.append(output_name)
            for output_arg_name in serial_op.output(output_name):
                key.append(
                    len(op_dist_attr.get_output_dims_mapping(output_arg_name)))
        key = tuple(key)

        if key in self._cached_candidates_info:
            cached_dist_attr_candidates = []
            cached_input_arg_names = self._cached_candidates_info[key][0]
            cached_output_arg_names = self._cached_candidates_info[key][1]
            for cached_dist_attr in self._cached_candidates_info[key][2]:
                new_op_dist_attr = copy.deepcopy(dist_op.dist_attr)
                i = 0
                for input_name in serial_op.input_names:
                    for input_arg_name in serial_op.input(input_name):
                        cached_dims_mapping = cached_dist_attr.get_input_dims_mapping(
                            cached_input_arg_names[i])
                        new_op_dist_attr.set_input_dims_mapping(
                            input_arg_name, cached_dims_mapping)
                        i += 1
                i = 0
                for output_name in serial_op.output_names:
                    for output_arg_name in serial_op.output(output_name):
                        cached_dims_mapping = cached_dist_attr.get_output_dims_mapping(
                            cached_output_arg_names[i])
                        new_op_dist_attr.set_output_dims_mapping(
                            output_arg_name, cached_dims_mapping)
                        i += 1
                cached_dist_attr_candidates.append(new_op_dist_attr)
            return cached_dist_attr_candidates

        # cached_candidates_info = []
        input_arg_names = []
        for input_name in serial_op.input_names:
            for input_arg_name in serial_op.input(input_name):
                input_arg_names.append(input_arg_name)
        self._cached_candidates_info[key].append(input_arg_names)
        # cached_candidates_info.append(input_arg_names)
        output_arg_names = []
        for output_name in serial_op.output_names:
            for output_arg_name in serial_op.output(output_name):
                output_arg_names.append(output_arg_name)
        self._cached_candidates_info[key].append(output_arg_names)
        # cached_candidates_info.append(output_arg_names)

        new_op_dist_attr = copy.deepcopy(dist_op.dist_attr)
        # Find valid dims_mapping candidates for inputs
        input_names = []
        dims_mapping_generated = []
        inputs_dist_attrs = op_dist_attr.inputs_dist_attrs
        for tensor_name, tensor_dist_attr in inputs_dist_attrs.items():
            original_dims_mapping = tensor_dist_attr.dims_mapping
            dims_mapping_len = len(original_dims_mapping)
            input_names.append(tensor_name)
            if dims_mapping_len < 1:
                dims_mapping_generated.append(
                    [copy.deepcopy(original_dims_mapping)])
            else:
                dims_mapping_generated.append(
                    self._generate_dims_mapping_candidates(
                        dims_mapping_len, process_mesh_len))
        input_dims_mapping_candidates = []
        for dims_mapping_list in itertools.product(*dims_mapping_generated):
            dims_mapping_list = list(dims_mapping_list)
            assert len(dims_mapping_list) == len(input_names)
            for i, dims_mapping in enumerate(dims_mapping_list):
                new_op_dist_attr.set_input_dims_mapping(input_names[i],
                                                        dims_mapping)
            new_dist_op = DistributedOperator(dist_op.serial_op,
                                              new_op_dist_attr)
            dist_op_impls = find_compatible_distributed_operator_impls(
                new_dist_op, fwd=True)
            if dist_op_impls is not None:
                input_dims_mapping_candidates.append(dims_mapping_list)

        # Find valid dims_mapping candidates for outputs
        output_names = []
        dims_mapping_generated = []
        outputs_dist_attrs = op_dist_attr.outputs_dist_attrs
        for tensor_name, tensor_dist_attr in outputs_dist_attrs.items():
            original_dims_mapping = tensor_dist_attr.dims_mapping
            dims_mapping_len = len(original_dims_mapping)
            output_names.append(tensor_name)
            if dims_mapping_len < 1:
                dims_mapping_generated.append(
                    [copy.deepcopy(original_dims_mapping)])
            else:
                dims_mapping_generated.append(
                    self._generate_dims_mapping_candidates(
                        dims_mapping_len, process_mesh_len))
        output_dims_mapping_candidates = []
        for dims_mapping_list in itertools.product(*dims_mapping_generated):
            dims_mapping_list = list(dims_mapping_list)
            assert len(dims_mapping_list) == len(output_names)
            for i, dims_mapping in enumerate(dims_mapping_list):
                new_op_dist_attr.set_output_dims_mapping(
                    output_names[i], dims_mapping)
            new_dist_op = DistributedOperator(dist_op.serial_op,
                                              new_op_dist_attr)
            dist_op_impls = find_compatible_distributed_operator_impls(
                new_dist_op, fwd=False)
            if dist_op_impls is not None:
                output_dims_mapping_candidates.append(dims_mapping_list)

        if not input_dims_mapping_candidates and output_dims_mapping_candidates:
            inout_dims_mapping_generated = [[[[-2]]],
                                            output_dims_mapping_candidates]
        elif input_dims_mapping_candidates and not output_dims_mapping_candidates:
            inout_dims_mapping_generated = [
                input_dims_mapping_candidates, [[[-2]]]
            ]
        elif not input_dims_mapping_candidates and not output_dims_mapping_candidates:
            inout_dims_mapping_generated = [[[[-2]]], [[[-2]]]]
        else:
            inout_dims_mapping_generated = [
                input_dims_mapping_candidates, output_dims_mapping_candidates
            ]
        # Find valid dims_mapping generated for both inputs and outputs
        cached_dist_attr_candidates = []
        for inout_dims_mapping_list in itertools.product(
                *inout_dims_mapping_generated):
            assert len(inout_dims_mapping_list) == 2
            if input_dims_mapping_candidates:
                assert len(inout_dims_mapping_list[0]) == len(input_names)
            if output_dims_mapping_candidates:
                assert len(inout_dims_mapping_list[1]) == len(output_names)
            # set the dims_mappings for inputs
            for i, dims_mapping in enumerate(inout_dims_mapping_list[0]):
                if dims_mapping != [-2]:
                    new_op_dist_attr.set_input_dims_mapping(
                        input_names[i], dims_mapping)
            # set the dims_mappings for outputs
            for i, dims_mapping in enumerate(inout_dims_mapping_list[1]):
                if dims_mapping != [-2]:
                    new_op_dist_attr.set_output_dims_mapping(
                        output_names[i], dims_mapping)
            new_dist_op = DistributedOperator(dist_op.serial_op,
                                              new_op_dist_attr)
            dist_op_impls = find_compatible_distributed_operator_impls(
                new_dist_op, partial=False)
            if dist_op_impls is None:
                continue
            for dist_op_impl in dist_op_impls:
                new_op_dist_attr.impl_type = dist_op_impl.type
                new_op_dist_attr.impl_idx = dist_op_impl.idx
                cached_dist_attr_candidates.append(
                    copy.deepcopy(new_op_dist_attr))
        self._cached_candidates_info[key].append(cached_dist_attr_candidates)
        return self._cached_candidates_info[key][2]
        # cached_candidates_info.append(cached_dist_attr_candidates)
        # return cached_candidates_info[2]

    def construct_space(self):
        inter_node_partitions, intra_node_partitions = self._partition_devices(
            self._num_machines, self._num_devices_per_machine)
        self._space.choice("inter_node_partitions",
                           inter_node_partitions,
                           default=inter_node_partitions[0])
        self._space.choice("intra_node_partitions",
                           intra_node_partitions,
                           default=intra_node_partitions[0])

        dist_ops = self._dist_context._dist_ops_for_program
        for op_id, dist_op in dist_ops.items():
            op_type = dist_op.serial_op.type
            if self._include_op_types:
                if op_type in self._include_op_types:
                    self._concerned_dist_ops[op_id] = dist_op
            else:
                self._concerned_dist_ops[op_id] = dist_op

        for op_id, dist_op in self._concerned_dist_ops.items():
            op_type = dist_op.serial_op.type
            if op_type in self._exclude_op_types:
                del self._concerned_dist_ops[op_id]

        print("Number of the concered dist ops",
              len(self._concerned_dist_ops),
              flush=True)
        search_space = 1
        for op_id, dist_op in self._concerned_dist_ops.items():
            op_dist_attr_candidates = self._generate_dist_attr_candidates(
                op_id, dist_op)
            print("concerned dist ops",
                  dist_op.serial_op.type,
                  len(op_dist_attr_candidates),
                  flush=True)
            search_space *= len(op_dist_attr_candidates)
            self._space.choice(str(op_id),
                               op_dist_attr_candidates,
                               default=op_dist_attr_candidates[0])
        # print("Number of the search space", f"{float(search_space): .2E}", flush=True)

    def _compute_values_hash(self, values):
        keys = sorted(values.keys())
        s = "".join(str(k) + "=" + str(values[k]) for k in keys)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]

    def _random_values(self):
        space = TunableSpace()
        collisions = 0
        while True:
            for v in self._space.variables.values():
                space._register(v)
                space.values[v.name] = v.random(self._seed_state)
                self._seed_state += 1
            values = space.values
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_values:
                collisions += 1
                if collisions > self._max_collisions:
                    return None
                continue
            self._tried_values.add(values_hash)
            break
        return values

    def _populate_space(self):
        values = self._random_values()
        if values is None:
            return {"status": TrialStatus.STOPPED, "values": None}
        return {"status": TrialStatus.RUNNING, "values": values}

    def _create_trial(self):
        trial_id = "{{:0{}d}}".format(len(str(self._max_trials)))
        trial_id = trial_id.format(self._num_trials)

        if self._max_trials and self._num_trials >= self._max_trials:
            status = TrialStatus.STOPPED
            values = None
        else:
            results = self._populate_space()
            status = results["status"]
            values = results["values"]

        space = TunableSpace()
        space.variables = self._space.variables
        space.values = values
        trial = Trial(tunable_space=space, trial_id=trial_id, status=status)
        self._num_trials += 1
        return trial

    def _generate_pipeline_starts(self, process_mesh_list):
        total_ops = len(self._dist_context._dist_ops_for_program)
        total_stages = len(process_mesh_list)
        ops_per_stage = total_ops // total_stages
        if ops_per_stage == 0:
            return None
        # Compute the initial pipeline starts
        pipeline_starts = []
        start = 0
        pipeline_starts.append(0)
        # The pipeline_starts have total_stages+1 items, and
        # at least have 2 items.
        for _ in process_mesh_list:
            start += ops_per_stage
            pipeline_starts.append(start)
        pipeline_starts[-1] = total_ops
        # Adjust the pipeline starts by random selection
        directions = []
        sizes = []
        half_ops_per_stage = ops_per_stage // 2
        if half_ops_per_stage > 0 and total_stages > 1:
            new_pipeline_starts = []
            # Don't change the first start
            new_pipeline_starts.append(0)
            # Consider the starts except the first and the last one
            for _ in pipeline_starts[1:-1]:
                directions.append(Boolean("direction"))
                sizes.append(
                    IntRange("size",
                             start=0,
                             stop=half_ops_per_stage,
                             endpoint=True))
            for i, start in enumerate(pipeline_starts[1:-1]):
                direction = directions[i].random(self._seed)
                size = sizes[i].random(self._seed)
                if direction:
                    # Substract 1 from size to avoid the overlapping of new starts
                    new_start = start - (size - 1)
                else:
                    new_start = start + size
                new_pipeline_starts.append(new_start)
            # Don't change the last start
            new_pipeline_starts.append(pipeline_starts[-1])
            # Validate the new starts
            print("Adjusted pipeline starts",
                  new_pipeline_starts,
                  half_ops_per_stage,
                  pipeline_starts,
                  flush=True)
            for i, new_start in enumerate(new_pipeline_starts[1:]):
                assert new_start > new_pipeline_starts[i]
            return new_pipeline_starts
        else:
            print("Non-adjusted pipeline starts",
                  pipeline_starts,
                  half_ops_per_stage,
                  flush=True)
            return pipeline_starts

    def _apply_pipeline_partition(self, process_mesh_list):
        op_id_to_process_mesh = {}
        total_ops = len(self._dist_context._dist_ops_for_program)
        total_stages = len(process_mesh_list)
        ops_per_stage = total_ops // total_stages
        if ops_per_stage == 0:
            return None
        pipeline_starts = self._generate_pipeline_starts(process_mesh_list)
        start_idx = 1
        sorted_op_ids = sorted(self._dist_context._dist_ops_for_program.keys())
        for idx, op_id in enumerate(sorted_op_ids):
            if idx < pipeline_starts[start_idx]:
                op_id_to_process_mesh[op_id] = process_mesh_list[start_idx - 1]
            else:
                start_idx += 1
                op_id_to_process_mesh[op_id] = process_mesh_list[start_idx - 1]
        return op_id_to_process_mesh

    def _apply_pipeline_partition_random(self, process_mesh_list):
        op_id_to_process_mesh = {}
        total_ops = len(self._dist_context._dist_ops_for_program)
        total_stages = len(process_mesh_list)
        ops_per_stages = total_ops // total_stages
        if ops_per_stages == 0:
            return None

        random_times = total_stages - 1
        pipeline_starts = [0, total_ops]
        random_start = 0
        while random_start < total_stages - 1:
            partition_index = np.random.randint(1, total_ops)
            if partition_index not in pipeline_starts:
                pipeline_starts.append(partition_index)
                random_start += 1
        pipeline_starts.sort()

        start = 1
        sorted_op_ids = sorted(self._dist_context._dist_ops_for_program.keys())
        for idx, op_id in enumerate(sorted_op_ids):
            if idx < pipeline_starts[start]:
                op_id_to_process_mesh[op_id] = process_mesh_list[start - 1]
            else:
                start += 1
                op_id_to_process_mesh[op_id] = process_mesh_list[start - 1]
        return op_id_to_process_mesh

    def _amend_dist_attr(self):
        # 1) Reshape the process mesh of [1, x] to [x] or [x, 1] to [x],
        # and amend the corresponding dims_mapping.
        # 2) Set the dim_mapping to -1 when the shape cannot be divided
        # by the corresponding processes.
        for dist_op in self._dist_context._dist_ops_for_program.values():
            dist_attr = dist_op.dist_attr
            process_mesh = dist_attr.process_mesh
            if process_mesh is None:
                continue
            assert process_mesh.ndim == 2
            dim_of_one = None
            dim_of_other = None
            if process_mesh.topology[0] == 1:
                dim_of_one = 0
                dim_of_other = 1
            elif process_mesh.topology[1] == 1:
                dim_of_one = 1
                dim_of_other = 0

            if dim_of_one is not None:
                dist_attr.process_mesh = ProcessMesh(process_mesh.processes)
                self._dist_context.add_process_mesh(dist_attr.process_mesh)

            for arg_name in dist_attr.inputs_dist_attrs.keys():
                new_dims_mapping = []
                dims_mapping = dist_attr.get_input_dims_mapping(arg_name)
                for dim_mapping in dims_mapping:
                    if dim_mapping == dim_of_one:
                        new_dims_mapping.append(-1)
                    elif dim_mapping == dim_of_other:
                        new_dims_mapping.append(0)
                    else:
                        new_dims_mapping.append(dim_mapping)
                dist_attr.set_input_dims_mapping(arg_name, new_dims_mapping)

                dims_mapping = dist_attr.get_input_dims_mapping(arg_name)
                dynamic_dims = dist_attr.get_input_dynamic_dims(arg_name)
                process_mesh = dist_attr.process_mesh
                process_shape = process_mesh.topology
                tensor = dist_op.get_serial_input(arg_name)
                if dims_mapping:
                    tensor_shape = tensor.shape
                else:
                    continue
                for i, dim_mapping in enumerate(dims_mapping):
                    if dim_mapping != -1 \
                        and (tensor_shape[i] % process_shape[dim_mapping] != 0 \
                        or dynamic_dims[i] == 1):
                        dims_mapping[i] = -1
                    # it is a fix-bug
                    if dim_mapping != -1 \
                        and process_shape[dim_mapping] == 1:
                        dims_mapping[i] = -1

            for arg_name in dist_attr.outputs_dist_attrs.keys():
                new_dims_mapping = []
                dims_mapping = dist_attr.get_output_dims_mapping(arg_name)
                for dim_mapping in dims_mapping:
                    if dim_mapping == dim_of_one:
                        new_dims_mapping.append(-1)
                    elif dim_mapping == dim_of_other:
                        new_dims_mapping.append(0)
                    else:
                        new_dims_mapping.append(dim_mapping)
                dist_attr.set_output_dims_mapping(arg_name, new_dims_mapping)

                dims_mapping = dist_attr.get_output_dims_mapping(arg_name)
                dynamic_dims = dist_attr.get_output_dynamic_dims(arg_name)
                process_mesh = dist_attr.process_mesh
                process_shape = process_mesh.topology

                tensor = dist_op.get_serial_output(arg_name)
                # if arg_name == "fill_constant_batch_size_like_5.tmp_0":
                #     print("parallel_tuner arg_name", tensor)
                if dims_mapping:
                    tensor_shape = tensor.shape
                else:
                    continue
                for i, dim_mapping in enumerate(dims_mapping):
                    if dim_mapping != -1 \
                        and (tensor_shape[i] % process_shape[dim_mapping] != 0 \
                        or dynamic_dims[i] == 1):
                        dims_mapping[i] = -1
                    # it is a fix-bug
                    if dim_mapping != -1 \
                        and process_shape[dim_mapping] == 1:
                        dims_mapping[i] = -1
            dist_op_impls = find_compatible_distributed_operator_impls(
                dist_op, partial=False)
            if dist_op_impls is not None:
                dist_op.dist_attr.impl_type = dist_op_impls[0].type
                dist_op.dist_attr.impl_idx = dist_op_impls[0].idx
            else:
                # Use the default dist op impl
                for arg_name in dist_attr.inputs_dist_attrs.keys():
                    dims_mapping = dist_attr.get_input_dims_mapping(arg_name)
                    for i, _ in enumerate(dims_mapping):
                        dims_mapping[i] = -1
                for arg_name in dist_attr.outputs_dist_attrs.keys():
                    dims_mapping = dist_attr.get_output_dims_mapping(arg_name)
                    for i, _ in enumerate(dims_mapping):
                        dims_mapping[i] = -1
                dist_op.dist_attr.impl_type = "default"
                dist_op.dist_attr.impl_idx = 0

    def _complete_tensor_dist_attr_by_op(self):
        # Complete tensor dist attr
        has_set_dist_attr = set()
        while_op = None
        for block in self._dist_context.serial_main_program.blocks:
            for op in block.ops:
                if op.type == "while":
                    while_op = op
                    continue
                if op.type == "read_from_array":
                    continue
                if op.type == "write_to_array":
                    continue
                for var_name in op.input_arg_names:
                    var = block._var_recursive(var_name)
                    if var.is_data or var.is_parameter:
                        if var_name not in has_set_dist_attr:
                            tensor_dist_attr = TensorDistributedAttribute()
                            tensor_dist_attr.process_mesh = self._dist_context.get_dist_op_for_program(
                                op).dist_attr.process_mesh
                            tensor_dist_attr.dims_mapping = self._dist_context.get_dist_op_for_program(
                                op).dist_attr.get_input_dims_mapping(
                                    var_name) if var.is_parameter else [
                                        -1 for i in var.shape
                                    ]
                            self._dist_context.set_tensor_dist_attr_for_program(
                                var, tensor_dist_attr)
                            has_set_dist_attr.add(var_name)
                for var_name in op.output_arg_names:
                    var = block._var_recursive(var_name)
                    if var_name not in has_set_dist_attr:
                        tensor_dist_attr = TensorDistributedAttribute()
                        tensor_dist_attr.process_mesh = self._dist_context.get_dist_op_for_program(
                            op).dist_attr.process_mesh
                        tensor_dist_attr.dims_mapping = self._dist_context.get_dist_op_for_program(
                            op).dist_attr.get_output_dims_mapping(var_name)
                        self._dist_context.set_tensor_dist_attr_for_program(
                            var, tensor_dist_attr)
                        has_set_dist_attr.add(var_name)

        # Find while op process mesh by ops in its sub block
        if while_op is None:
            return
        union_process_mesh = []
        block = self._dist_context.serial_main_program.blocks[while_op.attr(
            "sub_block").id]
        for op in block.ops:
            if op.type == "read_from_array" or op.type == "write_to_array":
                continue
            process_mesh = self._dist_context.get_dist_op_for_program(
                op).dist_attr.process_mesh

            # NOTE: The while op process mesh should contain output tensor process mesh
            for process in process_mesh.processes:
                if process not in union_process_mesh:
                    union_process_mesh.append(process)

            for output_name in op.output_arg_names:
                output = block._var_recursive(output_name)
                output_process_mesh = self._dist_context.get_dist_tensor_for_program(
                    output).dist_attr.process_mesh
                if output_process_mesh:
                    for process in output_process_mesh.processes:
                        if process not in union_process_mesh:
                            union_process_mesh.append(process)
        union_process_mesh.sort()
        self._dist_context.get_dist_op_for_program(
            while_op).dist_attr.process_mesh = union_process_mesh

        # Set array and fill_constant(it is relative with condition) op process mesh and tensor dist attr
        for block in self._dist_context.serial_main_program.blocks:
            for op in block.ops:
                # This is a trick to set fill_constant dist attr
                if op.type == "read_from_array" or op.type == "write_to_array" or op.type == "fill_constant":
                    self._dist_context.get_dist_op_for_program(
                        op).dist_attr.process_mesh = union_process_mesh
                    for var_name in op.output_arg_names:
                        var = block._var_recursive(var_name)
                        tensor_dist_attr = TensorDistributedAttribute()
                        # dims_mapping all -1
                        tensor_dist_attr.process_mesh = self._dist_context.get_dist_op_for_program(
                            op).dist_attr.process_mesh
                        self._dist_context.set_tensor_dist_attr_for_program(
                            var, tensor_dist_attr)
                        has_set_dist_attr.add(var_name)

                    for var_name in op.input_arg_names:
                        var = block._var_recursive(var_name)
                        if var.is_data or var.is_parameter:
                            if var_name not in has_set_dist_attr:
                                tensor_dist_attr = TensorDistributedAttribute()
                                tensor_dist_attr.process_mesh = self._dist_context.get_dist_op_for_program(
                                    op).dist_attr.process_mesh
                                self._dist_context.set_tensor_dist_attr_for_program(
                                    var, tensor_dist_attr)
                                has_set_dist_attr.add(var_name)

        # Set condition var and op process mesh
        condition_var = while_op.block._var_recursive(
            while_op.input("Condition")[0])
        for op in while_op.block.ops:
            if condition_var.name in op.output_arg_names:
                self._dist_context.get_dist_op_for_program(
                    op).dist_attr.process_mesh = union_process_mesh
                self._dist_context.get_dist_tensor_for_program(
                    condition_var).dist_attr.process_mesh = union_process_mesh
                break

        # NOTE: Some special tensors are not as input or output of op, they need be tricked to set
        var = self._dist_context.serial_main_program.global_block(
        ).vars["pos_embedding_2d_extra"]
        self._dist_context.get_dist_tensor_for_program(
            var).dist_attr.process_mesh = self._dist_context.process_meshes[0]
        var = self._dist_context.serial_main_program.global_block(
        ).vars["array_0.out"]
        ops = self._dist_context.serial_main_program.global_block().ops
        for idx, op in enumerate(ops):
            if "array_0.out" in op.input_arg_names:
                assign_start_idx = idx
                break
        self._dist_context.get_dist_tensor_for_program(
            var
        ).dist_attr.process_mesh = self._dist_context.get_dist_op_for_program(
            ops[assign_start_idx]).dist_attr.process_mesh
        var = self._dist_context.serial_main_program.global_block(
        ).vars["array_1.out"]
        self._dist_context.get_dist_tensor_for_program(
            var
        ).dist_attr.process_mesh = self._dist_context.get_dist_op_for_program(
            ops[assign_start_idx + 1]).dist_attr.process_mesh
        var = self._dist_context.serial_main_program.global_block(
        ).vars["array_2.out"]
        self._dist_context.get_dist_tensor_for_program(
            var
        ).dist_attr.process_mesh = self._dist_context.get_dist_op_for_program(
            ops[assign_start_idx + 2]).dist_attr.process_mesh
        var = self._dist_context.serial_main_program.global_block(
        ).vars["array_3.out"]
        self._dist_context.get_dist_tensor_for_program(
            var
        ).dist_attr.process_mesh = self._dist_context.get_dist_op_for_program(
            ops[assign_start_idx + 3]).dist_attr.process_mesh
        var = self._dist_context.serial_main_program.global_block(
        ).vars["array_4.out"]
        self._dist_context.get_dist_tensor_for_program(
            var
        ).dist_attr.process_mesh = self._dist_context.get_dist_op_for_program(
            ops[assign_start_idx + 4]).dist_attr.process_mesh

        # NOTE: This is a trick to avoid assign dist attr problem if it is a inplace operation
        for dist_op in self._dist_context._dist_ops_for_program.values():
            if dist_op.serial_op.type == "assign":
                output_tensor = dist_op.serial_op.block._var_recursive(
                    dist_op.serial_op.output("Out")[0])
                tensor_dims_mapping = self._dist_context.get_dist_tensor_for_program(
                    output_tensor).dist_attr.dims_mapping
                output_dims_mapping = dist_op.dist_attr.get_output_dims_mapping(
                    output_tensor.name)
                input_tensor = dist_op.serial_op.block._var_recursive(
                    dist_op.serial_op.input("X")[0])
                dist_op.dist_attr.set_output_dims_mapping(
                    output_tensor.name, tensor_dims_mapping)
                dist_op.dist_attr.set_input_dims_mapping(
                    input_tensor.name, tensor_dims_mapping)

    def _eval_trial(self, trial):
        if self._num_trials == 0:
            num_prev_trials = 0
        else:
            num_prev_trials = self._num_trials - 1

        results = None

        start_time = time.time()

        inter_node_partition = trial.space.values["inter_node_partitions"]
        intra_node_partition = trial.space.values["intra_node_partitions"]
        process_mesh_list = self._generate_process_mesh_list(
            inter_node_partition, intra_node_partition)
        op_id_to_process_mesh = self._apply_pipeline_partition_random(
            process_mesh_list)
        if op_id_to_process_mesh is None:
            print("Operators are less than pipeline stages", flush=True)
            return results

        op_id_to_dist_attr = {}
        for name, value in trial.space.values.items():
            if name != "inter_node_partitions" \
                and name !="intra_node_partitions":
                op_id_to_dist_attr[int(name)] = value

        end_time = time.time()
        cur_sample_time = end_time - start_time
        self._sample_time = (num_prev_trials * self._sample_time +
                             cur_sample_time) / self._num_trials
        print("\tsample_time",
              num_prev_trials,
              self._num_trials,
              self._sample_time,
              cur_sample_time,
              flush=True)

        assert len(op_id_to_process_mesh) == len(op_id_to_dist_attr)

        start_time = time.time()
        for op_id, process_mesh in op_id_to_process_mesh.items():
            dist_op = self._dist_context._dist_ops_for_program[op_id]
            dist_op.dist_attr = copy.deepcopy(op_id_to_dist_attr[op_id])
            assert dist_op.dist_attr.impl_type == op_id_to_dist_attr[
                op_id].impl_type
            assert dist_op.dist_attr.impl_idx == op_id_to_dist_attr[
                op_id].impl_idx
            dist_op.dist_attr.process_mesh = process_mesh
        self._amend_dist_attr()

        self._complete_tensor_dist_attr_by_op()

        self._dist_context.block_state.parse_forward_blocks(
            self._dist_context.serial_main_program)

        # For now, we only materialize the programs for all ranks for testing.
        # if self._materialized_for_all_ranks:
        #     self._parallelizer.parallel_all()
        end_time = time.time()
        cur_complete_time = end_time - start_time
        self._complete_time = (num_prev_trials * self._complete_time +
                               cur_complete_time) / self._num_trials
        print("\tcomplete_time",
              num_prev_trials,
              self._num_trials,
              self._complete_time,
              cur_complete_time,
              flush=True)

        start_time = time.time()
        estimate_time = self._estimate_trial()
        # print("time of cost model 0", estimate_time, flush=True)
        end_time = time.time()
        cur_estimate_time = end_time - start_time
        self._estimate_time = (num_prev_trials * self._estimate_time +
                               cur_estimate_time) / self._num_trials
        print("\testimate_time",
              num_prev_trials,
              self._num_trials,
              self._estimate_time,
              cur_estimate_time,
              flush=True)

        results = {"estimate_time": estimate_time}
        return results

    def _update_trail(self, trial, metrics, step=0):
        for metric_name, metric_value in metrics.items():
            trial.recorder.update(metric_name, metric_value, step=step)
        return trial.status

    def ernie_pass(self, dist_context):

        def is_union_process_mesh(process_mesh, dist_context):
            sub_set_count = 0
            for item in dist_context.process_meshes:
                for process in item.processes:
                    if process in process_mesh.processes:
                        sub_set_count += 1
                        break
            if sub_set_count > 1:
                return True
            return False

        # This is a trick to avoid output process mesh different from tensor process mesh (non-union process mesh) (for fill_constant_batch_size_like)
        for block in self._dist_context.serial_main_program.blocks:
            for serial_op in block.ops:
                if serial_op.type == "fill_constant_batch_size_like":
                    output_name = serial_op.output("Out")[0]
                    var = serial_op.block._var_recursive(output_name)
                    dist_tensor = self._dist_context.get_dist_tensor_for_program(
                        var)
                    for item in self._dist_context.serial_main_program.blocks:
                        for op in item.ops:
                            dist_op = self._dist_context.get_dist_op_for_program(
                                op)
                            for var_name in op.input_arg_names:
                                if var_name == output_name:
                                    # set op process mesh and dims_mapping the same as tensor
                                    if not is_union_process_mesh(
                                            dist_tensor.dist_attr.process_mesh,
                                            self._dist_context
                                    ) and not is_union_process_mesh(
                                            dist_op.dist_attr.process_mesh,
                                            self._dist_context):
                                        if dist_tensor.dist_attr.process_mesh != dist_op.dist_attr.process_mesh:
                                            dist_op.dist_attr.process_mesh = dist_tensor.dist_attr.process_mesh
                                        if dist_tensor.dist_attr.dims_mapping != dist_op.dist_attr.get_input_dims_mapping(
                                                var_name):
                                            if op.type == "while":
                                                continue
                                            dist_op.dist_attr.get_input_dist_attr(
                                                var_name
                                            ).dims_mapping = dist_tensor.dist_attr.dims_mapping
                                            if op.type == "concat":
                                                for _ in op.input_arg_names:
                                                    if _ != var_name:
                                                        dist_op.dist_attr.get_input_dist_attr(
                                                            _
                                                        ).dims_mapping = dist_tensor.dist_attr.dims_mapping
                                            # Gather in Erine, output dims_mapping should be the same with fill dims_mapping
                                            # Set output_dims_mapping
                                            output_var_name = op.output_arg_names[
                                                0]
                                            output_var = op.block._var_recursive(
                                                output_var_name)
                                            output_dist_tensor = self._dist_context.get_dist_tensor_for_program(
                                                output_var)
                                            output_dist_tensor.dist_attr.dims_mapping = dist_tensor.dist_attr.dims_mapping
                                            output_dist_tensor.dist_attr.process_mesh = dist_tensor.dist_attr.process_mesh
                                            # Set output tensor dims_mapping
                                            dist_op.dist_attr.get_output_dist_attr(
                                                output_var_name
                                            ).dims_mapping = dist_tensor.dist_attr.dims_mapping

                            for var_name in op.output_arg_names:
                                if var_name == output_name:
                                    if not is_union_process_mesh(
                                            dist_tensor.dist_attr.process_mesh,
                                            self._dist_context
                                    ) and not is_union_process_mesh(
                                            dist_op.dist_attr.process_mesh,
                                            self._dist_context):
                                        if dist_tensor.dist_attr.process_mesh != dist_op.dist_attr.process_mesh:
                                            dist_op.dist_attr.process_mesh = dist_tensor.dist_attr.process_mesh
                                        if dist_tensor.dist_attr.dims_mapping != dist_op.dist_attr.get_output_dims_mapping(
                                                var_name):
                                            dist_op.dist_attr.get_output_dist_attr(
                                                var_name
                                            ).dims_mapping = dist_tensor.dist_attr.dims_mapping

        # This is a trick to avoid assign op dist attr problem
        for dist_op in self._dist_context._dist_ops_for_program.values():
            if dist_op.serial_op.type == "assign":
                output_tensor = dist_op.serial_op.block._var_recursive(
                    dist_op.serial_op.output("Out")[0])
                tensor_dims_mapping = self._dist_context.get_dist_tensor_for_program(
                    output_tensor).dist_attr.dims_mapping
                output_dims_mapping = dist_op.dist_attr.get_output_dims_mapping(
                    output_tensor.name)
                input_tensor = dist_op.serial_op.block._var_recursive(
                    dist_op.serial_op.input("X")[0])
                dist_op.dist_attr.set_output_dims_mapping(
                    output_tensor.name, tensor_dims_mapping)
                dist_op.dist_attr.set_input_dims_mapping(
                    input_tensor.name, tensor_dims_mapping)

        # This is a trick to avoid shared parameter completion problem
        params_set = set()
        for dist_op in self._dist_context._dist_ops_for_program.values():
            serial_op = dist_op.serial_op
            if serial_op.type == "while":
                continue
            else:
                for var_name in serial_op.input_arg_names:
                    var = dist_op.serial_op.block._var_recursive(var_name)
                    if var.is_parameter:
                        if var.name not in params_set:
                            params_set.add(var_name)
                            dist_tensor = self._dist_context.get_dist_tensor_for_program(
                                var)
                            input_dims_mapping = dist_op.dist_attr.get_input_dims_mapping(
                                var_name)
                            dist_tensor.dist_attr.dims_mapping = input_dims_mapping

        # step1. find fill_const_batch_size_like op output whose shape is dynamic
        # step2. find assign and concat op whose input is tensor step founded.
        # step3. find op order
        # step4. change step3 op and tensor process_mesh

        # step1 and step 2
        dynamic_fill_constant_tensors = []
        fixed_indexs = {}
        dynamic_indexs = {}
        concat_and_assign_op_indexs = {}
        concat_output_tensors = []
        ref_mapping = {}
        concat_input_x_to_fill_constant_tensor = {}
        unsqueeze2_index = None  # process_mesh and tensor process_mesh should be the same as while op
        for block_idx, block in enumerate(
                dist_context.serial_main_program.blocks):
            fixed_indexs[block_idx] = []
            dynamic_indexs[block_idx] = []
            concat_and_assign_op_indexs[block_idx] = []
            ref_mapping[block_idx] = {}
            for idx, serial_op in enumerate(block.ops):
                # the first unsqueeze2 should on all
                if serial_op.type == "unsqueeze2":
                    if block_idx > 0 and unsqueeze2_index is None:
                        unsqueeze2_index = idx
                if serial_op.type == "fill_constant_batch_size_like":
                    if serial_op.attr('shape').count(0) >= 1:
                        dynamic_fill_constant_tensors.append(
                            serial_op.output("Out")[0])
                if serial_op.type == "concat":
                    input_X = serial_op.input("X")
                    for var_name in input_X:
                        if var_name in dynamic_fill_constant_tensors:
                            concat_and_assign_op_indexs[block_idx].append(idx)
                            ref_mapping[block_idx][idx] = var_name
                            break
                        # hard code
                        if "gather" in var_name:
                            concat_and_assign_op_indexs[block_idx].append(idx)
                            ref_mapping[block_idx][idx] = var_name
                            break
                    # concat output
                    if idx in ref_mapping[block_idx]:
                        concat_output = serial_op.output("Out")[0]
                        concat_output_tensors.append(concat_output)
                        concat_input_x_to_fill_constant_tensor[
                            concat_output] = ref_mapping[block_idx][idx]
                if serial_op.type == "assign":
                    output = serial_op.output("Out")[0]
                    if output in dynamic_fill_constant_tensors:
                        concat_and_assign_op_indexs[block_idx].append(idx)
                        ref_mapping[block_idx][idx] = output
                if serial_op.type == "matmul_v2":
                    if serial_op.input("Y")[0] in concat_output_tensors:
                        if serial_op.attr("trans_y"):
                            dynamic_indexs[block_idx].append(idx)
                            ref_mapping[block_idx][
                                idx] = concat_input_x_to_fill_constant_tensor[
                                    serial_op.input("Y")[0]]
                        else:
                            fixed_indexs[block_idx].append(idx)

        # NOTE: dims_mapping is not considered, so it stil will have dynamic shape
        # change assign and concat first，then change op among them, last change the unsqueeze2
        # unsqueeze2的，op process_mesh 和 output的process_mesh,取它output，它output作为input的process_mesh的并集
        for block_idx in concat_and_assign_op_indexs:
            block = dist_context.serial_main_program.blocks[block_idx]
            ops = block.ops
            for op_idx in concat_and_assign_op_indexs[block_idx]:
                ref_tensor = ref_mapping[block_idx][op_idx]
                ref_process_mesh = dist_context.get_dist_tensor_for_program(
                    block._var_recursive(ref_tensor)).dist_attr.process_mesh
                op_dist_attr = dist_context.get_dist_op_for_program(
                    ops[op_idx]).dist_attr
                op_process_mesh = op_dist_attr.process_mesh
                if op_process_mesh != ref_process_mesh:
                    op_dist_attr.process_mesh = ref_process_mesh
                    if ops[op_idx].type == "concat":
                        output = ops[op_idx].output("Out")[0]
                        output_dist_attr = dist_context.get_dist_tensor_for_program(
                            block._var_recursive(output)).dist_attr
                        if output_dist_attr.process_mesh != ref_process_mesh:
                            output_dist_attr.process_mesh = ref_process_mesh

                if ops[op_idx].type == "assign":
                    op_dist_attr.process_mesh = ref_process_mesh
                    output = ops[op_idx].output("Out")[0]
                    output_dist_attr = dist_context.get_dist_tensor_for_program(
                        block._var_recursive(output)).dist_attr
                    if output_dist_attr.process_mesh != ref_process_mesh:
                        output_dist_attr.process_mesh = ref_process_mesh
                    output_tensor = block._var_recursive(
                        ops[op_idx].output("Out")[0])
                    op_dist_attr = self._dist_context.get_dist_op_for_program(
                        ops[op_idx]).dist_attr
                    tensor_dims_mapping = self._dist_context.get_dist_tensor_for_program(
                        output_tensor).dist_attr.dims_mapping
                    output_dims_mapping = op_dist_attr.get_output_dims_mapping(
                        output_tensor.name)
                    input_tensor = block._var_recursive(
                        ops[op_idx].input("X")[0])
                    op_dist_attr.set_output_dims_mapping(
                        output_tensor.name, tensor_dims_mapping)
                    op_dist_attr.set_input_dims_mapping(input_tensor.name,
                                                        tensor_dims_mapping)

            for i in range(len(dynamic_indexs[block_idx])):
                ref_tensor = ref_mapping[block_idx][dynamic_indexs[block_idx]
                                                    [i]]
                ref_process_mesh = dist_context.get_dist_tensor_for_program(
                    block._var_recursive(ref_tensor)).dist_attr.process_mesh
                for op_idx in range(dynamic_indexs[block_idx][i],
                                    fixed_indexs[block_idx][i]):
                    op_dist_attr = dist_context.get_dist_op_for_program(
                        ops[op_idx]).dist_attr
                    op_process_mesh = op_dist_attr.process_mesh
                    if op_process_mesh != ref_process_mesh:
                        op_dist_attr.process_mesh = ref_process_mesh
                        for output in ops[op_idx].output_arg_names:
                            output_dist_attr = dist_context.get_dist_tensor_for_program(
                                block._var_recursive(output)).dist_attr
                            if output_dist_attr.process_mesh != ref_process_mesh:
                                output_dist_attr.process_mesh = ref_process_mesh

        assert unsqueeze2_index is not None
        block = dist_context.serial_main_program.blocks[1]
        unsqueeze2_op = block.ops[unsqueeze2_index]
        output = unsqueeze2_op.output("Out")[0]
        unsqueeze2_op_union_process_mesh = []
        for op in block.ops:
            if op.type == "while":
                continue
            for input in op.input_arg_names:
                if input == output:
                    process_mesh = dist_context.get_dist_op_for_program(
                        op).dist_attr.process_mesh
                    for process in process_mesh.processes:
                        if process not in unsqueeze2_op_union_process_mesh:
                            unsqueeze2_op_union_process_mesh.append(process)
        unsqueeze2_op_union_process_mesh.sort()
        dist_context.get_dist_op_for_program(
            unsqueeze2_op
        ).dist_attr.process_mesh = unsqueeze2_op_union_process_mesh
        dist_context.get_dist_tensor_for_program(block._var_recursive(
            output)).dist_attr.process_mesh = unsqueeze2_op_union_process_mesh
        dist_context.get_dist_tensor_for_program(
            block._var_recursive(
                unsqueeze2_op.output("XShape")
                [0])).dist_attr.process_mesh = unsqueeze2_op_union_process_mesh

        # deal with scale
        scale_op_output = unsqueeze2_op.input("X")[0]
        scale_op = None
        for idx, op in enumerate(block.ops):
            if op.type == "while":
                continue
            if scale_op_output in op.output_arg_names:
                scale_op = op
                break
        assert scale_op is not None
        dist_context.get_dist_op_for_program(
            scale_op).dist_attr.process_mesh = unsqueeze2_op_union_process_mesh
        dist_context.get_dist_tensor_for_program(
            block._var_recursive(scale_op_output)
        ).dist_attr.process_mesh = unsqueeze2_op_union_process_mesh

        # deal with concat
        concat_op_output = scale_op.input("X")[0]
        concat_op = None
        for idx, op in enumerate(block.ops):
            if op.type == "while":
                continue
            if concat_op_output in op.output_arg_names:
                concat_op = op
                break
        assert concat_op is not None
        dist_context.get_dist_op_for_program(
            concat_op).dist_attr.process_mesh = unsqueeze2_op_union_process_mesh
        dist_context.get_dist_tensor_for_program(
            block._var_recursive(concat_op_output)
        ).dist_attr.process_mesh = unsqueeze2_op_union_process_mesh

        # deal with op whose input is concat_op_output
        for idx, op in enumerate(block.ops):
            if op.type == "while":
                continue
            if concat_op_output in op.input_arg_names:
                op_dist_attr = dist_context.get_dist_op_for_program(
                    op).dist_attr
                if op_dist_attr.process_mesh != unsqueeze2_op_union_process_mesh:
                    op_dist_attr.process_mesh = unsqueeze2_op_union_process_mesh
                    for output in op.output_arg_names:
                        dist_tensor = dist_context.get_dist_tensor_for_program(
                            block._var_recursive(output))
                        if dist_tensor.dist_attr.process_mesh != unsqueeze2_op_union_process_mesh:
                            dist_tensor.dist_attr.process_mesh = unsqueeze2_op_union_process_mesh

        # deal with gather
        gather_output = "gather_1.tmp_0"
        gather_op = None
        for idx, op in enumerate(block.ops):
            if op.type == "while":
                continue
            if gather_output in op.output_arg_names:
                gather_op = op
                break
        assert gather_op is not None
        dist_context.get_dist_op_for_program(
            gather_op).dist_attr.process_mesh = unsqueeze2_op_union_process_mesh
        dist_context.get_dist_tensor_for_program(
            block._var_recursive(gather_output)
        ).dist_attr.process_mesh = unsqueeze2_op_union_process_mesh

        # deal with read_from_array
        read_array_output = gather_op.input("X")[0]
        read_array_op = None
        for idx, op in enumerate(block.ops):
            if op.type == "while":
                continue
            if read_array_output in op.output_arg_names:
                read_array_op = op
                break
        assert read_array_op is not None
        dist_context.get_dist_op_for_program(
            read_array_op
        ).dist_attr.process_mesh = unsqueeze2_op_union_process_mesh
        dist_context.get_dist_tensor_for_program(
            block._var_recursive(read_array_output)
        ).dist_attr.process_mesh = unsqueeze2_op_union_process_mesh

        # deal with assign_value
        block = self._dist_context.serial_main_program.global_block()
        for op in block.ops:
            if op.type == "assign_value":
                for output in op.output_arg_names:
                    dist_context.get_dist_tensor_for_program(
                        block._var_recursive(output)).dist_attr.dims_mapping = [
                            -1 for i in range(
                                len(block._var_recursive(output).shape))
                        ]
                    dist_context.get_dist_op_for_program(
                        op
                    ).dist_attr.set_output_dims_mapping(output, [
                        -1
                        for i in range(len(block._var_recursive(output).shape))
                    ])
                    dist_context.get_dist_tensor_for_program(
                        block._var_recursive(output)
                    ).dist_attr.process_mesh = dist_context.get_dist_op_for_program(
                        op).dist_attr.process_mesh

    def _estimate_trial(self):
        assert self._cluster is not None
        if self._mode == "eval":
            self._estimator = CostEstimator(
                self._dist_context.serial_main_program,
                self._cluster,
                loop_count=self._loop_count)
        elif self._mode == "predict":
            self._estimator = CostEstimator(
                self._dist_context.serial_main_program,
                self._cluster,
                loop_count=self._loop_count)
        elif self._mode == "train":
            # get serial main program with backward
            serial_main_program = self._dist_context.serial_main_program
            serial_startup_program = self._dist_context.serial_startup_program
            serial_optimizer = self._dist_context.serial_optimizer

            # Generate backward
            serial_loss = self._dist_context.serial_fetch_vars["loss"][0]
            params_grads = self._parallelizer._generate_backward(
                serial_main_program, serial_startup_program, serial_loss)

            self._estimator = CostEstimator(serial_main_program,
                                            self._cluster,
                                            loop_count=self._loop_count)

        max_memory = self._estimator._estimate_max_memory(self._dist_context)
        print("\tmax_memory", "{:,}".format(max_memory), flush=True)
        # The max memory must be less than 80% of 32GB (hard code)
        if max_memory > 32 * 0.8 * 1024 * 1024 * 1024:
            return math.inf
        else:
            global_cost = self._estimator.estimate(self._dist_context)
            return global_cost.time

        if self._is_ernie:
            self.ernie_pass(self._dist_context)
        global_cost = self._estimator.estimate(self._dist_context)
        return global_cost.time

    def _store_init_parallel_strategy(self):
        # If there is no annotation information, use the dp as the initial parallel strategy.
        # TODO: we should need a better way to set up the initial parallel strategy.
        if not self._dist_context.has_annotation \
            or not self._dist_context.process_meshes:
            ranks = self._num_machines * self._num_devices_per_machine
            tensor_node = self._dist_context._serial_ordered_tensor_nodes[0]
            tensor_node_id = _node_id(tensor_node)
            tensor = self._dist_context._dist_tensors_for_graph[
                tensor_node_id].serial_tensor
            tensor_dist_attr = self._dist_context._dist_tensors_for_graph[
                tensor_node_id].dist_attr
            tensor_dist_attr.process_mesh = ProcessMesh(list(range(ranks)))
            tensor_dist_attr.dims_mapping = [0] + [
                -1 for _ in range(len(tensor.shape) - 1)
            ]
            tensor_dist_attr.mark_annotated("process_mesh")
            tensor_dist_attr.mark_annotated("dims_mapping")
            print("Use dp as the init parallel strategy!", flush=True)

        # Do the sharding propagation
        self._completer.complete_forward_annotation()
        self._dist_context.block_state.parse_forward_blocks(
            self._dist_context.serial_main_program)

        # Backup the intital parallel strategy
        self._init_parallel_strategy[0] = copy.deepcopy(
            self._dist_context._dist_tensors_for_program)
        self._init_parallel_strategy[1] = copy.deepcopy(
            self._dist_context._dist_ops_for_program)
        self._init_parallel_strategy[2] = copy.deepcopy(
            self._dist_context.process_meshes)

        # Initialize the best parallel strategy to the initial one
        self._best_parallel_strategy[0] = copy.deepcopy(
            self._dist_context._dist_tensors_for_program)
        self._best_parallel_strategy[1] = copy.deepcopy(
            self._dist_context._dist_ops_for_program)
        self._best_parallel_strategy[2] = copy.deepcopy(
            self._dist_context._process_meshes)

    def _store_best_parallel_strategy(self):
        # Swap the best and the current parallel strategy
        tmp = [None, None, None]
        tmp[0] = self._best_parallel_strategy[0]
        tmp[1] = self._best_parallel_strategy[1]
        tmp[2] = self._best_parallel_strategy[2]
        self._best_parallel_strategy[
            0] = self._dist_context._dist_tensors_for_program
        self._best_parallel_strategy[
            1] = self._dist_context._dist_ops_for_program
        self._best_parallel_strategy[2] = self._dist_context._process_meshes
        self._dist_context._dist_tensors_for_program = tmp[0]
        self._dist_context._dist_ops_for_program = tmp[1]
        self._dist_context._process_meshes = tmp[2]

    def tune(self):
        global_start_time = time.time()
        self._dist_context._backup(serial=True, dist=True)
        # This store statement must follow the above backup statement
        self._store_init_parallel_strategy()
        init_time = self._estimate_trial()  # estimate_trial when init
        # print_program_with_dist_attr(self._dist_context.serial_main_program, self._dist_context)
        # We have to restore the distributed context, because the estimation of one trail need to
        # generate the backward and update parts. Since we will do the tuning process,
        # here we only need to reset all distributed information to the default one.
        self._dist_context._restore(serial=True,
                                    serial_mode="to_backup",
                                    dist=True,
                                    dist_mode="to_default")

        best_time = init_time
        start_time = time.time()
        self.construct_space()
        end_time = time.time()
        print("construct_space time",
              self._num_trials,
              end_time - start_time,
              flush=True)
        create_trial_time = 0.0
        eval_trial_time = 0.0
        self._sample_time = 0.0
        self._complete_time = 0.0
        self._estimate_time = 0.0
        found_better = False
        while True and not found_better:
            start_time = time.time()
            trial = self._create_trial()
            if self._num_trials == 0:
                num_prev_trials = 0
            else:
                num_prev_trials = self._num_trials - 1
            end_time = time.time()
            cur_create_trial_time = end_time - start_time
            create_trial_time = (num_prev_trials * create_trial_time +
                                 cur_create_trial_time) / self._num_trials
            print("create_trial time",
                  num_prev_trials,
                  self._num_trials,
                  create_trial_time,
                  cur_create_trial_time,
                  flush=True)
            if trial.status == TrialStatus.STOPPED:
                break
            # We need to backup the distributed context, because the evaluation of one trail will
            # generate the backward and update parts which may change the context.
            # However, the distributed information of the context aren't backup since a new one is used.
            self._dist_context._backup(serial=True, dist=False)

            start_time = time.time()
            results = self._eval_trial(trial)
            end_time = time.time()
            cur_eval_trial_time = end_time - start_time
            eval_trial_time = (num_prev_trials * eval_trial_time +
                               cur_eval_trial_time) / self._num_trials
            print("eval_trial time",
                  num_prev_trials,
                  self._num_trials,
                  eval_trial_time,
                  cur_eval_trial_time,
                  "\n",
                  flush=True)

            cur_time = results["estimate_time"]
            if cur_time < best_time:
                self._update_trail(trial, results)
                self._store_best_parallel_strategy()
                best_time = cur_time
                found_better = True
            # We need to restore the distributed context and reset the distributed information to the default.
            self._dist_context._restore(serial=True,
                                        serial_mode="to_backup",
                                        dist=True,
                                        dist_mode="to_default")
        # Select the best parallel strategy
        self._dist_context._dist_tensors_for_program = self._best_parallel_strategy[
            0]
        self._dist_context._dist_ops_for_program = self._best_parallel_strategy[
            1]
        self._dist_context._process_meshes = self._best_parallel_strategy[2]

        if self._is_ernie:
            self.ernie_pass(self._dist_context)

        # Calc cost again after ernie_pass
        self._estimator = CostEstimator(self._dist_context.serial_main_program,
                                        self._cluster,
                                        loop_count=self._loop_count)
        global_cost = self._estimator.estimate(self._dist_context)
        print("exec_cost is {}s ".format(global_cost.time / 1000))

        global_end_time = time.time()
        print("search {} times cost {}'s and best time is {}s".format(
            self._max_trials, global_end_time - global_start_time,
            best_time / 1000))
