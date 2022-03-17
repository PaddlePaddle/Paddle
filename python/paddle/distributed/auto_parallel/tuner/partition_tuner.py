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

import copy
import hashlib
import itertools
from collections import defaultdict
from numpy import random
from ..process_mesh import ProcessMesh
from ..dist_op import DistributedOperator
from ..dist_attribute import OperatorDistributedAttribute
from ..operators.common import find_best_compatible_distributed_operator_impl
from .trial import Trial, TrialStatus
from .tuner import Tuner
from .tunable_space import TunableSpace


class PartitionTuner(Tuner):
    def __init__(self,
                 dist_context,
                 cluster=None,
                 max_trials=None,
                 tuner_id=None,
                 seed=None,
                 logger=None):
        self._dist_context = dist_context
        self._cluster = cluster
        self._space = TunableSpace()
        self._objective = "cost"
        self._direction = "min"
        self._max_trials = max_trials
        self._tuner_id = tuner_id
        self._seed = seed or random.randint(1, 10000)
        self._seed_state = self._seed
        self._logger = logger
        self._trials = {}
        self._max_collisions = 3
        self._tried_values = set()

        self._op_id_to_dist_attr_candidates = defaultdict(list)
        self._cached_dims_mapping_candidates = {}
        self._cached_dist_attr_candidates = {}

    def _get_machines_info(self):
        # This function is for testing and will be replaced
        machines_info = {}
        if self._cluster is None:
            num_nodes = 4
            devices_per_node = 8
            machines_info = [devices_per_node for i in range(num_nodes)]
        else:
            return self._cluster.get_machines_info()
        return machines_info

    def generate_process_mesh_candidates(self, machines_info):
        process_mesh_candidates = []
        finest_process_meshes = []
        start_rank = 0
        for _, num_devices in machines_info.items():
            end_rank = start_rank + num_devices
            cur_process_mesh = [i for i in range(start_rank, end_rank)]
            finest_process_meshes.append(cur_process_mesh)
            start_rank += num_devices
        finest_process_meshes.sort(key=lambda pm: len(pm))

        max_interval = 0
        cur_interval = 0
        cur_len = 0
        start_indices = []
        for i, cur_process_mesh in enumerate(finest_process_meshes):
            if i == 0:
                start_indices.append(i)
                cur_len = len(cur_process_mesh)
                max_interval = 1
                cur_interval = 1
            else:
                if len(cur_process_mesh) != cur_len:
                    start_indices.append(i)
                    cur_len = len(cur_process_mesh)
                    if max_interval < cur_interval:
                        max_interval = cur_interval
                        cur_interval = 0
                cur_interval += 1
        start_indices.append(len(finest_process_meshes))

        process_mesh_candidates.append(finest_process_meshes)
        if max_interval < 2:
            return process_mesh_candidates

        for interval in range(2, max_interval):
            new_process_mesh_config = []
            for i in range(len(start_indices) - 1):
                start = start_indices[i]
                stop = start_indices[i + 1]
                merged_process_mesh = finest_process_meshes[start]
                for j in range(start, stop):
                    if (j - start) % interval == 0 and j != start:
                        new_process_mesh_config.append(merged_process_mesh)
                        merged_process_mesh = finest_process_meshes[j]
                    else:
                        merged_process_mesh.extend(finest_process_meshes[j])
                new_process_mesh_config.append(merged_process_mesh)
            process_mesh_candidates.append(new_process_mesh_config)
        return process_mesh_candidates

    def _generate_dims_mapping_candidates_helper(self, dims_mapping, dims_list,
                                                 start, visited, candidates):
        if start == len(dims_mapping):
            candidates.add(copy.deepcopy(dims_mapping))
            return
        # Add -1 since it is always a valid mapping
        dims_list.append(-1)
        visited.append(False)
        for idx, dim in enumerate(dims_list):
            if visited[idx] == False:
                dims_mapping[start] = dim
                visited[idx] = True
                self._generate_dims_mapping_candidates_helper(
                    dims_mapping, dims_list, start + 1, visited, candidates)
                visited[idx] = False
        # Pop -1 after backtracing
        dims_list.pop()
        visited.pop()

    def _generate_dims_mapping_candidates(self, dims_mapping_len,
                                          process_mesh_len):
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

    def generate_dist_attr_candidates(self, op_id, dist_op):
        serial_op = dist_op.serial_op
        op_dist_attr = dist_op.dist_attr

        key = []
        key.append(serial_op.type)
        for input_name in serial_op.input_names:
            key.append(input_name)
            for input_arg_name in serial_op.input(input_name):
                key.append(op_dist_attr.get_input_dims_mapping(input_arg_name))
        for output_name in serial_op.output_names:
            key.append(output_name)
            for output_arg_name in serial_op.output(output_name):
                key.append(
                    op_dist_attr.get_output_dims_mapping(output_arg_name))
        if key in self._cached_dist_attr_candidates:
            return self._cached_dist_attr_candidates[key]

        inputs_dist_attrs = op_dist_attr.inputs_dist_attrs
        outputs_dist_attrs = op_dist_attr.outputs_dist_attrs
        for tensor_name, tensor_dist_attr in inputs_dist_attrs.items():
            key.append()
        is_inputs = []
        tensor_names = []
        dist_attr_candidates = []
        for tensor_name, tensor_dist_attr in inputs_dist_attrs.items():
            original_dims_mapping = tensor_dist_attr.dims_mapping
            dims_mapping_len = len(original_dims_mapping)
            is_inputs.append(True)
            tensor_names.append(tensor_name)
            dist_attr_candidates.append(
                self._generate_dims_mapping_candidates(dims_mapping_len, 2))
        for tensor_name, tensor_dist_attr in outputs_dist_attrs.items():
            original_dims_mapping = tensor_dist_attr.dims_mapping
            dims_mapping_len = len(original_dims_mapping)
            is_inputs.append(False)
            tensor_names.append(tensor_name)
            dist_attr_candidates.append(
                self._generate_dims_mapping_candidates(dims_mapping_len, 2))
        for dims_mapping_candidate in itertools.product(*dist_attr_candidates):
            new_op_dist_attr = OperatorDistributedAttribute()
            for i, dims_mapping in enumerate(dims_mapping_candidate):
                if is_inputs[i]:
                    new_op_dist_attr.set_input_dims_mapping(tensor_names[i],
                                                            dims_mapping)
                else:
                    new_op_dist_attr.set_output_dims_mapping(tensor_names[i],
                                                             dims_mapping)
            new_dist_op = DistributedOperator(dist_op.serial_op,
                                              new_op_dist_attr)
            if find_best_compatible_distributed_operator_impl(
                    new_dist_op, partial=False):
                self._op_id_to_dist_attr_candidates[op_id].append(
                    new_op_dist_attr)
        self._cached_dist_attr_candidates[
            key] = self._op_id_to_dist_attr_candidates[op_id]
        return self._op_id_to_dist_attr_candidates[op_id]

    def _compute_values_hash(self, values):
        keys = sorted(values.keys())
        s = "".join(str(k) + "=" + str(values[k]) for k in keys)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]

    def _random_values(self, trial_id):
        trial = self._trials[trial_id]
        space = trial.space
        collisions = 0
        while True:
            for v in space.variables:
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

    def construct_space(self):
        process_mesh_candidates = self.generate_process_mesh_candidates(None)
        self._space.choice(
            "process_mesh",
            process_mesh_candidates,
            default=process_mesh_candidates[0])

        dist_ops = self._dist_context._dist_ops_for_program()
        for op_id, dist_op in dist_ops.items():
            op_dist_attr_candidates = self.generate_dist_attr_candidates(
                op_id, dist_op)
            self._space.choice(
                op_id,
                op_dist_attr_candidates,
                default=op_dist_attr_candidates[0])

    def populate_space(self, trial_id):
        values = self._random_values(trial_id)
        if values is None:
            return {"status": TrialStatus.STOPPED, "values": None}
        return {"status": TrialStatus.RUNNING, "values": values}

    def tune(self):
        while True:
            trial = self.create_trial()
            if trial.status == TrialStatus.STOPPED:
                break
            results = self.eval_trial(trial.id)
            self.update_trial(trial.id, results)

    def create_trial(self):
        trial_id = "{{:0{}d}}".format(len(str(self._max_trials)))
        trial_id = trial_id.format(len(self._trials))

        if self._max_trials and len(self._trials) >= self._max_trials:
            status = TrialStatus.STOPPED
            values = None
        else:
            status = TrialStatus.RUNNING
            values = self.populate_space(trial_id)

        space = TunableSpace()
        space.variables = self._space.variables
        space.values = values
        trial = Trial(tunable_space=space, trial_id=trial_id, status=status)
        self._trials[trial.id] = trial
        return trial

    def apply_pipeline_partition(self, process_mesh_list):
        op_id_to_process_mesh = {}
        total_processes = 0
        for process_mesh in process_mesh_list:
            total_processes += len(process_mesh)
        total_ops = len(self._dist_context._dist_ops_for_program)
        ops_per_process = total_ops / total_processes
        pipeline_starts = []
        start = 0
        pipeline_starts.append(0)
        for process_mesh in process_mesh_list:
            processes = len(process_mesh)
            start += int(processes * ops_per_process)
            pipeline_starts.append(start)
        pipeline_starts[-1] = total_ops - 1
        start = 1
        for idx, op_id in enumerate(self._dist_context._dist_ops_for_program()
                                    .keys()):
            if idx < pipeline_starts[start]:
                op_id_to_process_mesh[op_id] = process_mesh_list[start - 1]
            else:
                start += 1
                op_id_to_process_mesh[op_id] = process_mesh_list[start - 1]
        return op_id_to_process_mesh

    def eval_trial(self, trial_id):
        results = None
        trial = self._trials[trial_id]
        process_mesh_list = trial.values["process_mesh"]
        op_id_to_process_mesh = self.apply_pipeline_partition(process_mesh_list)
        op_id_to_dist_attr = {}
        for name, value in trial.values.items():
            if name != "process_mesh":
                op_id_to_dist_attr[name] = value
        assert len(op_id_to_process_mesh) == len(op_id_to_dist_attr)
        for op_id, process_mesh in op_id_to_process_mesh.items():
            dist_op = self._dist_context._dist_ops_for_program[op_id]
            dist_op.dist_attr = op_id_to_dist_attr[op_id]
            dist_op.dist_attr.process_mesh = process_mesh
        # TODO: eval the partiton strategy by calling the cost model
        return results

    def update_trial(self, trial_id, metrics, step=0):
        trial = self._trials[trial_id]
        for metric_name, metric_value in metrics.items():
            trial.metrics.update(metric_name, metric_value, step=step)
        return trial.status
