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
import time
import random
import logging
from functools import reduce
from itertools import chain, product
from collections import OrderedDict

import numpy as np

import paddle
import paddle.distributed.auto_parallel as auto
from .cost_model import estimate_cost
from .dist_op import DistributedOperator
from .process_group import _g_process_group_map
from .process_group import ProcessGroup, get_process_group
from .completion import is_elementwise_like_op
from .operators.common import get_distributed_operator_impl_container
from .utils import update_op_dims_mapping_by_default_dist_impl
from .utils import update_op_dims_mapping_by_elementwise_like_dist_impl
from .utils import get_all_distributed_main_program
from .dist_context import DistributedContext, DistributedOperatorContext
from .dist_attribute import OperatorDistributedAttribute, TensorDistributedAttribute

paddle.enable_static()
paddle.seed(123)
random.seed(123)
np.random.seed(123)


class PlanFilter:
    @staticmethod
    def check_dims_mapping_for_tensor(process_mesh_topology, tensor_shape,
                                      dims_mapping):
        valid = True
        assert len(tensor_shape) == len(dims_mapping)

        for idx, dim_mapping in enumerate(dims_mapping):
            if dim_mapping != -1:
                if tensor_shape[idx] % process_mesh_topology[
                        dim_mapping] != 0 or dims_mapping.count(
                            dim_mapping) > 1:
                    valid = False
            if dim_mapping != -1 and process_mesh_topology[0] == 1:
                valid = False

        return valid

    @staticmethod
    def check_dims_mapping_for_op(op, op_dist_attr, vars):
        process_mesh = op_dist_attr.process_mesh
        assert process_mesh is not None, "The process mesh should not be None."
        for var_name in op.input_arg_names:
            dims_mapping = op_dist_attr.get_input_dims_mapping(var_name)
            if not PlanFilter.check_dims_mapping_for_tensor(
                    process_mesh.topology, vars[var_name].shape, dims_mapping):
                return False
            if vars[var_name].is_data and len(dims_mapping) > 1:
                for dim in dims_mapping[1:]:
                    if dim != -1:
                        return False

        for var_name in op.output_arg_names:
            dims_mapping = op_dist_attr.get_output_dims_mapping(var_name)
            if not PlanFilter.check_dims_mapping_for_tensor(
                    process_mesh.topology, vars[var_name].shape, dims_mapping):
                return False

        return True

    @staticmethod
    def check_dims_mapping_for_special_op(op, op_dist_attr, vars):
        if op.type == "layer_norm":
            bias_dims_mapping = op_dist_attr.get_input_dims_mapping(
                op.input("Bias")[0])
            scale_dims_mapping = op_dist_attr.get_input_dims_mapping(
                op.input("Scale")[0])
            x_dims_mapping = op_dist_attr.get_input_dims_mapping(
                op.input("X")[0])
            mean_dims_mapping = op_dist_attr.get_output_dims_mapping(
                op.output("Mean")[0])
            variance_dims_mapping = op_dist_attr.get_output_dims_mapping(
                op.output("Variance")[0])
            y_dims_mapping = op_dist_attr.get_output_dims_mapping(
                op.output("Y")[0])
            if x_dims_mapping != y_dims_mapping:
                return False

            if scale_dims_mapping[0] != x_dims_mapping[-1]:
                return False

            if bias_dims_mapping[0] != y_dims_mapping[-1]:
                return False

            if mean_dims_mapping[0] != x_dims_mapping[0]:
                return False

            if variance_dims_mapping[0] != x_dims_mapping[0]:
                return False

        return True


class PlanSpace:
    not_enum_ops = ["create_py_reader", "create_double_buffer_reader", "read"]
    special_vars = [
        "lod_tensor_blocking_queue_0", "create_py_reader_0", "double_buffer_0"
    ]

    @staticmethod
    def _enum_dims_mapping(process_mesh_topology, visited, path, depth, res,
                           tensor_shape):
        """Enumerate dims mapping of tensor by the given process_mesh_topology"""
        nums = list(range(-1, len(process_mesh_topology)))
        if depth == len(tensor_shape):
            valid = True
            for idx, item in enumerate(path):
                if item != -1:
                    if tensor_shape[idx] % process_mesh_topology[
                            item] != 0 or path.count(item) > 1:
                        valid = False
            if valid:
                res.append(copy.deepcopy(path))
            return

        for i in range(len(nums)):
            if not visited[i]:
                if i != 0:
                    visited[i] = True
                path.append(nums[i])
                PlanSpace._enum_dims_mapping(process_mesh_topology, visited,
                                             path, depth + 1, res, tensor_shape)
                visited[i] = False
                path.pop()

    @staticmethod
    def enum_process_mesh_topology(processes):
        """Enumerate all process meshes with the given processes."""
        assert processes >= 1, "The processes must be number and greater than 0."
        # compute divisors
        divisors = []
        for i in range(1, processes + 1):
            if processes % i == 0:
                divisors.append(i)

        # compute valid process mesh
        results = []
        for i in range(len(divisors) - 1, 0, -1):
            result = []
            result.append(divisors[i])
            if i == len(divisors) - 1:
                results.append(copy.deepcopy(result))
                continue

            j = 1
            while j < len(divisors):
                if len(result) == 1:
                    result.append(divisors[j])
                elif len(result) == 2:
                    if processes % (result[0] * result[1]) == 0:
                        if processes // (result[0] * result[1]) == 1:
                            results.append(copy.deepcopy(result))
                            break
                        else:
                            result.append(processes // (result[0] * result[1]))
                            results.append(copy.deepcopy(result))
                            result.pop(-1)
                            result.pop(-1)
                            j += 1
                    else:
                        if result[0] * result[1] < processes:
                            result.pop(-1)
                            j += 1
                        else:
                            break
        return results

    @staticmethod
    def _enum_valid_dist_attr_for_op(program, op, process_mesh):
        """Enumerate the valid distributed attribute for op based on the given process mesh."""
        vars = program.global_block().vars
        dims_mapping_dict = OrderedDict()
        op_valid_dist_attrs = []
        dist_op_impl_container = get_distributed_operator_impl_container(
            op.type)

        # enumerate all valid dims mapping of tensor when process mesh given
        for var_name in chain(op.input_arg_names, op.output_arg_names):
            visited = [
                False
                for _ in range(
                    len(list(range(-1, len(process_mesh.topology)))))
            ]
            depth = 0
            path = []
            dims_mapping_list = []
            PlanSpace._enum_dims_mapping(process_mesh.topology, visited, path,
                                         depth, dims_mapping_list,
                                         vars[var_name].shape)
            dims_mapping_dict[var_name] = copy.deepcopy(dims_mapping_list)

        # compose dims mapping
        composed_dims_mapping_list = list(
            product(
                *[dims_mapping_dict[key] for key in dims_mapping_dict.keys()]))
        for composed_dims_mapping in composed_dims_mapping_list:
            op_dist_attr = OperatorDistributedAttribute()
            op_dist_attr.process_mesh = process_mesh
            var_names = list(dims_mapping_dict.keys())

            for idx, dims_mapping in enumerate(composed_dims_mapping):
                if var_names[idx] in op.input_arg_names:
                    op_dist_attr.set_input_dims_mapping(var_names[idx],
                                                        dims_mapping)
                elif var_names[idx] in op.output_arg_names:
                    op_dist_attr.set_output_dims_mapping(var_names[idx],
                                                         dims_mapping)
                else:
                    raise ValueError(
                        "The {varname} is not input or output of op {op}.".
                        format(
                            varname='var_names[idx]', op='op'))

            dist_op = DistributedOperator(op, op_dist_attr)
            if dist_op_impl_container is None:
                if is_elementwise_like_op(op.type):
                    changed = True
                    valid = True
                    try:
                        changed = update_op_dims_mapping_by_elementwise_like_dist_impl(
                            dist_op)
                    except Exception as e:
                        valid = False
                    if valid and not changed:
                        if PlanFilter.check_dims_mapping_for_op(
                                op, dist_op.dist_attr, vars
                        ) and PlanFilter.check_dims_mapping_for_special_op(
                                op, dist_op.dist_attr, vars):
                            dist_op.dist_attr.impl_idx = -1
                            op_valid_dist_attrs.append(dist_op.dist_attr)
                    continue
                else:
                    changed = True
                    valid = True
                    try:
                        changed = update_op_dims_mapping_by_default_dist_impl(
                            dist_op)
                    except Exception as e:
                        valid = False
                    if valid and not changed:
                        if PlanFilter.check_dims_mapping_for_op(
                                op, dist_op.dist_attr, vars
                        ) and PlanFilter.check_dims_mapping_for_special_op(
                                op, dist_op.dist_attr, vars):
                            dist_op.dist_attr.impl_idx = -2
                            op_valid_dist_attrs.append(dist_op.dist_attr)
                    continue

            # if op has distributed implements, find all valid dist attr of this op
            impls = dist_op_impl_container.get_impls()
            for idx, impl in enumerate(impls):
                if impl.is_auto_compatible(dist_op):
                    if PlanFilter.check_dims_mapping_for_op(
                            op, dist_op.dist_attr, vars):
                        dist_op.dist_attr.impl_idx = idx
                        op_valid_dist_attrs.append(dist_op.dist_attr)

        # set default dist attr for some special ops whose distributed attributes can not be enumerated
        if not op_valid_dist_attrs:
            op_dist_attr = OperatorDistributedAttribute()
            op_dist_attr.process_mesh = process_mesh
            dist_op = DistributedOperator(op, op_dist_attr)
            for var_name in op.input_arg_names:
                op_dist_attr.set_input_dims_mapping(
                    vars[var_name], [-1 for i in vars[var_name].shape])
            for var_name in op.output_arg_names:
                op_dist_attr.set_output_dims_mapping(
                    vars[var_name], [-1 for i in vars[var_name].shape])
            dist_op.dist_attr.impl_idx = -1
            op_valid_dist_attrs.append(dist_op.dist_attr)

        return op_valid_dist_attrs

    @staticmethod
    def enum_valid_dist_attr_for_program(program,
                                         process_mesh_topology,
                                         is_pipeline=False):
        """Enumerate valid distributed attributes for all ops in program."""
        valid_dist_attr_dict = OrderedDict()
        ops = program.global_block().ops
        vars = program.global_block().vars

        processes = reduce(lambda x, y: x * y, process_mesh_topology)
        global_group = [i for i in range(processes)]
        global_process_mesh = None
        pipeline_process_meshes = None

        # in the pipeline mode, there are some process meshes
        if is_pipeline:
            pipeline_stages = process_mesh_topology[-1]
            op_count_per_stage = len(ops) // pipeline_stages
            if len(process_mesh_topology) > 1:
                process_mesh_shape = process_mesh_topology[:-1]
                per_process_mesh_group = processes // pipeline_stages
                pipeline_process_meshes = [auto.ProcessMesh(mesh=np.array(global_group[i*per_process_mesh_group: \
                (i+1)*per_process_mesh_group]).reshape(process_mesh_shape).tolist()) for i in range(pipeline_stages)]
            elif len(process_mesh_topology) == 1:
                pipeline_process_meshes = [
                    auto.ProcessMesh(mesh=[i]) for i in range(pipeline_stages)
                ]
        else:
            if len(process_mesh_topology) > 1:
                global_process_mesh = auto.ProcessMesh(mesh=np.array(
                    global_group).reshape(process_mesh_topology).tolist())
            else:
                global_process_mesh = auto.ProcessMesh(mesh=global_group)

        # enumerate valid distributed attribute for each op in the program
        for idx, op in enumerate(ops):
            op_valid_dist_attrs = None
            op_process_mesh = global_process_mesh
            pipeline_stage = -1
            if pipeline_process_meshes is not None:
                pipeline_stage = idx // op_count_per_stage if idx // op_count_per_stage < len(
                    pipeline_process_meshes) else idx // op_count_per_stage - 1
                if pipeline_stage >= len(pipeline_process_meshes):
                    pipeline_stage = len(pipeline_process_meshes) - 1
                op_process_mesh = pipeline_process_meshes[pipeline_stage]

            if op.type in PlanSpace.not_enum_ops:
                op_dist_attr = OperatorDistributedAttribute()
                op_dist_attr.process_mesh = op_process_mesh
                for var_name in op.input_arg_names:
                    if var_name in PlanSpace.special_vars:
                        op_dist_attr.set_input_dims_mapping(var_name, [])
                    else:
                        dims_mapping = [-1 for i in vars[var_name].shape]
                        op_dist_attr.set_input_dims_mapping(var_name,
                                                            dims_mapping)

                for var_name in op.output_arg_names:
                    if var_name in PlanSpace.special_vars:
                        op_dist_attr.set_output_dims_mapping(var_name, [])
                    else:
                        dims_mapping = [-1 for i in vars[var_name].shape]
                        op_dist_attr.set_output_dims_mapping(var_name,
                                                             dims_mapping)
                op_valid_dist_attrs = [op_dist_attr]
                pipeline_stage = 0 if pipeline_stage != -1 else pipeline_stage
            else:
                op_valid_dist_attrs = PlanSpace._enum_valid_dist_attr_for_op(
                    program, op, op_process_mesh)

            assert op_valid_dist_attrs is not None, "Enumerate {} valid distributed attribute failed.".format(
                op)
            valid_dist_attr_dict[op.desc.id(
            )] = [op_valid_dist_attrs, pipeline_stage]

        return valid_dist_attr_dict, pipeline_process_meshes, global_process_mesh


class SearchAlgorithm:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        self.name = name

    def search(self):
        raise NotImplementedError("Please Implement this method in subclass.")


class MCMC(SearchAlgorithm):
    def __init__(self, serial_program_info, parallelizer, max_search_times=5):
        super(MCMC, self).__init__("mcmc")
        self._serial_program_info = serial_program_info
        self._max_search_times = max_search_times
        self._parallelizer = parallelizer

    @property
    def serial_program_info(self):
        return self._serial_program_info

    @property
    def parallelizer(self):
        return self._parallelizer

    @property
    def max_search_times(self):
        return self._max_search_times

    def make_special_op_unshard(self, op, ops, vars, dist_context,
                                valid_dist_attr_dict):
        if op.type == "softmax_with_cross_entropy":
            for var_name in op.input_arg_names:
                dims_mapping = dist_context.get_op_dist_attr_for_program(
                    op).get_input_dims_mapping(var_name)
                if dims_mapping != dist_context.get_tensor_dist_attr_for_program(
                        vars[var_name]).dims_mapping:
                    has_changed = False
                    for search_op in ops:
                        if var_name in search_op.output_arg_names:
                            op_dist_attr_list = valid_dist_attr_dict[
                                search_op.desc.id()][0]
                            for op_dist_attr in op_dist_attr_list:
                                if op_dist_attr.get_output_dims_mapping(
                                        var_name) == dims_mapping:
                                    dist_context.set_op_dist_attr_for_program(
                                        search_op, op_dist_attr)
                                    tensor_dist_attr = TensorDistributedAttribute(
                                    )
                                    tensor_dist_attr.process_mesh = op_dist_attr.process_mesh
                                    tensor_dist_attr.dims_mapping = op_dist_attr.get_output_dims_mapping(
                                        var_name)
                                    dist_context.set_tensor_dist_attr_for_program(
                                        vars[var_name], tensor_dist_attr)
                                    has_changed = True
                                    break
                        if has_changed:
                            break
                    if not has_changed:
                        raise ValueError(
                            "Change softmax_with_cross_entropy dist attr failed")

    def init_program(self, valid_dist_attr_dict, program,
                     pipeline_process_meshes, global_process_mesh):
        ops = program.global_block().ops
        vars = program.global_block().vars
        new_dist_context = DistributedContext()

        for op in ops:
            op_valid_dist_attr_list = valid_dist_attr_dict[op.desc.id()][0]
            random_op_dist_attr = np.random.randint(
                len(op_valid_dist_attr_list))
            init_op_dist_attr = op_valid_dist_attr_list[random_op_dist_attr]
            new_dist_context.set_op_dist_attr_for_program(op, init_op_dist_attr)
            for var_name in op.input_arg_names:
                if var_name == "lod_tensor_blocking_queue_0":
                    continue
                if new_dist_context.get_tensor_dist_attr_for_program(vars[
                        var_name]) is None:
                    tensor_dist_attr = TensorDistributedAttribute()
                    tensor_dist_attr.process_mesh = init_op_dist_attr.process_mesh
                    tensor_dist_attr.dims_mapping = init_op_dist_attr.get_input_dims_mapping(
                        var_name)
                    new_dist_context.set_tensor_dist_attr_for_program(
                        vars[var_name], tensor_dist_attr)

            for var_name in op.output_arg_names:
                tensor_dist_attr = TensorDistributedAttribute()
                tensor_dist_attr.process_mesh = init_op_dist_attr.process_mesh
                tensor_dist_attr.dims_mapping = init_op_dist_attr.get_output_dims_mapping(
                    var_name)
                new_dist_context.set_tensor_dist_attr_for_program(
                    vars[var_name], tensor_dist_attr)

            # NOTE: this is a temporary solution to make softmax_with_cross_entropy unshard
            self.make_special_op_unshard(op, ops, vars, new_dist_context,
                                         valid_dist_attr_dict)

        # add process meshes to distributed context
        if global_process_mesh is not None:
            new_dist_context.add_process_mesh(global_process_mesh)
        elif pipeline_process_meshes is not None:
            for process_mesh in pipeline_process_meshes:
                new_dist_context.add_process_mesh(process_mesh)

        return new_dist_context

    def estimate_searched_strategy_cost(self,
                                        dist_context,
                                        pipeline_process_meshes=None):
        cost = None
        # get all distributed programs
        all_dist_main_program = get_all_distributed_main_program(
            self.serial_program_info, dist_context, self.parallelizer)
        pipeline_config = [
            process_mesh.processes for process_mesh in pipeline_process_meshes
        ] if pipeline_process_meshes is not None else None
        microbatch_size = 1
        for program in all_dist_main_program:
            searched_batch_size = False
            for var in program.list_vars():
                if var.is_data and "@RESHARD" in var.name:
                    microbatch_size = var.shape[0]
                    searched_batch_size = True
                    break
            if searched_batch_size:
                break

        from .utils import get_standalone_cost_data
        standalone_cost_data = get_standalone_cost_data(all_dist_main_program)

        # cost model does not support cluster argument
        cost = estimate_cost(
            all_dist_main_program,
            cluster=None,
            pipeline_config=pipeline_config,
            standalone_cost_data=standalone_cost_data,
            batch_size=microbatch_size)

        return cost

    def set_tensor_dist_attr(self, op, op_dist_attr, vars, dist_context):
        # set output tensor distributed attribute
        for var_name in op.output_arg_names:
            process_mesh = op_dist_attr.process_mesh
            tensor_dist_attr = TensorDistributedAttribute()
            tensor_dist_attr.process_mesh = process_mesh
            tensor_dist_attr.dims_mapping = op_dist_attr.get_output_dims_mapping(
                var_name)
            dist_context.set_tensor_dist_attr_for_program(vars[var_name],
                                                          tensor_dist_attr)

        # set input tensor distributed attribute if input is data or parameter
        for var_name in op.input_arg_names:
            if vars[var_name].is_parameter or vars[var_name].is_data:
                process_mesh = op_dist_attr.process_mesh
                tensor_dist_attr = TensorDistributedAttribute()
                tensor_dist_attr.process_mesh = process_mesh
                tensor_dist_attr.dims_mapping = op_dist_attr.get_input_dims_mapping(
                    var_name)
                dist_context.set_tensor_dist_attr_for_program(vars[var_name],
                                                              tensor_dist_attr)

    def change_process_mesh(self, op, changed_process_mesh, vars, dist_context):
        dist_context.get_op_dist_attr_for_program(
            op).process_mesh = changed_process_mesh
        for var_name in op.output_arg_names:
            dist_context.get_tensor_dist_attr_for_program(vars[
                var_name]).process_mesh = changed_process_mesh
        for var_name in op.input_arg_names:
            if vars[var_name].is_parameter or vars[var_name].is_data:
                dist_context.get_tensor_dist_attr_for_program(vars[
                    var_name]).process_mesh = changed_process_mesh

    def search_once(self,
                    program,
                    valid_dist_attr_dict,
                    dist_context,
                    pipeline_process_meshes=None):
        raw_ops = program.global_block().ops
        ops = []
        for op in raw_ops:
            if op.type not in PlanSpace.not_enum_ops:
                ops.append(op)
        assert ops, "The ops of program have no distributed attributes."
        vars = program.global_block().vars
        new_dist_context = copy.deepcopy(dist_context)
        new_dist_context._dist_op_context = DistributedOperatorContext()
        new_valid_dist_attr_dict = None
        random_selected_op_idx = np.random.randint(len(ops))
        selected_op = ops[random_selected_op_idx]
        op_valid_dist_attr_list = valid_dist_attr_dict[selected_op.desc.id()][0]
        pipeline_stage = valid_dist_attr_dict[selected_op.desc.id()][1]
        random_selected_dist_attr_idx = np.random.randint(
            len(op_valid_dist_attr_list))
        selected_op_dist_attr = copy.deepcopy(op_valid_dist_attr_list[
            random_selected_dist_attr_idx])

        start_idx = ops[0].desc.id()
        if pipeline_stage > -1:
            # in pipeline mode, the above phase just select a dims mapping
            # 0 represents not changed, 1 represents to be the same with before stage, 2 represents to be the same with the latter stage
            new_valid_dist_attr_dict = copy.deepcopy(valid_dist_attr_dict)
            changed_mode = np.random.randint(3)
            if changed_mode == 0:
                # not change the process mesh, just change dims mapping
                new_dist_context.set_op_dist_attr_for_program(
                    selected_op, selected_op_dist_attr)
                self.set_tensor_dist_attr(selected_op, selected_op_dist_attr,
                                          vars, new_dist_context)

            elif changed_mode == 1:
                changed_stage = pipeline_stage - 1
                if changed_stage == -1 or random_selected_op_idx == len(ops) - 1 or \
                (random_selected_op_idx + 1 == len(ops) - 1 and new_valid_dist_attr_dict[ops[random_selected_op_idx + 1].desc.id()][1] == pipeline_stage + 1 ):
                    new_dist_context.set_op_dist_attr_for_program(
                        selected_op, selected_op_dist_attr)
                    self.set_tensor_dist_attr(selected_op,
                                              selected_op_dist_attr, vars,
                                              new_dist_context)

                else:
                    selected_op_process_mesh = pipeline_process_meshes[
                        pipeline_stage]
                    next_op_id = ops[random_selected_op_idx + 1].desc.id()
                    if new_valid_dist_attr_dict[next_op_id][
                            1] == pipeline_stage + 1 and random_selected_op_idx + 1 != len(
                                ops) - 1:
                        new_valid_dist_attr_dict[next_op_id][1] = pipeline_stage
                        for op_dist_attr in new_valid_dist_attr_dict[
                                next_op_id][0]:
                            op_dist_attr.process_mesh = selected_op_process_mesh
                        # set next op dist attr in the discontext and output/input tensor process mesh
                        self.change_process_mesh(
                            ops[random_selected_op_idx + 1],
                            selected_op_process_mesh, vars, new_dist_context)

                    # change the selected op stage and output dist attr
                    new_valid_dist_attr_dict[selected_op.desc.id()][
                        1] = changed_stage
                    new_process_mesh = pipeline_process_meshes[changed_stage]
                    selected_op_dist_attr.process_mesh = new_process_mesh
                    for op_dist_attr in new_valid_dist_attr_dict[
                            selected_op.desc.id()][0]:
                        op_dist_attr.process_mesh = new_process_mesh
                    new_dist_context.set_op_dist_attr_for_program(
                        selected_op, selected_op_dist_attr)

                    self.set_tensor_dist_attr(selected_op,
                                              selected_op_dist_attr, vars,
                                              new_dist_context)

                    # change the pre op stage
                    for idx in range(random_selected_op_idx - 1, -1, -1):
                        stage = new_valid_dist_attr_dict[ops[idx].desc.id()][1]
                        valid_dist_attr_list = new_valid_dist_attr_dict[ops[
                            idx].desc.id()][0]
                        new_process_mesh = pipeline_process_meshes[
                            changed_stage]
                        if stage == changed_stage + 1:
                            new_valid_dist_attr_dict[ops[idx].desc.id()][
                                1] = changed_stage
                            for op_dist_attr in valid_dist_attr_list:
                                op_dist_attr.process_mesh = new_process_mesh
                            new_dist_context.get_op_dist_attr_for_program(ops[
                                idx]).process_mesh = new_process_mesh
                            # change process mesh of the output and input tensor
                            self.change_process_mesh(ops[idx], new_process_mesh,
                                                     vars, new_dist_context)
                        else:
                            break

            else:
                changed_stage = pipeline_stage + 1
                if changed_stage == len(
                        pipeline_process_meshes) or random_selected_op_idx == 0 or \
                        (new_valid_dist_attr_dict[ops[random_selected_op_idx - 1].desc.id()][1] == pipeline_stage - 1 and (random_selected_op_idx == 1)):
                    new_dist_context.set_op_dist_attr_for_program(
                        selected_op, selected_op_dist_attr)
                    self.set_tensor_dist_attr(selected_op,
                                              selected_op_dist_attr, vars,
                                              new_dist_context)

                else:
                    selected_op_process_mesh = pipeline_process_meshes[
                        pipeline_stage]
                    pre_op_id = ops[random_selected_op_idx - 1].desc.id()
                    if new_valid_dist_attr_dict[pre_op_id][
                            1] == pipeline_stage - 1 and random_selected_op_idx != 1:
                        new_valid_dist_attr_dict[pre_op_id][1] = pipeline_stage
                        for op_dist_attr in new_valid_dist_attr_dict[pre_op_id][
                                0]:
                            op_dist_attr.process_mesh = selected_op_process_mesh
                        # set pre op dist attr in the discontext and output tensor process mesh
                        self.change_process_mesh(
                            ops[random_selected_op_idx - 1],
                            selected_op_process_mesh, vars, new_dist_context)

                    # change the selected op stage and output tensor dist attr
                    new_valid_dist_attr_dict[selected_op.desc.id()][
                        1] = changed_stage
                    new_process_mesh = pipeline_process_meshes[changed_stage]
                    selected_op_dist_attr.process_mesh = new_process_mesh
                    for op_dist_attr in new_valid_dist_attr_dict[
                            selected_op.desc.id()][0]:
                        op_dist_attr.process_mesh = new_process_mesh
                    new_dist_context.set_op_dist_attr_for_program(
                        selected_op, selected_op_dist_attr)
                    self.set_tensor_dist_attr(selected_op,
                                              selected_op_dist_attr, vars,
                                              new_dist_context)

                    # change the next op stage
                    for idx in range(random_selected_op_idx + 1, len(ops)):
                        stage = new_valid_dist_attr_dict[ops[idx].desc.id()][1]
                        valid_dist_attr_list = new_valid_dist_attr_dict[ops[
                            idx].desc.id()][0]
                        new_process_mesh = pipeline_process_meshes[
                            changed_stage]
                        if stage == changed_stage - 1:
                            new_valid_dist_attr_dict[ops[idx].desc.id()][
                                1] = changed_stage
                            for op_dist_attr in valid_dist_attr_list:
                                op_dist_attr.process_mesh = new_process_mesh

                            new_dist_context.get_op_dist_attr_for_program(ops[
                                idx]).process_mesh = new_process_mesh
                            # change the output tensor dist attr
                            self.change_process_mesh(ops[idx], new_process_mesh,
                                                     vars, new_dist_context)
                        else:
                            break
        else:
            new_dist_context.set_op_dist_attr_for_program(selected_op,
                                                          selected_op_dist_attr)
            self.set_tensor_dist_attr(selected_op, selected_op_dist_attr, vars,
                                      new_dist_context)

        for op in ops:
            # make softmax_with_cross_entropy unshard
            if op.type == "softmax_with_cross_entropy":
                self.make_special_op_unshard(op, ops, vars, new_dist_context,
                                             valid_dist_attr_dict)
                break

        if new_valid_dist_attr_dict is None:
            return valid_dist_attr_dict, new_dist_context
        else:
            return new_valid_dist_attr_dict, new_dist_context

    def _search_core(self,
                     valid_dist_attr_dict,
                     init_dist_context,
                     pipeline_process_meshes=None):
        times = 0
        best_dist_context = init_dist_context
        cost = self.estimate_searched_strategy_cost(
            init_dist_context, pipeline_process_meshes).runtime
        min_cost = cost
        while times < self.max_search_times:
            times += 1
            new_dist_context = self.search_once(
                self.serial_program_info.train_program, valid_dist_attr_dict,
                best_dist_context, pipeline_process_meshes)[1]
            cur_cost = self.estimate_searched_strategy_cost(
                new_dist_context, pipeline_process_meshes).runtime
            if (min_cost - cur_cost) > 0:
                best_dist_context = copy.deepcopy(new_dist_context)
                min_cost = cur_cost
                times = 0
        return best_dist_context, min_cost

    def search(self):
        logging.info("Start MCMC searching.")
        start_time = time.time()
        train_program = self.serial_program_info.train_program
        cluster = self.serial_program_info.cluster
        processes = paddle.distributed.get_world_size(
        ) if cluster is None else len(cluster.get_all_devices("GPU"))
        assert processes > 0, "Get process failed."

        process_mesh_topology_list = PlanSpace.enum_process_mesh_topology(
            processes)
        searched_dist_context = None
        min_cost = None

        searched_pipeline_dist_context = None
        pipeline_min_cost = None
        for process_mesh_topology in process_mesh_topology_list:
            logging.info(
                "MCMC search: search process mesh {} with pipeline mode.".
                format(process_mesh_topology))
            valid_dist_attr_dict, pipeline_process_meshes, global_process_mesh = PlanSpace.enum_valid_dist_attr_for_program(
                train_program, process_mesh_topology, True)
            init_dist_context = self.init_program(
                valid_dist_attr_dict, train_program, pipeline_process_meshes,
                global_process_mesh)
            best_dist_context, cost = self._search_core(valid_dist_attr_dict,
                                                        init_dist_context,
                                                        pipeline_process_meshes)
            logging.info(
                "MCMC search: the min cost is {} in the process mesh {} with pipeline mode.".
                format(cost, process_mesh_topology))
            best_dist_context._dist_op_context = DistributedOperatorContext()
            pipeline_min_cost = cost if pipeline_min_cost is None else pipeline_min_cost
            searched_pipeline_dist_context = best_dist_context if searched_pipeline_dist_context is None else searched_pipeline_dist_context
            if pipeline_min_cost > cost:
                searched_pipeline_dist_context = best_dist_context
                pipeline_min_cost = cost

        searched_non_pipeline_dist_context = None
        non_pipeline_min_cost = None
        for process_mesh_topology in process_mesh_topology_list:
            # if process_mesh_topology shape is 3, include pipeline mode by default
            if len(process_mesh_topology) == 3:
                continue
            logging.info(
                "MCMC search: search process mesh {} without pipeline mode.".
                format(process_mesh_topology))
            valid_dist_attr_dict, pipeline_process_meshes, global_process_mesh = PlanSpace.enum_valid_dist_attr_for_program(
                train_program, process_mesh_topology, False)
            init_dist_context = self.init_program(
                valid_dist_attr_dict, train_program, pipeline_process_meshes,
                global_process_mesh)
            best_dist_context, cost = self._search_core(valid_dist_attr_dict,
                                                        init_dist_context,
                                                        pipeline_process_meshes)
            logging.info(
                "MCMC search: the min cost is {} in the process mesh {} without pipeline mode.".
                format(cost, process_mesh_topology))
            best_dist_context._dist_op_context = DistributedOperatorContext()
            non_pipeline_min_cost = cost if non_pipeline_min_cost is None else non_pipeline_min_cost
            searched_non_pipeline_dist_context = best_dist_context if searched_non_pipeline_dist_context is None else searched_non_pipeline_dist_context
            if non_pipeline_min_cost > cost:
                searched_non_pipeline_dist_context = best_dist_context
                non_pipeline_min_cost = cost

        if non_pipeline_min_cost > pipeline_min_cost:
            searched_dist_context = searched_pipeline_dist_context
            min_cost = pipeline_min_cost
            logging.info(
                "Better set FLAGS_benchmark=1 to avoid hang problem in the pipeline mode."
            )
        else:
            searched_dist_context = searched_non_pipeline_dist_context
            min_cost = non_pipeline_min_cost

        # rebuild g_process_group
        pg0 = get_process_group(0)
        for process_mesh in searched_dist_context._process_meshes:
            pg0.add_ranks(process_mesh.processes)
        end_time = time.time()
        logging.info(
            "End MCMC searching: the min cost is {} and the search time is {}s.".
            format(min_cost, end_time - start_time))
        return searched_dist_context, min_cost


class Planner:
    def __init__(self, serial_program_info, parallelizer,
                 algorithm_config=None):
        self._serial_program_info = serial_program_info
        self._parallelizer = parallelizer
        self._algorithm_config = algorithm_config
        self._algorithm_searcher = self.create_algorithm_searcher(
            algorithm_config)

    @property
    def serial_program_info(self):
        return self._serial_program_info

    @property
    def algorithm_config(self):
        return self._algorithm_config

    @property
    def algorithm_searcher(self):
        return self._algorithm_searcher

    @property
    def parallelizer(self):
        return self._parallelizer

    def create_algorithm_searcher(self, algorithm_config):
        name = algorithm_config.get("name", None)
        assert name is not None, "Invalid algorithm config."

        algorithm_searcher = None
        if name == "mcmc":
            # NOTE: Only GPU clusters are supported now.
            max_search_times = algorithm_config.get("max_search_times", None)
            algorithm_searcher = MCMC(
                self.serial_program_info, self.parallelizer,
                max_search_times) if max_search_times is not None else MCMC(
                    self.serial_program_info, self.parallelizer)
        else:
            raise NotImplementedError(
                "Other search algorithms have not been supported now.")

        return algorithm_searcher

    def search(self):
        return self.algorithm_searcher.search()
