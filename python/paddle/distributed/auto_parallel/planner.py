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
