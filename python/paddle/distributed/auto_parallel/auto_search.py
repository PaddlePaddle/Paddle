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
from itertools import chain, product
from functools import reduce
from collections import OrderedDict
import math

import numpy as np

import paddle
import paddle.nn as nn
import paddle.static as static
import paddle.nn.functional as F
import paddle.utils as utils
from paddle.fluid import core
import paddle.distributed.auto_parallel as auto
from paddle.distributed.auto_parallel.context import get_default_distributed_context
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.partitioner import Partitioner
from paddle.distributed.auto_parallel.reshard import reshard
from paddle.distributed.auto_parallel.process import new_process_group
from paddle.distributed.auto_parallel.operators.common import get_distributed_operator
from paddle.distributed.auto_parallel.context import DistributedContext
from paddle.distributed.auto_parallel.attribute import OperatorDistributedAttribute, TensorDistributedAttribute
from paddle.distributed.auto_parallel.completion import update_op_dims_mapping_by_elementwise_like_dist_impl
from paddle.distributed.auto_parallel.completion import update_op_dims_mapping_by_default_dist_impl
from paddle.distributed.auto_parallel.completion import is_elementwise_like_op
from paddle.distributed.auto_parallel.utils import make_data_unshard

paddle.enable_static()


def enumerate_process_mesh(processes):
    """enumerate all process meshes with the given processes"""
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


def enumerate_dims_mapping(process_mesh_topology, visited, path, depth, res,
                           tensor_shape):
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
            enumerate_dims_mapping(process_mesh_topology, visited, path,
                                   depth + 1, res, tensor_shape)
            visited[i] = False
            path.pop()


def check_op_dims_mapping(op, op_dist_attr, vars):
    """Check the op dims_mapping whether valid"""
    process_mesh = op_dist_attr.get_process_mesh()
    assert process_mesh is not None, "The process mesh should not be None."
    for var_name in op.input_arg_names:
        dims_mapping = op_dist_attr.get_input_dims_mapping(var_name)
        if not check_dims_mapping(process_mesh.topology, vars[var_name].shape,
                                  dims_mapping):
            return False

    for var_name in op.output_arg_names:
        dims_mapping = op_dist_attr.get_output_dims_mapping(var_name)
        if not check_dims_mapping(process_mesh.topology, vars[var_name].shape,
                                  dims_mapping):
            return False

    return True


def check_dims_mapping(process_mesh_topology, tensor_shape, dims_mapping):
    valid = True
    assert len(tensor_shape) == len(dims_mapping)
    for idx, item in enumerate(dims_mapping):
        if item != -1:
            if tensor_shape[idx] % process_mesh_topology[
                    item] != 0 or dims_mapping.count(item) > 1:
                valid = False
    return valid


def enumerate_op_valid_dist_attr(program, op, process_mesh, dist_context):
    """enumerate the valid dist attr of one op based on the given process mesh."""
    vars = program.global_block().vars
    dims_mapping_dict = OrderedDict()
    valid_op_dist_attr_list = []
    elementwise_op_dist_attr_list = []
    default_op_dist_attr_list = []
    dist_op = get_distributed_operator(op.type)

    # enumerate valid dims_mapping of tensor when process_mesh given
    for var_name in chain(op.input_arg_names, op.output_arg_names):
        visited = [
            False
            for _ in range(len(list(range(-1, len(process_mesh.topology)))))
        ]
        depth = 0
        path = []
        dims_mapping_list = []
        enumerate_dims_mapping(process_mesh.topology, visited, path, depth,
                               dims_mapping_list, vars[var_name].shape)
        dims_mapping_dict[var_name] = copy.deepcopy(dims_mapping_list)

    # compose dims_mapping of all tensor
    composed_dims_mapping_list = list(
        product(*[dims_mapping_dict[key] for key in dims_mapping_dict.keys()]))
    for composed_dims_mapping in composed_dims_mapping_list:
        op_dist_attr = OperatorDistributedAttribute(op, dist_context)
        op_dist_attr.set_process_mesh(process_mesh)
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
                    "The {varname} is not input or output of op {op}.".format(
                        varname='var_names[idx]', op='op'))

        if dist_op is None:
            # if a op has no distributed implement, it has elementwise or default implement.
            if is_elementwise_like_op(op.type):
                changed = True
                valid = True
                while changed:
                    # use try...except due to the called api will raise exception.
                    try:
                        changed = update_op_dims_mapping_by_elementwise_like_dist_impl(
                            op_dist_attr)
                    except:
                        valid = False
                        break
                if valid:
                    # to ensure the op dist attr different and valid
                    if check_op_dims_mapping(op, op_dist_attr, vars):
                        _ = []
                        for var_name in op.input_arg_names:
                            _.append(
                                op_dist_attr.get_input_dims_mapping(var_name))
                        for var_name in op.output_arg_names:
                            _.append(
                                op_dist_attr.get_output_dims_mapping(var_name))
                        if _ not in elementwise_op_dist_attr_list:
                            op_dist_attr.set_impl_idx(-1)
                            valid_op_dist_attr_list.append(op_dist_attr)
                            elementwise_op_dist_attr_list.append(_)
                continue
            else:
                changed = True
                valid = True
                while changed:
                    try:
                        changed = update_op_dims_mapping_by_default_dist_impl(
                            op_dist_attr)
                    except:
                        valid = False
                        break
                if valid:
                    if check_op_dims_mapping(op, op_dist_attr, vars):
                        _ = []
                        for var_name in op.input_arg_names:
                            _.append(
                                op_dist_attr.get_input_dims_mapping(var_name))
                        for var_name in op.output_arg_names:
                            _.append(
                                op_dist_attr.get_output_dims_mapping(var_name))
                        if _ not in default_op_dist_attr_list:
                            op_dist_attr.set_impl_idx(-2)
                            valid_op_dist_attr_list.append(op_dist_attr)
                            default_op_dist_attr_list.append(_)
                continue
        # if op has distributed implements, find all valid dist attr of this op
        for idx, impl in enumerate(dist_op.get_impls()):
            if impl.is_compatible(op_dist_attr):
                if check_op_dims_mapping(op, op_dist_attr, vars):
                    op_dist_attr.set_impl_idx(idx)
                    valid_op_dist_attr_list.append(op_dist_attr)

    return valid_op_dist_attr_list


def check_ops_valid_dist_attr(program, op_valid_dist_attr_dict):
    pass


def enumerate_ops_valid_dist_attr(program,
                                  process_mesh_topology,
                                  dist_context,
                                  pipeline_mode=False):
    op_valid_dist_attr_dict = OrderedDict()
    ops = program.global_block().ops
    processes = reduce(lambda x, y: x * y, process_mesh_topology)
    process_mesh_global_group = [i for i in range(processes)]
    global_process_mesh = auto.ProcessMesh(mesh=list(
        np.array(process_mesh_global_group).reshape(process_mesh_topology)))
    pipeline_process_mesh_list = None
    if pipeline_mode:
        pipeline_stages = process_mesh_topology[-1]
        op_count_per_stage = len(ops) // pipeline_stages
        process_mesh_shape = process_mesh_topology[:-1]
        per_process_mesh_group = processes // pipeline_stages
        pipeline_process_mesh_list = [auto.ProcessMesh(mesh=list(np.array(process_mesh_global_group[i*per_process_mesh_group: \
        (i+1)*per_process_mesh_group]).reshape(process_mesh_shape)), parent=global_process_mesh) for i in range(pipeline_stages)]

    for idx, op in enumerate(ops):
        op_process_mesh = global_process_mesh
        pipeline_stage = -1
        if pipeline_process_mesh_list is not None:
            pipeline_stage = idx // op_count_per_stage if idx // op_count_per_stage < len(
                pipeline_process_mesh_list) else idx // op_count_per_stage - 1
            op_process_mesh = pipeline_process_mesh_list[pipeline_stage]
        op_valid_dist_attr_list = enumerate_op_valid_dist_attr(
            program, op, op_process_mesh, dist_context)
        op_valid_dist_attr_dict[op.desc.id(
        )] = [op_valid_dist_attr_list, pipeline_stage]

    check_ops_valid_dist_attr(program, op_valid_dist_attr_dict)
    return op_valid_dist_attr_dict, pipeline_process_mesh_list


def mcmc_search_strategy(program,
                         op_valid_dist_attr_dict,
                         dist_context,
                         pipeline_process_mesh_list=None):
    """search a distributed training strategy after init"""
    # randomly select one op and op dist attr corresponding to the selected op
    ops = program.global_block().ops
    vars = program.global_block().vars
    new_dist_context = copy.deepcopy(dist_context)
    new_op_valid_dist_attr_dict = None
    random_selected_idx = np.random.randint(len(ops))
    selected_op = ops[random_selected_idx]
    op_valid_dist_attr_list = op_valid_dist_attr_dict[selected_op.desc.id()][0]
    pipeline_stage = op_valid_dist_attr_dict[selected_op.desc.id()][1]
    random_selected_idx = np.random.randint(len(op_valid_dist_attr_list))
    selected_op_dist_attr = copy.deepcopy(op_valid_dist_attr_list[
        random_selected_idx])

    start_idx = ops[0].desc.id()
    if pipeline_stage > -1:
        # in pipeline mode, the above phase just select a dims mapping
        # 0 represents not changed, 1 represents to be the same with before stage, 2 represents to be the same with the latter stage
        new_op_valid_dist_attr_dict = copy.deepcopy(op_valid_dist_attr_dict)
        changed_mode = np.random.randint(3)
        if changed_mode == 0:
            # not change the process mesh, just change dims mapping
            new_dist_context.set_op_distributed_attr_for_program(
                selected_op, selected_op_dist_attr)
        elif changed_mode == 1:
            changed_stage = pipeline_stage - 1
            if changed_stage == -1 or random_selected_idx == len(ops) - 1:
                new_dist_context.set_op_distributed_attr_for_program(
                    selected_op, selected_op_dist_attr)
            else:
                selected_op_process_mesh = pipeline_process_mesh_list[
                    pipeline_stage]
                next_op_id = ops[random_selected_idx + 1].desc.id()
                if new_op_valid_dist_attr_dict[next_op_id][
                        1] == pipeline_stage + 1:
                    new_op_valid_dist_attr_dict[next_op_id][1] = pipeline_stage
                    for op_dist_attr in new_op_valid_dist_attr_dict[next_op_id]:
                        op_dist_attr.set_process_mesh(selected_op_process_mesh)
                    # set next op dist attr in the discontext and output tensor process mesh
                    new_dist_context.get_op_distributed_attr_for_program(ops[
                        random_selected_idx + 1]).set_process_mesh(
                            selected_op_process_mesh)
                    for var_name in ops[random_selected_idx +
                                        1].output_arg_names:
                        new_dist_context.get_tensor_distributed_attr_for_program(
                            vars[var_name]).set_process_mesh(
                                selected_op_process_mesh)

                # change the selected op stage and output dist attr
                new_op_valid_dist_attr_dict[selected_op.desc.id()][
                    1] = changed_stage
                new_process_mesh = pipeline_process_mesh_list[changed_stage]
                selected_op_dist_attr.set_process_mesh(new_process_mesh)
                for op_dist_attr in new_op_valid_dist_attr_dict[
                        selected_op.desc.id()][0]:
                    op_dist_attr.set_process_mesh(new_process_mesh)
                new_dist_context.set_op_distributed_attr_for_program(
                    selected_op, selected_op_dist_attr)
                for var_name in selected_op.output_arg_names:
                    new_dist_context.get_tensor_distributed_attr_for_program(
                        vars[var_name]).set_process_mesh(new_process_mesh)
                    dims_mapping = selected_op_dist_attr.get_output_dims_mapping(
                        var_name)
                    new_dist_context.get_tensor_distributed_attr_for_program(
                        vars[var_name]).set_dims_mapping(dims_mapping)

                # change the pre op stage
                for idx in range(random_selected_idx - 1, -1, -1):
                    stage = new_op_valid_dist_attr_dict[ops[idx].desc.id()][1]
                    valid_dist_attr_list = new_op_valid_dist_attr_dict[ops[
                        idx].desc.id()][0]
                    new_process_mesh = pipeline_process_mesh_list[changed_stage]
                    if stage == changed_stage + 1:
                        for op_dist_attr in valid_dist_attr_list:
                            op_dist_attr.set_process_mesh(new_process_mesh)

                        new_dist_context.get_op_distributed_attr_for_program(
                            ops[idx]).set_process_mesh(new_process_mesh)
                        # change the output tensor process mesh
                        for var_name in ops[idx].output_arg_names:
                            new_dist_context.get_tensor_distributed_attr_for_program(
                                vars[var_name]).set_process_mesh(
                                    new_process_mesh)
                    else:
                        break
        else:
            changed_stage = pipeline_stage + 1
            if changed_stage == len(
                    pipeline_process_mesh_list) or random_selected_idx == 0:
                new_dist_context.set_op_distributed_attr_for_program(
                    selected_op, selected_op_dist_attr)
            else:
                selected_op_process_mesh = pipeline_process_mesh_list[
                    pipeline_stage]
                pre_op_id = ops[random_selected_idx - 1].desc.id()
                if new_op_valid_dist_attr_dict[pre_op_id][
                        1] == pipeline_stage - 1:
                    new_op_valid_dist_attr_dict[pre_op_id][1] = pipeline_stage
                    for op_dist_attr in new_op_valid_dist_attr_dict[pre_op_id]:
                        op_dist_attr.set_process_mesh(selected_op_process_mesh)
                    # set pre op dist attr in the discontext and output tensor process mesh
                    new_dist_context.get_op_distributed_attr_for_program(ops[
                        random_selected_idx - 1]).set_process_mesh(
                            selected_op_process_mesh)
                    for var_name in ops[random_selected_idx -
                                        1].output_arg_names:
                        new_dist_context.get_tensor_distributed_attr_for_program(
                            vars[var_name]).set_process_mesh(
                                selected_op_process_mesh)

                # change the selected op stage and output tensor dist attr
                new_op_valid_dist_attr_dict[selected_op.desc.id()][
                    1] = changed_stage
                new_process_mesh = pipeline_process_mesh_list[changed_stage]
                selected_op_dist_attr.set_process_mesh(new_process_mesh)
                for op_dist_attr in new_op_valid_dist_attr_dict[
                        selected_op.desc.id()][0]:
                    op_dist_attr.set_process_mesh(new_process_mesh)
                new_dist_context.set_op_distributed_attr_for_program(
                    selected_op, selected_op_dist_attr)
                for var_name in selected_op.output_arg_names:
                    new_dist_context.get_tensor_distributed_attr_for_program(
                        vars[var_name]).set_process_mesh(new_process_mesh)
                    dims_mapping = selected_op_dist_attr.get_output_dims_mapping(
                        var_name)
                    new_dist_context.get_tensor_distributed_attr_for_program(
                        vars[var_name]).set_dims_mapping(dims_mapping)

                # change the pre op stage
                for idx in range(random_selected_idx - 1, -1, -1):
                    stage = new_op_valid_dist_attr_dict[ops[idx].desc.id()][1]
                    valid_dist_attr_list = new_op_valid_dist_attr_dict[ops[
                        idx].desc.id()][0]
                    new_process_mesh = pipeline_process_mesh_list[changed_stage]
                    if stage == changed_stage - 1:
                        for op_dist_attr in valid_dist_attr_list:
                            op_dist_attr.set_process_mesh(new_process_mesh)

                        new_dist_context.get_op_distributed_attr_for_program(
                            ops[idx]).set_process_mesh(new_process_mesh)
                        # change the output tensor dist attr
                        for var_name in ops[idx].output_arg_names:
                            new_dist_context.get_tensor_distributed_attr_for_program(
                                vars[var_name]).set_process_mesh(
                                    new_process_mesh)
                    else:
                        break
    else:
        new_dist_context.set_op_distributed_attr_for_program(
            selected_op, selected_op_dist_attr)
        for var_name in selected_op.output_arg_names:
            process_mesh = selected_op_dist_attr.get_process_mesh(var_name)
            new_dist_context.get_tensor_distributed_attr_for_program(vars[
                var_name]).set_process_mesh(process_mesh)
            dims_mapping = selected_op_dist_attr.get_output_dims_mapping(
                var_name)
            new_dist_context.get_tensor_distributed_attr_for_program(vars[
                var_name]).set_dims_mapping(dims_mapping)
        for var_name in selected_op.input_arg_names:
            process_mesh = selected_op_dist_attr.get_process_mesh(var_name)
            new_dist_context.get_tensor_distributed_attr_for_program(vars[
                var_name]).set_process_mesh(process_mesh)
            dims_mapping = selected_op_dist_attr.get_input_dims_mapping(
                var_name)
            new_dist_context.get_tensor_distributed_attr_for_program(vars[
                var_name]).set_dims_mapping(dims_mapping)

    for key in init_dist_context._op_distributed_attr_map_for_program.keys():
        init_dist_context._op_distributed_attr_map_for_program[
            key]._owner_context = new_dist_context
    for key in init_dist_context._tensor_distributed_attr_map_for_program.keys(
    ):
        init_dist_context._tensor_distributed_attr_map_for_program[
            key]._owner_context = new_dist_context
    if new_op_valid_dist_attr_dict is None:
        return op_valid_dist_attr_dict, new_dist_context
    else:
        return new_op_valid_dist_attr_dict, new_dist_context


def get_rank_id(cluster=None):
    rank_id = None
    if cluster is None:
        rank_id = paddle.distributed.get_rank()
    else:
        rank_id = cluster.get_rank()
    assert rank_id is not None, "get rank id failed."


def get_ranks(cluster=None):
    ranks = None
    if cluster is None:
        ranks = paddle.distributed.get_world_size()
    else:
        ranks = cluster.processes
    return ranks


def get_distributed_program(train_program, startup_program, dist_context, loss,
                            rank_id, optimizer):
    dist_strategy = fleet.DistributedStrategy()
    partitioner = Partitioner(dist_strategy, dist_context, rank_id)
    dist_main_program, dist_startup_program = partitioner.transpile_forward(
        train_program, startup_program)
    dist_params_grads = partitioner.apply_backward(
        loss, train_program, startup_program, dist_main_program,
        dist_startup_program)
    opt_ops = partitioner.apply_optimize(
        optimizer, dist_params_grads, dist_main_program, dist_startup_program)
    make_data_unshard(dist_main_program, dist_startup_program, dist_context)
    reshard(dist_main_program, dist_startup_program, rank_id, dist_context)
    return dist_main_program, dist_startup_program


def get_all_distributed_main_program(train_program,
                                     startup_program,
                                     dist_context,
                                     loss,
                                     optimizer,
                                     cluster=None):

    all_dist_main_program = []
    ranks = get_ranks(cluster)
    for rank_id in range(ranks):
        used_dist_context = copy.deepcopy(dist_context)
        for key in used_dist_context._op_distributed_attr_map_for_program.keys(
        ):
            used_dist_context._op_distributed_attr_map_for_program[
                key]._owner_context = used_dist_context
        for key in used_dist_context._tensor_distributed_attr_map_for_program.keys(
        ):
            used_dist_context._tensor_distributed_attr_map_for_program[
                key]._owner_context = used_dist_context
        dist_main_program, dist_startup_program = get_distributed_program(
            train_program, startup_program, used_dist_context, loss, rank_id,
            optimizer)
        all_dist_main_program.append([dist_main_program, used_dist_context])
    return all_dist_main_program


def get_single_node_data(train_program,
                         startup_program,
                         op_valid_dist_attr_dict,
                         pipeline_process_mesh_list=None):
    device = "gpu" if core.is_compiled_with_cuda() else "cpu"
    cost_model = core.CostModel()
    cost_data = cost_model.profile_measure(train_program, startup_program,
                                           device, ["time"])
    op_name2cost = []
    for i in range(len(pipeline_process_mesh_list)):
        op_name2cost.append({})
    for idx, op in enumerate(train_program.blocks[0].ops):
        pipeline_stage = op_valid_dist_attr_dict[op.desc.id()][1]
        op_name2cost[0][op.type] = cost_data.get_op_time_ms(idx)
    return op_name2cost


def estimate_searched_strategy_cost(train_program,
                                    startup_program,
                                    dist_context,
                                    loss,
                                    optimizer,
                                    standalone_cost_data,
                                    pipeline_process_mesh_list=None,
                                    cluster=None):
    cost = None
    # get all distributed programs
    all_dist_main_program = get_all_distributed_main_program(
        train_program, startup_program, dist_context, loss, optimizer)
    pipeline_config = [
        process_mesh.process_group
        for process_mesh in pipeline_process_mesh_list
    ] if pipeline_process_mesh_list is not None else None
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
    cost = estimate_cost(
        all_dist_main_program,
        cluster=cluster,
        pipeline_config=pipeline_config,
        standalone_cost_data=standalone_cost_data,
        batch_size=microbatch_size)

    return cost


def init(op_valid_dist_attr_dict, program):
    ops = program.global_block().ops
    new_dist_context = DistributedContext()
    vars = program.global_block().vars
    for op in ops:
        op_valid_dist_attr_list = op_valid_dist_attr_dict[op.desc.id()][0]
        random_op_dist_attr = np.random.randint(len(op_valid_dist_attr_list))
        init_op_dist_attr = op_valid_dist_attr_list[random_op_dist_attr]
        new_dist_context.set_op_distributed_attr_for_program(op,
                                                             init_op_dist_attr)
        for var_name in op.output_arg_names:
            tensor_dist_attr = TensorDistributedAttribute(vars[var_name],
                                                          new_dist_context)
            tensor_dist_attr.set_process_mesh(
                init_op_dist_attr.get_process_mesh())
            tensor_dist_attr.set_dims_mapping(
                init_op_dist_attr.get_output_dims_mapping(var_name))
            new_dist_context.set_tensor_distributed_attr_for_program(
                vars[var_name], tensor_dist_attr)
        for var_name in op.input_arg_names:
            tensor_dist_attr = TensorDistributedAttribute(vars[var_name],
                                                          new_dist_context)
            tensor_dist_attr.set_process_mesh(
                init_op_dist_attr.get_process_mesh())
            tensor_dist_attr.set_dims_mapping(
                init_op_dist_attr.get_input_dims_mapping(var_name))
            new_dist_context.set_tensor_distributed_attr_for_program(
                vars[var_name], tensor_dist_attr)
    for key in new_dist_context._op_distributed_attr_map_for_program.keys():
        new_dist_context._op_distributed_attr_map_for_program[
            key]._owner_context = new_dist_context
    for key in new_dist_context._tensor_distributed_attr_map_for_program.keys():
        new_dist_context._tensor_distributed_attr_map_for_program[
            key]._owner_context = new_dist_context
    return new_dist_context


def mcmc(train_program,
         start_program,
         op_valid_dist_attr_dict,
         init_dist_context,
         cluster=None,
         max_search_times=15,
         pipeline_process_mesh_list=None):
    times = 0
    MAX_SEARCH_TIMES = max_search_times
    while times < MAX_SEARCH_TIMES:
        times += 1
        standalone_cost_data = get_single_node_data(
            train_program,
            start_program,
            op_valid_dist_attr_dict,
            pipeline_process_mesh_list=pipeline_process_mesh_list)
        cost = estimate_searched_strategy_cost(
            train_program,
            start_program,
            init_dist_context,
            loss,
            optimizer,
            standalone_cost_data,
            pipeline_process_mesh_list,
            cluster=None).runtime
        MINNUM_COST = cost
        BEST_DIST_CONTEXT = init_dist_context
        new_dist_context = mcmc_search_strategy(
            train_program, op_valid_dist_attr_dict, init_dist_context,
            pipeline_process_mesh_list)
        cur_cost = estimate_searched_strategy_cost(
            train_program,
            start_program,
            new_dist_context,
            loss,
            optimizer,
            standalone_cost_data,
            pipeline_process_mesh_list,
            cluster=None).runtime
        alpha = min(1, math.exp(0.05 * (MINNUM_COST - cur_cost)))
        accp = np.random.uniform(low=0.0, high=1.0)

        if alpha < accp:
            BEST_DIST_CONTEXT = new_dist_context
            MINNUM_COST = cur_cost
            i = 0
    return BEST_DIST_CONTEXT


def auto_search(serial_main_program,
                serial_startup_program,
                search_algorithm="mcmc",
                cluster=None,
                search_config=None):
    """
    auto search steps:
    step1: enumerate all process meshes corresponding to the processes
    step2: for all process meshes, find a best strategy:
        1) non-pipeline:
        enumerate all valid op dist attr of single op
        init: randomly select a valid op dist of one op and get runtime by cost model.
        mcmc search: randomly select a op and select a valid op dist attr of its, get new runtime by cost model, compare runtime and judge replace the op dist attr
        2) pipeline:
        evenly dividing the pipeline stage of all ops, then enumerate all valid op dist attr of one op. 
        when select the op dist attr, it firstly selects dims_mapping and then select process mesh.
    """
    processes = get_ranks(cluster=None)
    process_meshes = enumerate_process_mesh(processes)

    train_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    loss, train_program, start_program = mlp_forward(train_program,
                                                     startup_program)
    dist_context = DistributedContext()
    op_valid_dist_attr_dict = enumerate_ops_valid_dist_attr(
        train_program, [2, 4], dist_context, False)[0]
    init_dist_context = init(op_valid_dist_attr_dict, train_program)
    optimizer = paddle.optimizer.SGD(learning_rate=1e-3)
    best_dist_context = mcmc(
        train_program,
        start_program,
        op_valid_dist_attr_dict,
        init_dist_context=init_dist_context,
        cluster=None,
        max_search_times=15,
        pipeline_process_mesh_list=pipeline_process_mesh_list)
    return best_dist_context
