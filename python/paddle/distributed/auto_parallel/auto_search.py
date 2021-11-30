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
import random
from itertools import chain, product
from functools import reduce
from collections import OrderedDict

import numpy as np

import paddle
from paddle.distributed import fleet
from .partitioner import Partitioner
from .utils import set_grad_var_shape
from .cost_model import estimate_cost
from paddle.cost_model import CostModel
from .dist_op import DistributedOperator
from .process_group import _g_process_group_map
from .completion import is_elementwise_like_op
import paddle.distributed.auto_parallel as auto
from .utils import is_dim_shard, compute_compatible_dims_mapping
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from .reshard import reshard, HAS_SENT, HAS_RECV, HAS_ALLGATHER
from .utils import make_data_unshard, compute_compatible_dim_mapping
from .operators.common import get_distributed_operator_impl_container
from .dist_context import DistributedContext, DistributedOperatorContext
from .dist_attribute import OperatorDistributedAttribute, TensorDistributedAttribute

paddle.enable_static()
paddle.seed(123)
random.seed(123)
np.random.seed(123)


def update_op_dims_mapping_by_default_dist_impl(dist_op):
    changed = False
    op_dist_attr = dist_op.dist_attr
    op_desc = dist_op.serial_op.desc
    # The following statement will be replaced by a more elegent way
    if op_desc.type() == "shape" or op_desc.type() == "slice":
        return False
    output_names = op_desc.output_names()
    xshape_arg_names = []
    if "XShape" in output_names:
        xshape_arg_names = op_desc.output("XShape")
    batch_dim_mappings = []
    for arg_name in op_desc.input_arg_names():
        serial_tensor = dist_op.get_serial_input(arg_name)
        if serial_tensor.is_parameter:
            continue
        dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
        if len(dims_mapping) > 1:
            for idx, mapping in enumerate(dims_mapping[1:]):
                assert mapping == -1, \
                    "{} only the batch dimension (0-dim) can be sharded, but the dimension {} is sharded by {} part."\
                        .format(op_desc.type(), idx, mapping)
        batch_dim_mappings.append(dims_mapping[0])
    for arg_name in op_desc.output_arg_names():
        serial_tensor = dist_op.get_serial_output(arg_name)
        if serial_tensor.is_parameter:
            continue
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        if arg_name not in xshape_arg_names:
            if len(dims_mapping) > 1:
                for idx, mapping in enumerate(dims_mapping[1:]):
                    assert mapping == -1, \
                        "{} only the batch dimension (0-dim) can be sharded, but the dimension {} is sharded by {} part."\
                            .format(op_desc.type(), idx, mapping)
            batch_dim_mappings.append(dims_mapping[0])
        else:
            assert dims_mapping[0] == -1, \
                "{} only the batch dimension (1-dim) of XShape can be sharded, but the dimension 0 is sharded by {} part."\
                    .format(op_desc.type(), mapping)
            if len(dims_mapping) > 2:
                for idx, mapping in enumerate(dims_mapping[2:]):
                    assert mapping == -1, \
                        "{} only the batch dimension (1-dim) of XShape can be sharded, but the dimension {} is sharded by {} part."\
                            .format(op_desc.type(), idx, mapping)
            batch_dim_mappings.append(dims_mapping[1])

    compatible_dim_mapping = compute_compatible_dim_mapping(batch_dim_mappings)
    assert compatible_dim_mapping is not None, "There is no compatible dim mapping."
    for arg_name in op_desc.input_arg_names():
        serial_tensor = dist_op.get_serial_input(arg_name)
        if serial_tensor.is_parameter:
            continue
        dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
        if compatible_dim_mapping != dims_mapping[0]:
            dims_mapping[0] = compatible_dim_mapping
            changed = True
    for arg_name in op_desc.output_arg_names():
        serial_tensor = dist_op.get_serial_output(arg_name)
        if serial_tensor.is_parameter:
            continue
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        if arg_name not in xshape_arg_names:
            if compatible_dim_mapping != dims_mapping[0]:
                dims_mapping[0] = compatible_dim_mapping
                changed = True
        else:
            if compatible_dim_mapping != dims_mapping[1]:
                dims_mapping[1] = compatible_dim_mapping
                changed = True

    return changed


def update_op_dims_mapping_by_elementwise_like_dist_impl(dist_op):
    """Element-wise operator can be sharded in any way (but should take care of broadcasting)."""
    changed = False
    op_dist_attr = dist_op.dist_attr
    op_desc = dist_op.serial_op.desc
    input_arg_names = op_desc.input_arg_names()
    input_dims_mapping_dict = {}
    input_dims_mapping_lens = {}
    max_dims_mapping_len = -1
    for arg_name in input_arg_names:
        dims_mapping = op_dist_attr.get_input_dims_mapping(arg_name)
        if max_dims_mapping_len < len(dims_mapping):
            max_dims_mapping_len = len(dims_mapping)
        input_dims_mapping_dict[arg_name] = dims_mapping
        input_dims_mapping_lens[arg_name] = len(dims_mapping)

    dims_mapping_list = []
    for arg_name in input_arg_names:
        if input_dims_mapping_lens[arg_name] < max_dims_mapping_len:
            new_dims_mapping = [-1 for _ in range(max_dims_mapping_len)]
            for i in range(input_dims_mapping_lens[arg_name]):
                new_idx = (max_dims_mapping_len -
                           input_dims_mapping_lens[arg_name]) + i
                new_dims_mapping[new_idx] = input_dims_mapping_dict[arg_name][i]
            dims_mapping_list.append(new_dims_mapping)
        else:
            dims_mapping_list.append(input_dims_mapping_dict[arg_name])
    output_arg_names = op_desc.output_arg_names()
    for arg_name in output_arg_names:
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        assert len(dims_mapping) == max_dims_mapping_len
        dims_mapping_list.append(dims_mapping)

    compatible_dims_mapping = compute_compatible_dims_mapping(dims_mapping_list)
    assert compatible_dims_mapping is not None, "There is no compatible dim mapping."

    for arg_name in input_arg_names:
        if input_dims_mapping_lens[arg_name] < max_dims_mapping_len:
            new_dims_mapping = [
                -1 for _ in range(input_dims_mapping_lens[arg_name])
            ]
            for i in range(input_dims_mapping_lens[arg_name]):
                new_idx = (max_dims_mapping_len -
                           input_dims_mapping_lens[arg_name]) + i
                new_dims_mapping[i] = compatible_dims_mapping[new_idx]
            if new_dims_mapping != input_dims_mapping_dict[arg_name]:
                op_dist_attr.set_input_dims_mapping(arg_name, new_dims_mapping)
                changed = True
        else:
            if compatible_dims_mapping != input_dims_mapping_dict[arg_name]:
                op_dist_attr.set_input_dims_mapping(arg_name,
                                                    compatible_dims_mapping)
                changed = True

    for arg_name in output_arg_names:
        dims_mapping = op_dist_attr.get_output_dims_mapping(arg_name)
        if compatible_dims_mapping != dims_mapping:
            op_dist_attr.set_output_dims_mapping(arg_name,
                                                 compatible_dims_mapping)
            changed = True

    return changed


def print_valid_all_op_dist_attr(valid_op_dist_attr_dict):
    for key in valid_op_dist_attr_dict.keys():
        print("op id:", key)
        for item in valid_op_dist_attr_dict[key][0]:
            print("op dist attr: ", item)
        print("")


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


def check_dims_mapping(process_mesh_topology, tensor_shape, dims_mapping):
    valid = True
    assert len(tensor_shape) == len(dims_mapping)
    for idx, item in enumerate(dims_mapping):
        if item != -1:
            if tensor_shape[idx] % process_mesh_topology[
                    item] != 0 or dims_mapping.count(item) > 1:
                valid = False
        if item != -1 and process_mesh_topology[0] == 1:
            valid = False
    return valid


def check_op_dims_mapping(op, op_dist_attr, vars):
    """Check the op dims_mapping whether valid"""
    process_mesh = op_dist_attr.process_mesh
    assert process_mesh is not None, "The process mesh should not be None."
    for var_name in op.input_arg_names:
        dims_mapping = op_dist_attr.get_input_dims_mapping(var_name)
        if not check_dims_mapping(process_mesh.topology, vars[var_name].shape,
                                  dims_mapping):
            return False
        if vars[var_name].is_data and len(dims_mapping) > 1:
            for dim in dims_mapping[1:]:
                if dim != -1:
                    return False

    for var_name in op.output_arg_names:
        dims_mapping = op_dist_attr.get_output_dims_mapping(var_name)
        if not check_dims_mapping(process_mesh.topology, vars[var_name].shape,
                                  dims_mapping):
            return False

    return True


def enumerate_op_valid_dist_attr(program, op, process_mesh):
    """enumerate the valid dist attr of one op based on the given process mesh."""
    vars = program.global_block().vars
    dims_mapping_dict = OrderedDict()
    valid_op_dist_attr_list = []
    elementwise_op_dist_attr_list = []
    default_op_dist_attr_list = []
    dist_op_impl_container = get_distributed_operator_impl_container(op.type)

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
        product(* [dims_mapping_dict[key] for key in dims_mapping_dict.keys()]))
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
                    "The {varname} is not input or output of op {op}.".format(
                        varname='var_names[idx]', op='op'))

        dist_op = DistributedOperator(op, op_dist_attr)
        if dist_op_impl_container is None:
            # if a op has no distributed implement, it has elementwise or default implement.
            if is_elementwise_like_op(op.type):
                changed = True
                valid = True
                while changed:
                    # use try...except due to the called api will raise exception.
                    try:
                        changed = update_op_dims_mapping_by_elementwise_like_dist_impl(
                            dist_op)
                    except Exception as e:
                        valid = False
                        break
                if valid:
                    # to ensure the op dist attr different and valid
                    if check_op_dims_mapping(op, dist_op.dist_attr, vars):
                        _ = []
                        for var_name in op.input_arg_names:
                            _.append(
                                dist_op.dist_attr.get_input_dims_mapping(
                                    var_name))
                        for var_name in op.output_arg_names:
                            _.append(
                                dist_op.dist_attr.get_output_dims_mapping(
                                    var_name))
                        if _ not in elementwise_op_dist_attr_list and \
                        check_special_op_dims_mapping(op, dist_op.dist_attr, vars):
                            dist_op.dist_attr.impl_idx = -1
                            valid_op_dist_attr_list.append(dist_op.dist_attr)
                            elementwise_op_dist_attr_list.append(_)
                continue
            else:
                changed = True
                valid = True
                while changed:
                    try:
                        changed = update_op_dims_mapping_by_default_dist_impl(
                            dist_op)
                    except Exception as e:
                        valid = False
                        break
                if valid:
                    if check_op_dims_mapping(op, dist_op.dist_attr, vars):
                        _ = []
                        for var_name in op.input_arg_names:
                            _.append(
                                dist_op.dist_attr.get_input_dims_mapping(
                                    var_name))
                        for var_name in op.output_arg_names:
                            _.append(
                                dist_op.dist_attr.get_output_dims_mapping(
                                    var_name))
                        if _ not in default_op_dist_attr_list and \
                        check_special_op_dims_mapping(op, dist_op.dist_attr, vars):
                            dist_op.dist_attr.impl_idx = -2
                            valid_op_dist_attr_list.append(dist_op.dist_attr)
                            default_op_dist_attr_list.append(_)
                continue

        # if op has distributed implements, find all valid dist attr of this op
        impls = dist_op_impl_container.get_impls()
        for idx, impl in enumerate(impls):
            if impl.is_auto_compatible(dist_op):
                if check_op_dims_mapping(op, dist_op.dist_attr, vars):
                    dist_op.dist_attr.impl_idx = idx
                    valid_op_dist_attr_list.append(dist_op.dist_attr)

    # set default op dist attr
    if not valid_op_dist_attr_list:
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
        valid_op_dist_attr_list.append(dist_op.dist_attr)
    return valid_op_dist_attr_list


def check_special_op_dims_mapping(op, op_dist_attr, vars):
    if op.type == "layer_norm":
        bias_dims_mapping = op_dist_attr.get_input_dims_mapping(
            op.input("Bias")[0])
        scale_dims_mapping = op_dist_attr.get_input_dims_mapping(
            op.input("Scale")[0])
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(op.input("X")[0])
        mean_dims_mapping = op_dist_attr.get_output_dims_mapping(
            op.output("Mean")[0])
        variance_dims_mapping = op_dist_attr.get_output_dims_mapping(
            op.output("Variance")[0])
        y_dims_mapping = op_dist_attr.get_output_dims_mapping(op.output("Y")[0])
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


def check_tensors_dist_attr(dist_context):
    dims_mapping_list = []
    for dist_tensor in dist_context._dist_tensors_for_program.values():
        dist_attr = dist_tensor.dist_attr
        dims_mapping = dist_attr.dims_mapping
        dims_mapping_list.append(dims_mapping)
    for dims_mapping_item in dims_mapping_list:
        for i in dims_mapping_item:
            if is_dim_shard(i):
                return True
    return False


def check_ops_valid_dist_attr(program, op_valid_dist_attr_dict):
    pass


no_need_enum_ops = ["create_py_reader", "create_double_buffer_reader", "read"]


def enumerate_ops_valid_dist_attr(program,
                                  process_mesh_topology,
                                  pipeline_mode=False):
    global no_need_enum_ops
    special_vars = [
        "lod_tensor_blocking_queue_0", "create_py_reader_0", "double_buffer_0"
    ]
    op_valid_dist_attr_dict = OrderedDict()
    ops = program.global_block().ops
    vars = program.global_block().vars
    processes = reduce(lambda x, y: x * y, process_mesh_topology)
    process_mesh_global_group = [i for i in range(processes)]
    global_process_mesh = None
    pipeline_process_mesh_list = None

    if pipeline_mode:
        pipeline_stages = process_mesh_topology[-1]
        op_count_per_stage = len(ops) // pipeline_stages

        if len(process_mesh_topology) > 1:
            process_mesh_shape = process_mesh_topology[:-1]
            per_process_mesh_group = processes // pipeline_stages
            pipeline_process_mesh_list = [auto.ProcessMesh(mesh=np.array(process_mesh_global_group[i*per_process_mesh_group: \
            (i+1)*per_process_mesh_group]).reshape(process_mesh_shape).tolist()) for i in range(pipeline_stages)]
        elif len(process_mesh_topology) == 1:

            pipeline_process_mesh_list = [
                auto.ProcessMesh(mesh=[i]) for i in range(pipeline_stages)
            ]
    else:
        if len(process_mesh_topology) > 1:
            global_process_mesh = auto.ProcessMesh(
                mesh=np.array(process_mesh_global_group).reshape(
                    process_mesh_topology).tolist())
        else:
            global_process_mesh = auto.ProcessMesh(
                mesh=process_mesh_global_group)

    for idx, op in enumerate(ops):
        op_process_mesh = global_process_mesh
        pipeline_stage = -1
        if pipeline_process_mesh_list is not None:
            pipeline_stage = idx // op_count_per_stage if idx // op_count_per_stage < len(
                pipeline_process_mesh_list) else idx // op_count_per_stage - 1
            if pipeline_stage >= len(pipeline_process_mesh_list):
                pipeline_stage = len(pipeline_process_mesh_list) - 1
            op_process_mesh = pipeline_process_mesh_list[pipeline_stage]

        if op.type in no_need_enum_ops:
            op_dist_attr = OperatorDistributedAttribute()
            op_dist_attr.process_mesh = op_process_mesh
            for var_name in op.input_arg_names:
                if var_name in special_vars:
                    op_dist_attr.set_input_dims_mapping(var_name, [])
                else:
                    dims_mapping = [-1 for i in vars[var_name].shape]
                    op_dist_attr.set_input_dims_mapping(var_name, dims_mapping)

            for var_name in op.output_arg_names:
                if var_name in special_vars:
                    op_dist_attr.set_output_dims_mapping(var_name, [])
                else:
                    dims_mapping = [-1 for i in vars[var_name].shape]
                    op_dist_attr.set_output_dims_mapping(var_name, dims_mapping)
            op_valid_dist_attr_list = [op_dist_attr]
            pipeline_stage = 0 if pipeline_stage != -1 else pipeline_stage

        else:
            op_valid_dist_attr_list = enumerate_op_valid_dist_attr(
                program, op, op_process_mesh)
        op_valid_dist_attr_dict[op.desc.id(
        )] = [op_valid_dist_attr_list, pipeline_stage]
    check_ops_valid_dist_attr(program, op_valid_dist_attr_dict)
    return op_valid_dist_attr_dict, pipeline_process_mesh_list, global_process_mesh


def mcmc_search_strategy(program,
                         op_valid_dist_attr_dict,
                         dist_context,
                         pipeline_process_mesh_list=None):
    """search a distributed training strategy after init"""
    # randomly select one op and op dist attr corresponding to the selected op
    raw_ops = program.global_block().ops
    ops = []
    global no_need_enum_ops
    for op in raw_ops:
        if op.type not in no_need_enum_ops:
            ops.append(op)
    assert ops, "The ops of program have no distributed attributes."
    vars = program.global_block().vars
    new_dist_context = copy.deepcopy(dist_context)
    new_dist_context._dist_op_context = DistributedOperatorContext()
    new_op_valid_dist_attr_dict = None
    random_selected_op_idx = np.random.randint(len(ops))
    selected_op = ops[random_selected_op_idx]
    op_valid_dist_attr_list = op_valid_dist_attr_dict[selected_op.desc.id()][0]
    pipeline_stage = op_valid_dist_attr_dict[selected_op.desc.id()][1]
    random_selected_dist_attr_idx = np.random.randint(
        len(op_valid_dist_attr_list))
    selected_op_dist_attr = copy.deepcopy(op_valid_dist_attr_list[
        random_selected_dist_attr_idx])

    start_idx = ops[0].desc.id()
    if pipeline_stage > -1:
        # in pipeline mode, the above phase just select a dims mapping
        # 0 represents not changed, 1 represents to be the same with before stage, 2 represents to be the same with the latter stage
        new_op_valid_dist_attr_dict = copy.deepcopy(op_valid_dist_attr_dict)
        changed_mode = np.random.randint(3)
        if changed_mode == 0:
            # not change the process mesh, just change dims mapping
            new_dist_context.set_op_dist_attr_for_program(selected_op,
                                                          selected_op_dist_attr)
            for var_name in selected_op.output_arg_names:
                process_mesh = selected_op_dist_attr.process_mesh
                tensor_dist_attr = TensorDistributedAttribute()
                tensor_dist_attr.process_mesh = process_mesh
                tensor_dist_attr.dims_mapping = selected_op_dist_attr.get_output_dims_mapping(
                    var_name)
                new_dist_context.set_tensor_dist_attr_for_program(
                    vars[var_name], tensor_dist_attr)

            for var_name in selected_op.input_arg_names:
                if vars[var_name].is_parameter or vars[var_name].is_data:
                    process_mesh = selected_op_dist_attr.process_mesh
                    tensor_dist_attr = TensorDistributedAttribute()
                    tensor_dist_attr.process_mesh = process_mesh
                    tensor_dist_attr.dims_mapping = selected_op_dist_attr.get_input_dims_mapping(
                        var_name)
                    new_dist_context.set_tensor_dist_attr_for_program(
                        vars[var_name], tensor_dist_attr)

        elif changed_mode == 1:
            changed_stage = pipeline_stage - 1
            if changed_stage == -1 or random_selected_op_idx == len(ops) - 1 or \
            (random_selected_op_idx + 1 == len(ops) - 1 and new_op_valid_dist_attr_dict[ops[random_selected_op_idx + 1].desc.id()][1] == pipeline_stage + 1 ):
                new_dist_context.set_op_dist_attr_for_program(
                    selected_op, selected_op_dist_attr)

                for var_name in selected_op.output_arg_names:
                    process_mesh = selected_op_dist_attr.process_mesh
                    tensor_dist_attr = TensorDistributedAttribute()
                    tensor_dist_attr.process_mesh = process_mesh
                    tensor_dist_attr.dims_mapping = selected_op_dist_attr.get_output_dims_mapping(
                        var_name)
                    new_dist_context.set_tensor_dist_attr_for_program(
                        vars[var_name], tensor_dist_attr)

                for var_name in selected_op.input_arg_names:
                    if vars[var_name].is_parameter or vars[var_name].is_data:
                        process_mesh = selected_op_dist_attr.process_mesh
                        tensor_dist_attr = TensorDistributedAttribute()
                        tensor_dist_attr.process_mesh = process_mesh
                        tensor_dist_attr.dims_mapping = selected_op_dist_attr.get_input_dims_mapping(
                            var_name)
                        new_dist_context.set_tensor_dist_attr_for_program(
                            vars[var_name], tensor_dist_attr)

            else:
                selected_op_process_mesh = pipeline_process_mesh_list[
                    pipeline_stage]
                next_op_id = ops[random_selected_op_idx + 1].desc.id()
                if new_op_valid_dist_attr_dict[next_op_id][
                        1] == pipeline_stage + 1 and random_selected_op_idx + 1 != len(
                            ops) - 1:
                    new_op_valid_dist_attr_dict[next_op_id][1] = pipeline_stage
                    for op_dist_attr in new_op_valid_dist_attr_dict[next_op_id][
                            0]:
                        op_dist_attr.process_mesh = selected_op_process_mesh
                    # set next op dist attr in the discontext and output/input tensor process mesh
                    new_dist_context.get_op_dist_attr_for_program(
                        ops[random_selected_op_idx +
                            1]).process_mesh = selected_op_process_mesh
                    for var_name in ops[random_selected_op_idx +
                                        1].output_arg_names:
                        new_dist_context.get_tensor_dist_attr_for_program(vars[
                            var_name]).process_mesh = selected_op_process_mesh
                    for var_name in ops[random_selected_op_idx +
                                        1].input_arg_names:
                        if vars[var_name].is_parameter or vars[
                                var_name].is_data:
                            new_dist_context.get_tensor_dist_attr_for_program(
                                vars[var_name]
                            ).process_mesh = selected_op_process_mesh

                # change the selected op stage and output dist attr
                new_op_valid_dist_attr_dict[selected_op.desc.id()][
                    1] = changed_stage
                new_process_mesh = pipeline_process_mesh_list[changed_stage]
                selected_op_dist_attr.process_mesh = new_process_mesh
                for op_dist_attr in new_op_valid_dist_attr_dict[
                        selected_op.desc.id()][0]:
                    op_dist_attr.process_mesh = new_process_mesh
                new_dist_context.set_op_dist_attr_for_program(
                    selected_op, selected_op_dist_attr)
                for var_name in selected_op.output_arg_names:
                    process_mesh = selected_op_dist_attr.process_mesh
                    tensor_dist_attr = TensorDistributedAttribute()
                    tensor_dist_attr.process_mesh = process_mesh
                    tensor_dist_attr.dims_mapping = selected_op_dist_attr.get_output_dims_mapping(
                        var_name)
                    new_dist_context.set_tensor_dist_attr_for_program(
                        vars[var_name], tensor_dist_attr)

                for var_name in selected_op.input_arg_names:
                    if vars[var_name].is_parameter or vars[var_name].is_data:
                        process_mesh = selected_op_dist_attr.process_mesh
                        tensor_dist_attr = TensorDistributedAttribute()
                        tensor_dist_attr.process_mesh = process_mesh
                        tensor_dist_attr.dims_mapping = selected_op_dist_attr.get_input_dims_mapping(
                            var_name)
                        new_dist_context.set_tensor_dist_attr_for_program(
                            vars[var_name], tensor_dist_attr)

                # change the pre op stage
                for idx in range(random_selected_op_idx - 1, -1, -1):
                    stage = new_op_valid_dist_attr_dict[ops[idx].desc.id()][1]
                    valid_dist_attr_list = new_op_valid_dist_attr_dict[ops[
                        idx].desc.id()][0]
                    new_process_mesh = pipeline_process_mesh_list[changed_stage]
                    if stage == changed_stage + 1:
                        new_op_valid_dist_attr_dict[ops[idx].desc.id()][
                            1] = changed_stage
                        for op_dist_attr in valid_dist_attr_list:
                            op_dist_attr.process_mesh = new_process_mesh
                        new_dist_context.get_op_dist_attr_for_program(ops[
                            idx]).process_mesh = new_process_mesh
                        # change the output and input tensor process mesh
                        for var_name in ops[idx].output_arg_names:
                            new_dist_context.get_tensor_dist_attr_for_program(
                                vars[var_name]).process_mesh = new_process_mesh
                        for var_name in ops[idx].input_arg_names:
                            if vars[var_name].is_parameter or vars[
                                    var_name].is_data:
                                new_dist_context.get_tensor_dist_attr_for_program(
                                    vars[var_name]
                                ).process_mesh = new_process_mesh
                    else:
                        break
        else:
            changed_stage = pipeline_stage + 1
            # pre_op_id = ops[random_selected_op_idx - 1].desc.id()
            if changed_stage == len(
                    pipeline_process_mesh_list) or random_selected_op_idx == 0 or \
                    (new_op_valid_dist_attr_dict[ops[random_selected_op_idx - 1].desc.id()][1] == pipeline_stage - 1 and (random_selected_op_idx == 1)):
                new_dist_context.set_op_dist_attr_for_program(
                    selected_op, selected_op_dist_attr)
                for var_name in selected_op.output_arg_names:
                    process_mesh = selected_op_dist_attr.process_mesh
                    tensor_dist_attr = TensorDistributedAttribute()
                    tensor_dist_attr.process_mesh = process_mesh
                    tensor_dist_attr.dims_mapping = selected_op_dist_attr.get_output_dims_mapping(
                        var_name)
                    new_dist_context.set_tensor_dist_attr_for_program(
                        vars[var_name], tensor_dist_attr)
                for var_name in selected_op.input_arg_names:
                    if vars[var_name].is_parameter or vars[var_name].is_data:
                        process_mesh = selected_op_dist_attr.process_mesh
                        tensor_dist_attr = TensorDistributedAttribute()
                        tensor_dist_attr.process_mesh = process_mesh
                        tensor_dist_attr.dims_mapping = selected_op_dist_attr.get_input_dims_mapping(
                            var_name)
                        new_dist_context.set_tensor_dist_attr_for_program(
                            vars[var_name], tensor_dist_attr)
            else:
                selected_op_process_mesh = pipeline_process_mesh_list[
                    pipeline_stage]
                pre_op_id = ops[random_selected_op_idx - 1].desc.id()
                if new_op_valid_dist_attr_dict[pre_op_id][
                        1] == pipeline_stage - 1 and random_selected_op_idx != 1:
                    new_op_valid_dist_attr_dict[pre_op_id][1] = pipeline_stage
                    for op_dist_attr in new_op_valid_dist_attr_dict[pre_op_id][
                            0]:
                        op_dist_attr.process_mesh = selected_op_process_mesh
                    # set pre op dist attr in the discontext and output tensor process mesh
                    new_dist_context.get_op_dist_attr_for_program(
                        ops[random_selected_op_idx -
                            1]).process_mesh = selected_op_process_mesh
                    for var_name in ops[random_selected_op_idx -
                                        1].output_arg_names:
                        new_dist_context.get_tensor_dist_attr_for_program(vars[
                            var_name]).process_mesh = selected_op_process_mesh
                    for var_name in ops[random_selected_op_idx -
                                        1].input_arg_names:
                        if vars[var_name].is_parameter or vars[
                                var_name].is_data:
                            new_dist_context.get_tensor_dist_attr_for_program(
                                vars[var_name]
                            ).process_mesh = selected_op_process_mesh

                # change the selected op stage and output tensor dist attr
                new_op_valid_dist_attr_dict[selected_op.desc.id()][
                    1] = changed_stage
                new_process_mesh = pipeline_process_mesh_list[changed_stage]
                selected_op_dist_attr.process_mesh = new_process_mesh
                for op_dist_attr in new_op_valid_dist_attr_dict[
                        selected_op.desc.id()][0]:
                    op_dist_attr.process_mesh = new_process_mesh
                new_dist_context.set_op_dist_attr_for_program(
                    selected_op, selected_op_dist_attr)

                for var_name in selected_op.output_arg_names:
                    process_mesh = selected_op_dist_attr.process_mesh
                    tensor_dist_attr = TensorDistributedAttribute()
                    tensor_dist_attr.process_mesh = process_mesh
                    tensor_dist_attr.dims_mapping = selected_op_dist_attr.get_output_dims_mapping(
                        var_name)
                    new_dist_context.set_tensor_dist_attr_for_program(
                        vars[var_name], tensor_dist_attr)

                for var_name in selected_op.input_arg_names:
                    if vars[var_name].is_parameter or vars[var_name].is_data:
                        process_mesh = selected_op_dist_attr.process_mesh
                        tensor_dist_attr = TensorDistributedAttribute()
                        tensor_dist_attr.process_mesh = process_mesh
                        tensor_dist_attr.dims_mapping = selected_op_dist_attr.get_input_dims_mapping(
                            var_name)
                        new_dist_context.set_tensor_dist_attr_for_program(
                            vars[var_name], tensor_dist_attr)
                # change the next op stage
                for idx in range(random_selected_op_idx + 1, len(ops)):
                    stage = new_op_valid_dist_attr_dict[ops[idx].desc.id()][1]
                    valid_dist_attr_list = new_op_valid_dist_attr_dict[ops[
                        idx].desc.id()][0]
                    new_process_mesh = pipeline_process_mesh_list[changed_stage]
                    if stage == changed_stage - 1:
                        new_op_valid_dist_attr_dict[ops[idx].desc.id()][
                            1] = changed_stage
                        for op_dist_attr in valid_dist_attr_list:
                            op_dist_attr.process_mesh = new_process_mesh

                        new_dist_context.get_op_dist_attr_for_program(ops[
                            idx]).process_mesh = new_process_mesh
                        # change the output tensor dist attr
                        for var_name in ops[idx].output_arg_names:
                            new_dist_context.get_tensor_dist_attr_for_program(
                                vars[var_name]).process_mesh = new_process_mesh
                        for var_name in ops[idx].input_arg_names:
                            if vars[var_name].is_parameter or vars[
                                    var_name].is_data:
                                new_dist_context.get_tensor_dist_attr_for_program(
                                    vars[var_name]
                                ).process_mesh = new_process_mesh
                    else:
                        break
    else:
        new_dist_context.set_op_dist_attr_for_program(selected_op,
                                                      selected_op_dist_attr)
        for var_name in selected_op.output_arg_names:
            process_mesh = selected_op_dist_attr.process_mesh
            tensor_dist_attr = TensorDistributedAttribute()
            tensor_dist_attr.process_mesh = process_mesh
            tensor_dist_attr.dims_mapping = selected_op_dist_attr.get_output_dims_mapping(
                var_name)
            new_dist_context.set_tensor_dist_attr_for_program(vars[var_name],
                                                              tensor_dist_attr)

        for var_name in selected_op.input_arg_names:
            if vars[var_name].is_parameter or vars[var_name].is_data:
                process_mesh = selected_op_dist_attr.process_mesh
                tensor_dist_attr = TensorDistributedAttribute()
                tensor_dist_attr.process_mesh = process_mesh
                tensor_dist_attr.dims_mapping = selected_op_dist_attr.get_input_dims_mapping(
                    var_name)
                new_dist_context.set_tensor_dist_attr_for_program(
                    vars[var_name], tensor_dist_attr)

    # make softmax_with_cross_entropy unshard
    for op in ops:
        if op.type == "softmax_with_cross_entropy":
            for var_name in op.input_arg_names:
                dims_mapping = new_dist_context.get_op_dist_attr_for_program(
                    op).get_input_dims_mapping(var_name)
                if dims_mapping != new_dist_context.get_tensor_dist_attr_for_program(
                        vars[var_name]).dims_mapping:
                    # find the op whose output is var_name
                    has_changed = False
                    for search_op in ops:
                        if var_name in search_op.output_arg_names:
                            op_dist_attr_list = op_valid_dist_attr_dict[
                                search_op.desc.id()][0]
                            for op_dist_attr in op_dist_attr_list:
                                if op_dist_attr.get_output_dims_mapping(
                                        var_name) == dims_mapping:
                                    new_dist_context.set_op_dist_attr_for_program(
                                        search_op, op_dist_attr)
                                    # change tensor dist attr
                                    tensor_dist_attr = TensorDistributedAttribute(
                                    )
                                    tensor_dist_attr.process_mesh = op_dist_attr.process_mesh
                                    tensor_dist_attr.dims_mapping = op_dist_attr.get_output_dims_mapping(
                                        var_name)
                                    new_dist_context.set_tensor_dist_attr_for_program(
                                        vars[var_name], tensor_dist_attr)
                                    has_changed = True
                                    break
                        if has_changed:
                            break
                    if not has_changed:
                        raise ValueError(
                            f"change softmax_with_cross_entropy dist attr failed"
                        )
            break
    if new_op_valid_dist_attr_dict is None:
        return op_valid_dist_attr_dict, new_dist_context
    else:
        return new_op_valid_dist_attr_dict, new_dist_context


def mcmc(train_program,
         start_program,
         op_valid_dist_attr_dict,
         init_dist_context,
         loss,
         optimizer,
         cluster=None,
         max_search_times=1,
         pipeline_process_mesh_list=None):
    times = 0
    best_dist_context = init_dist_context
    cost = np.random.randint(100000)
    min_cost = cost
    while times < max_search_times:
        times += 1
        new_dist_context = mcmc_search_strategy(
            train_program, op_valid_dist_attr_dict, best_dist_context,
            pipeline_process_mesh_list)[1]
        cur_cost = np.random.randint(100000)
        print("cur_cost: ", cur_cost, "min_cost: ", min_cost)
        if (min_cost - cur_cost) > 0:
            best_dist_context = copy.deepcopy(new_dist_context)
            min_cost = cur_cost
            times = 0
    return best_dist_context, min_cost


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
        copy.deepcopy(optimizer), dist_params_grads, dist_main_program,
        dist_startup_program)
    set_grad_var_shape(dist_main_program, dist_context)
    make_data_unshard(dist_main_program, dist_startup_program, dist_context)
    reshard(dist_main_program, dist_startup_program, rank_id, dist_context)
    HAS_SENT.clear()
    HAS_RECV.clear()
    HAS_ALLGATHER.clear()
    _g_process_group_map.clear()
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
        used_dist_context._dist_op_context = DistributedOperatorContext()
        dist_main_program, dist_startup_program = get_distributed_program(
            train_program, startup_program, used_dist_context, loss, rank_id,
            optimizer)
        all_dist_main_program.append(dist_main_program)
    return all_dist_main_program


def get_standalone_cost_data(distributed_programs):
    cost_model = CostModel()
    cost_model.static_cost_data()
    DEFAULT_MULTIPLE = 2
    OP_NAME_MAPPING = {
        "c_embedding": "embedding",
        "matmul_v2": "matmul",
        "transpose2": "transpose",
        "reshape2": "reshape",
        "unsqueeze2": "unsqueeze",
        "reduce_sum": "sum",
        "elementwise_div": "divide"
    }

    standalone_cost_data = []
    global no_need_enum_ops
    for distributed_program in distributed_programs:
        cost_data = {}
        vars = distributed_program.global_block().vars
        for op in distributed_program.global_block().ops:
            runtime = 0
            if op.type in no_need_enum_ops:
                cost_data[op.desc.id()] = runtime
                continue
            dtype = str(vars[op.input_arg_names[0]]
                        .dtype) if op.input_arg_names else "float32"
            if int(op.attr('op_role')) == int(OpRole.Backward):
                if "_grad" in op.type:
                    forward_op_name = op.type[:-5]
                    if forward_op_name in OP_NAME_MAPPING.keys():
                        forward_op_name = OP_NAME_MAPPING[forward_op_name]
                    op_cost = cost_model.get_static_op_time(
                        forward_op_name, forward=False, dtype=dtype)
                    if op_cost:
                        runtime = _compute_runtime(op_cost, op, vars)
                    else:
                        op_cost = cost_model.get_static_op_time(
                            forward_op_name, dtype=dtype)
                        if op_cost:
                            runtime = 2 * _compute_runtime(op_cost, op, vars)
            elif int(op.attr('op_role')) == int(OpRole.Forward):
                op_name = OP_NAME_MAPPING[
                    op.type] if op.type in OP_NAME_MAPPING.keys() else op.type
                op_cost = cost_model.get_static_op_time(op_name)
                if op_cost:
                    runtime = _compute_runtime(op_cost, op, vars)

            cost_data[op.desc.id()] = runtime

        standalone_cost_data.append(cost_data)
    return standalone_cost_data


def _compute_runtime(op_cost, op, vars):
    runtime = 0
    try:
        runtime = float(op_cost["op_time"])
    except:
        return runtime
    op_config = op_cost["config"]
    total_static_input_size = 0
    total_actual_input_size = 0
    parsed_info = op_config.split("\n")
    variable = "(Variable)"
    for info in parsed_info:
        variable = "(Variable)" if "(Variable)" in info else "(list<Variable>"
        if variable in info:
            arg_name_lower = info[:info.find(variable) - 1]
            shape_left_boundary = info.find("[")
            shape_right_boundary = info.find("]")
            assert shape_left_boundary > 0 and shape_right_boundary > 0 and shape_right_boundary > shape_left_boundary, "Get shape failed."
            shape = info[shape_left_boundary + 1:shape_right_boundary].split(
                ",")
            shape = list(map(lambda x: int(x.strip()), shape))
            dtype_factor = 1
            total_static_input_size += reduce(lambda x, y: x * y, shape)
            if op.type == "c_embedding":
                arg_name_lower = "w" if arg_name_lower == "weight" else "ids"
            for arg_name in op.input_names:
                if arg_name.lower() == arg_name_lower:
                    for var_name in op.input(arg_name):
                        var = vars[var_name]
                        total_actual_input_size += reduce(lambda x, y: x * y,
                                                          var.shape)
                    break
    assert total_static_input_size > 0 and total_actual_input_size > 0, "Get input size failed."
    actual_runtime = total_actual_input_size / total_static_input_size * runtime

    return actual_runtime


def estimate_searched_strategy_cost(train_program,
                                    startup_program,
                                    dist_context,
                                    loss,
                                    optimizer,
                                    pipeline_process_mesh_list=None,
                                    cluster=None):
    cost = None
    # get all distributed programs
    all_dist_main_program = get_all_distributed_main_program(
        train_program, startup_program, dist_context, loss, optimizer)
    pipeline_config = [
        process_mesh.processes for process_mesh in pipeline_process_mesh_list
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
    standalone_cost_data = get_standalone_cost_data(all_dist_main_program)
    cost = estimate_cost(
        all_dist_main_program,
        cluster=cluster,
        pipeline_config=pipeline_config,
        standalone_cost_data=standalone_cost_data,
        batch_size=microbatch_size)

    return cost


def init(op_valid_dist_attr_dict, program, pipeline_process_mesh_list,
         global_process_mesh):
    ops = program.global_block().ops
    new_dist_context = DistributedContext()
    vars = program.global_block().vars
    for op in ops:
        op_valid_dist_attr_list = op_valid_dist_attr_dict[op.desc.id()][0]
        random_op_dist_attr = np.random.randint(len(op_valid_dist_attr_list))
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
            new_dist_context.set_tensor_dist_attr_for_program(vars[var_name],
                                                              tensor_dist_attr)
        # make softmax_with_cross_entropy unshard
        if op.type == "softmax_with_cross_entropy":
            for var_name in op.input_arg_names:
                dims_mapping = new_dist_context.get_op_dist_attr_for_program(
                    op).get_input_dims_mapping(var_name)
                if dims_mapping != new_dist_context.get_tensor_dist_attr_for_program(
                        vars[var_name]).dims_mapping:
                    # find the op whose output is var_name
                    has_changed = False
                    for search_op in ops:
                        if var_name in search_op.output_arg_names:
                            op_dist_attr_list = op_valid_dist_attr_dict[
                                search_op.desc.id()][0]
                            for op_dist_attr in op_dist_attr_list:
                                if op_dist_attr.get_output_dims_mapping(
                                        var_name) == dims_mapping:
                                    new_dist_context.set_op_dist_attr_for_program(
                                        search_op, op_dist_attr)
                                    # change tensor dist attr
                                    tensor_dist_attr = TensorDistributedAttribute(
                                    )
                                    tensor_dist_attr.process_mesh = op_dist_attr.process_mesh
                                    tensor_dist_attr.dims_mapping = op_dist_attr.get_output_dims_mapping(
                                        var_name)
                                    new_dist_context.set_tensor_dist_attr_for_program(
                                        vars[var_name], tensor_dist_attr)
                                    has_changed = True
                                    break
                        if has_changed:
                            break
                    if not has_changed:
                        raise ValueError(
                            f"change softmax_with_cross_entropy dist attr failed"
                        )

    # add process meshes
    if global_process_mesh is not None:
        new_dist_context.add_process_mesh(global_process_mesh)
    elif pipeline_process_mesh_list is not None:
        for process_mesh in pipeline_process_mesh_list:
            new_dist_context.add_process_mesh(process_mesh)
    print("=====init end=====")
    return new_dist_context


def auto_search(serial_main_program,
                serial_startup_program,
                loss,
                optimizer,
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
    designated_process_mesh = None
    if len(process_meshes) > 1:
        for process_mesh in process_meshes:
            if len(process_mesh) == 1:
                designated_process_mesh = process_mesh
    else:
        designated_process_mesh = process_meshes[0]

    op_valid_dist_attr_dict, pipeline_process_mesh_list, global_process_mesh = enumerate_ops_valid_dist_attr(serial_main_program, designated_process_mesh, False)
    init_dist_context = init(op_valid_dist_attr_dict, serial_main_program, pipeline_process_mesh_list, global_process_mesh)
    best_dist_context, runtime = mcmc(
        serial_main_program,
        serial_startup_program,
        op_valid_dist_attr_dict,
        init_dist_context,
        loss,
        optimizer,
        pipeline_process_mesh_list=pipeline_process_mesh_list,
        cluster=None)
    best_dist_context._dist_op_context = DistributedOperatorContext()
    return best_dist_context, runtime
