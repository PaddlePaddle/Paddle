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
# limitations under the License

import os
import paddle
import threading
import numpy as np
import warnings
import logging

import paddle.fluid.core as core
from paddle.fluid.io import is_parameter, is_belong_to_optimizer
from paddle.framework.io import _to_LodTensor


def is_valid_list_index(list, index):
    if index >= -len(list) and index < len(list):
        return True
    else:
        return False


def is_dim_shard(mapping):
    if mapping != -1:
        return True
    else:
        return False


def is_dim_replicate(mapping):
    if mapping == -1:
        return True
    else:
        return False


def compute_compatible_dim_mapping(dim_mappings):
    if not dim_mappings:
        return None
    compatible_mapping = dim_mappings[0]
    for mapping in dim_mappings:
        if compatible_mapping == -1:
            compatible_mapping = mapping
        elif mapping == -1:
            continue
        elif compatible_mapping == mapping:
            continue
        else:
            return None
    return compatible_mapping


def compute_compatible_dims_mapping(dims_mapping_list):
    if not dims_mapping_list:
        return None
    length = len(dims_mapping_list[0])
    for dims_mapping in dims_mapping_list:
        assert dims_mapping is not None, \
            "Dims mapping must not be None for compatible computation"
        assert len(dims_mapping) == length, \
            "The length of dims_mapping in list must be same for compatible computation."
    compatible_result = []
    for dim_mappings in zip(*dims_mapping_list):
        compatible_dim_mapping = compute_compatible_dim_mapping(
            list(dim_mappings))
        if compatible_dim_mapping is None:
            return None
        compatible_result.append(compatible_dim_mapping)
    return compatible_result


def compute_compatible_process_mesh(process_mesh_list):
    compatible_process_mesh = None
    if not process_mesh_list:
        return compatible_process_mesh
    for process_mesh in process_mesh_list:
        if process_mesh is not None:
            if compatible_process_mesh is None or compatible_process_mesh == process_mesh:
                compatible_process_mesh = process_mesh
            else:
                return None
    return compatible_process_mesh


def compute_compatible_and_update_dim_mapping(dims_mapping_list, index_list):
    assert len(dims_mapping_list) == len(index_list)
    changed = False
    dim_mappings = []
    for i in range(len(dims_mapping_list)):
        assert is_valid_list_index(dims_mapping_list[i], index_list[i])
        dim_mappings.append(dims_mapping_list[i][index_list[i]])
    compatible_dim_mapping = compute_compatible_dim_mapping(dim_mappings)
    if compatible_dim_mapping is None:
        return False
    for i in range(len(dims_mapping_list)):
        if compatible_dim_mapping != dims_mapping_list[i][index_list[i]]:
            dims_mapping_list[i][index_list[i]] = compatible_dim_mapping
            changed = True
    return changed


def append_distributed_attr_suffix(name):
    """
    Append auto parallel suffix for distributed attribute name.
    """
    return name + core.kAutoParallelSuffix()


def remove_distributed_attr_suffix(name):
    """
    Remove auto parallel suffix from distributed attribute name.
    """
    return name.strip(core.kAutoParallelSuffix())


def check_distributed_attr_for_program(program, dist_context=None):
    from .dist_context import get_default_distributed_context
    if dist_context is None:
        dist_context = get_default_distributed_context()
    assert dist_context.is_initialized_for_program(), \
        "Distributed attributes must be initialized before check."
    for block in program.blocks:
        for tensor in block.vars.values():
            dist_tensor = dist_context.get_dist_tensor_for_graph(tensor)
            tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                tensor)
            if (tensor_dist_attr is not None) and (not dist_tensor.is_valid()):
                return False
        for op in block.ops:
            dist_op = dist_context.get_dist_op_for_graph(tensor)
            op_dist_attr = dist_context.get_op_dist_attr_for_program(op)
            if (op_dist_attr is not None) and (not dist_op.is_valid()):
                return False
    return True


def print_program_with_dist_attr(program, dist_context=None):
    """
    This function reuses the original program output ability with a distributed context.
    Using lock can avoid multiple threads change the default distributed context simultaneously.
    """
    lock = threading.Lock()
    lock.acquire()
    from .dist_context import get_default_distributed_context
    from .dist_context import set_default_distributed_context
    if dist_context is None:
        dist_context = get_default_distributed_context()
        print(program)
    else:
        original_default_context = get_default_distributed_context()
        set_default_distributed_context(dist_context)
        print(program)
        set_default_distributed_context(original_default_context)
    lock.release()


def _get_comm_group(processes, shape, axis, rank):
    """
    Given a rank and the processes mesh the rank belongs to,  
    compute the communication peers of the rank based on the give axis in the mesh.

    Example: 16 processes managed in a 4-Dimensinal mesh with shape of [2, 2, 2, 2].
    the rank communication peers of rank 0 (included) are following:
    in axis 0: [0, 1]
    in axis 1: [0, 2]
    in axis 2: [0, 4]
    in axis 3: [0, 8]
    """

    # NOTE _linear_idx2coordinate assume processes mesh start with 0 and continuous
    # tricks to support processes mesh when it is not start with 0 or continuous
    assert rank in processes, "rank [{}] is NOT in processes group {}".format(
        rank, processes)
    rank_relatvie = processes.index(rank)
    coordinate = _linear_idx2coordinate(shape, rank_relatvie)
    coordinates_in_group = [coordinate[:] for i in range(shape[axis])]

    # select comm group
    for i in range(shape[axis]):
        coordinates_in_group[i][axis] = i

    ranks_in_group_relative = [
        _coordinate2linear_idx(shape, coordinate)
        for coordinate in coordinates_in_group
    ]
    ranks_in_group = [processes[idx] for idx in ranks_in_group_relative]

    return sorted(ranks_in_group)


def _get_idx_in_axis(processes, shape, axis, rank):
    """
    Given a rank and the processes mesh the rank belongs to,  
    compute the index of the rank in given axis.

    Example: 27 processes managed in a 3-Dimensinal mesh with shape of [3, 3, 3].
    the index of rank 22 are:
    in axis 0: 1
    in axis 1: 1
    in axis 2: 2
    """

    # NOTE _linear_idx2coordinate assume processes mesh start with 0 and continuous
    #  tricks to support processes mesh when it is not start with 0 or continuous
    rank_relatvie = processes.index(rank)
    coordinate = _linear_idx2coordinate(shape, rank_relatvie)
    return coordinate[axis]


def _coordinate2linear_idx(mesh_shape, coordinate):
    """
    convert a coordinate in multidimensional mesh space into a scala idx in linear space.

    it use Row-major order for dimension conversion. 
    so it has:  [most_significant_dim, ..., least_significant_dim]
    assume: 

        the size of i-th dimension to be:  S[i]
        the index of j-th dimension is: I[j]

    linear_idx of a n dimensional coordinate is: 

        I[n-1] * (S[n-2] * S[n-3] * S[n-4] *     ....    S[0]) +
        I[n-2] * (         S[n-3] * S[n-4] *     ....    S[0]) +       
        I[n-3] * (                  S[n-4] *     ....    S[0]) +  
        ...
        I[1]   * (                                       S[0]) + 
        I[0]

    """
    # NOTE the following function work based on a strong an assumption
    # that the processes in mesh are
    #    1. starts from 0
    #    2. continuous
    # it will be wrong if ths above condition doesnot meet,
    # e.g. process_mesh = { process_groups = [7, 8, 9,10, 12, 13, 14, 15], mesh = [2, 4]}
    # if you want a more general mapping, you should use cartesian product

    assert len(mesh_shape) == len(
        coordinate
    ), "coordinate should have the same size as mesh shape, but got shape: {}, coordinate: {}".format(
        mesh_shape, coordinate)
    for i in range(len(mesh_shape)):
        assert coordinate[
            i] >= 0, "index in dimension [{}] is least than zero. coordinate: {}".format(
                i, coordinate)
        assert coordinate[i] < mesh_shape[
            i], "index beyond extent in dimension [{}]. shape: {}, coordinate: {}".format(
                i, mesh_shape, coordinate)

    base = mesh_shape[-1]
    linear_idx = coordinate[-1]

    # row major order
    for i in range(len(mesh_shape) - 2, -1, -1):
        linear_idx += base * coordinate[i]
        base *= mesh_shape[i]

    return linear_idx


def _linear_idx2coordinate(mesh_shape, linear_idx):
    """
    mapping a linear scala into multidimensional mesh space, return it coordinate in that space.

    it is the inverse function of _coordinate2linear_idx.
    assume: 

        the size of i-th dimension to be:  S[i]
        the index of j-th dimension is: I[j]

    the coordinate given linear_idx is:

        I[0] = linear_idx                                  % S[0]
        I[0] = (linear_idx / S[0])                         % S[1]
        I[0] = (linear_idx / (S[0] * S[1]))                % S[2]
        ....

    """

    assert linear_idx >= 0, "linear index [{}] is least than zero".format(
        linear_idx)
    assert linear_idx < np.prod(
        mesh_shape
    ), "linear index beyond the extent of mesh shape. shape: {}, linear index: {}".format(
        mesh_shape, linear_idx)

    base = 1
    coordinate = [-1] * len(mesh_shape)

    for i in reversed(range(len(mesh_shape))):
        offset = linear_idx / base
        coordinate[i] = int(offset % mesh_shape[i])
        base *= mesh_shape[i]

    # row major order
    return coordinate


def _get_corresponding_rank(dist_context, target_mesh, rank):

    # TODO(JZ-LIANG) a hack method to support varying mesh in Pipeline parallelism case.
    # we assume that all mesh are evenly divide from a parent mesh and should have same size.
    # to revise this in future.

    coordinate = None
    for mesh in dist_context.process_meshes:
        if rank in mesh.processes and mesh.topology == target_mesh.topology:
            coordinate = _linear_idx2coordinate(mesh.topology,
                                                mesh.processes.index(rank))
            break

    assert coordinate is not None, "could NOT found rank [{}] in any registered mesh".format(
        rank)
    return target_mesh.processes[_coordinate2linear_idx(mesh.topology,
                                                        coordinate)]


def _get_unshard_dist_shape(var, dist_attr):
    var_shape = var.shape
    mapping = dist_attr.dims_mapping
    mesh = dist_attr.process_mesh.topology
    assert len(var_shape) == len(
        mapping
    ), "variable shape [{}] and dim_mapping [{}] is NOT match !".format(
        var_shape, mapping)
    new_shape = []
    for idx in range(len(var_shape)):
        if var_shape[idx] == -1 or mapping[idx] == -1:
            new_shape.append(var_shape[idx])
        else:
            new_shape.append(var_shape[idx] * mesh[mapping[idx]])

    return new_shape


def make_data_unshard(dist_main_prog, dist_startup_prog, dist_context=None):
    from .dist_context import get_default_distributed_context
    if dist_context is None:
        dist_context = get_default_distributed_context()

    for var in dist_main_prog.list_vars():
        if var.is_data:
            tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                var)
            inverse_shape = _get_unshard_dist_shape(var, tensor_dist_attr)
            var.desc.set_shape(inverse_shape)
            dim_mapping = tensor_dist_attr.dims_mapping
            dim_mapping = [-1] * len(dim_mapping)
            tensor_dist_attr.dims_mapping = dim_mapping
            dist_context.set_tensor_dist_attr_for_program(var, tensor_dist_attr)


def _check_addition_info(addition_info):
    """
    Validity check of additional information
    """
    if not addition_info:
        return addition_info
    elif not isinstance(addition_info, dict):
        raise TypeError(
            "The type of addition_info should be 'dict', but got {}".format(
                str(type(addition_info))))
    else:
        return addition_info


def _check_valid_path(file_path):
    """
    Validity check of input file path
    """
    if not file_path:
        return file_path
    elif isinstance(file_path, str):
        if not os.path.exists(file_path):
            raise ValueError("The file_path '{}' does not exist.".format(
                file_path))
        else:
            return [file_path]
    elif isinstance(file_path, list):
        if not all(isinstance(file, str) for file in file_path):
            raise ValueError("The type of each file_path should be str.")
        if not all(os.path.exists(file) for file in file_path):
            raise ValueError("The file_path's file does not exist.")
        return file_path
    else:
        raise TypeError(
            "The type of file_path should be 'str' or 'list', but got '{}'.".
            format(str(type(file_path))))


def save_distributed_checkpoint(program,
                                checkpoint_path,
                                is_integrated=False,
                                addition_info=None,
                                dist_attr_path=None):
    """ 
    Save model parameter state, optimzer state, distributed attribute and 
    additional information of each rank.

    Args:
        program(Program): The program to be saved.
        checkpoint_path(str): The path of the checkpoint file to be saved.
        is_integrated(bool, optional): Whether to integrate param before save. Default: False.
        addition_info(dict, optional): Additional information. Default: None.
        dist_attr_path(str, optional): The path of distributed attribute file to be saved. Default: None

    Returns:
        None

    Examples:
        .. code-block:: python

            ckpt_path = os.path.join(args.output_dir, "step_%d" % step)
            os.makedirs(ckpt_path, exist_ok=True)
            save_distributed_checkpoint(program, ckpt_path)
    """
    if not is_integrated:
        rank = paddle.distributed.get_rank()
        ckpt_file_name = os.path.join(checkpoint_path,
                                      "model_state_rank{}.pdmodel".format(rank))

        state_dict = {
            "model": program.state_dict(),
            "ranks": paddle.distributed.get_world_size()
        }
        if _check_addition_info(addition_info):
            state_dict["addition_info"] = addition_info

        paddle.save(state_dict, ckpt_file_name)
        logging.info("Already save model to {}".format(checkpoint_path))

        if dist_attr_path:
            raise NotImplementedError(
                "Save distributed attribute has not been implemented.")
    else:
        # TODO: integrate param before save
        raise NotImplementedError(
            "Integrating parameter has not been implemented.")


def load_distributed_checkpoint(checkpoint_path,
                                program=None,
                                dist_attr_path=None):
    """ 
    Load parameter, optimizer, distributed attribute and addition_info of model.

    Args:
        checkpoint_path(str|list[str]): checkpoint_path's type can be 'str' or 'list', \
            which must be in order of rank id when type is 'list'.
        program(Program, optional): The program to be updated with checkpoint_path. Default: None.
        dist_attr_path(str|list[str], optional): dist_attr_path's type can be 'str' or 'list', \
            which must be in order of rank id when type is 'list'. Default: None.

    Returns:
        None or addition_info which user saved in last train.

    Examples:
        .. code-block:: python

            exe.run(startup_program)
            ckpt_path = ['./output/step_10/model_state_rank0.pdmodel', 
                         './output/step_10/model_state_rank1.pdmodel']
            load_distributed_checkpoint(ckpt_path, main_program)
    """
    checkpoint_path = _check_valid_path(checkpoint_path)
    dist_attr_path = _check_valid_path(dist_attr_path)

    if checkpoint_path and dist_attr_path:
        raise NotImplementedError(
            "Merge&Slice parameter with dist_attr has not been implemented.")

    elif checkpoint_path:
        assert len(checkpoint_path) == paddle.distributed.get_world_size(), \
            "The number of checkpoint_path must equal to the number of ranks"
        rank = paddle.distributed.get_rank()
        state_dict_info = paddle.load(checkpoint_path[rank])
        state_dict = state_dict_info["model"]
    else:
        raise ValueError("'checkpoint_path' can not be None.")

    program.set_state_dict(state_dict) if program else \
        warnings.warn("'Program' is None, parameters will not be loaded.")

    if "addition_info" not in state_dict_info:
        return

    return state_dict_info["addition_info"]


# NOTE(JZ-LIANG) in auto parallel we need to distinguish the recompute ops and grad ops in 
# backward segment, so we give recompute op a new op_role which is 9 (0x0009) temporary.
OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
OpRole = core.op_proto_and_checker_maker.OpRole


def is_forward_op(op):
    ref_role1 = int(core.op_proto_and_checker_maker.OpRole.Forward)
    ref_role2 = int(core.op_proto_and_checker_maker.OpRole.Loss)
    op_role = int(op.attr('op_role'))
    return OP_ROLE_KEY in op.attr_names and (op_role == ref_role1 or
                                             op_role == ref_role2)


def is_backward_op(op):
    return OP_ROLE_KEY in op.attr_names and \
            int(op.all_attrs()[OP_ROLE_KEY]) & int(OpRole.Backward)


def is_recompute_op(op):
    return OP_ROLE_KEY in op.attr_names and \
            int(op.all_attrs()[OP_ROLE_KEY]) == 9
