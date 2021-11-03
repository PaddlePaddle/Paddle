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
from paddle.framework.io import _to_LodTensor
from paddle.fluid.io import is_parameter, is_belong_to_optimizer


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
    """ Validity check of additional information """
    if not addition_info:
        return addition_info
    elif not isinstance(addition_info, dict):
        raise TypeError(
            "The type of 'addition_info' should be 'dict', but got '{}'.".
            format(str(type(addition_info))))
    else:
        for item, value in addition_info.items():
            if item not in ["epoch", "batch", "batch_size"]:
                raise ValueError(
                    "The key of 'addition_info' should be one of the ['epoch', 'batch', 'batch_size']."
                )
            if not isinstance(value, int):
                raise ValueError(
                    "The value of 'addition_info' should be 'int', but got '{}'.".
                    format(str(type(value))))
        return addition_info


def _check_valid_path(file_path):
    """ Validity check of input file path """
    if not file_path:
        return file_path
    elif isinstance(file_path, str):
        if not os.path.exists(file_path):
            raise ValueError("The path '{}' does not exist.".format(file_path))
        else:
            return [file_path]
    elif isinstance(file_path, list):
        for file in file_path:
            if not isinstance(file, str):
                raise TypeError(
                    "The type of path should be 'str', but got '{}'.".format(
                        str(type(file))))
            if not os.path.exists(file):
                raise ValueError("The path '{}' does not exist.".format(file))
        return file_path
    else:
        raise TypeError(
            "The type of path should be 'str' or 'list', but got '{}'.".format(
                str(type(file_path))))


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
        addition_info(dict, optional): Additional information, should select in ['epoch', 'batch', 'batch_size']. Default: None.
        dist_attr_path(str, optional): The path of distributed attribute file to be saved. Default: None

    Returns:
        None

    Examples:
        .. code-block:: python

            path = os.path.join("./output", "step_%d" % step)
            os.makedirs(ckpt_path, exist_ok=True)
            add_info = {'batch': step, "batch_size": args.global_batch_size}
            save_distributed_checkpoint(program, ckpt_path, 
                                        addition_info=add_info, 
                                        dist_attr_path=path)
    """
    assert isinstance(program, paddle.fluid.framework.Program)
    addition_info = _check_addition_info(addition_info)

    if not is_integrated:
        rank = paddle.distributed.get_rank()
        ckpt_file_name = os.path.join(checkpoint_path,
                                      "model_state_rank{}.pdmodel".format(rank))

        state_dict = {
            "model": program.state_dict(),
            "world_size": paddle.distributed.get_world_size(),
            "addition_info": addition_info
        }
        paddle.save(state_dict, ckpt_file_name)
        logging.info("Already saved model to {}".format(checkpoint_path))

        if dist_attr_path:
            save_distributed_attribute(program, dist_attr_path)
    else:
        # TODO: integrate param before save
        raise NotImplementedError(
            "Integrating parameter has not been implemented.")


def load_distributed_checkpoint(checkpoint_path, dist_attr_path=None):
    """ 
    Load parameter, optimizer, distributed attribute and addition_info.

    Args:
        checkpoint_path(str|list[str]): checkpoint_path's type can be 'str' or 'list', \
            which must be in order of rank id when type is 'list'.
        dist_attr_path(str|list[str], optional): dist_attr_path's type can be 'str' or 'list', \
            which must be in order of rank id when type is 'list'. Default: None.

    Returns:
        param_dict(dict): 
        dist_attr(dict): 
        addition_info(dict): user saved in last train. 

    Examples:
        .. code-block:: python

            ckpt_path = ['./model_state_rank0.pdmodel', 
                         './model_state_rank1.pdmodel']
            dist_attr_path = ['./dist_attr_rank0.pdattr', 
                              './dist_attr_rank1.pdattr']
            param_dict, dist_attr, add_info = load_distributed_checkpoint(ckpt_path, dist_attr_path)
    """
    checkpoint_path = _check_valid_path(checkpoint_path)
    dist_attr_path = _check_valid_path(dist_attr_path)

    if checkpoint_path and dist_attr_path:
        state_dict_info = load_distributed_state_dict(checkpoint_path)
        dist_attr = load_distributed_attribute(dist_attr_path)

        param_dict = state_dict_info["model"]
        addition_info = state_dict_info["addition_info"]
        if addition_info:
            return param_dict, dist_attr, addition_info
        else:
            return param_dict, dist_attr

    elif checkpoint_path:
        assert len(checkpoint_path) == paddle.distributed.get_world_size(), \
            "If distributed attribute is not provided, the number of 'checkpoint_path' \
                must equal to the number of world_size"

        rank = paddle.distributed.get_rank()
        state_dict_info = paddle.load(checkpoint_path[rank])

        param_dict = state_dict_info["model"]
        addition_info = state_dict_info["addition_info"]
        if addition_info:
            return param_dict, addition_info
        else:
            return param_dict

    else:
        raise ValueError("'checkpoint_path' can not be None.")


def load_checkpoint_into_program(checkpoint_path,
                                 program=None,
                                 dist_attr_path=None):
    """ 
    Load parameter, optimizer, distributed attribute and addition_info into model.

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
            load_checkpoint_into_program(ckpt_path, main_program)
    """
    checkpoint_path = _check_valid_path(checkpoint_path)
    dist_attr_path = _check_valid_path(dist_attr_path)

    if checkpoint_path and dist_attr_path:
        all_state_dict_info = load_distributed_state_dict(checkpoint_path)
        all_pre_dist_attr = load_distributed_attribute(dist_attr_path)
        all_cur_dist_attr = get_dist_attr(program)

        all_param_dict = all_state_dict_info["model"]
        addition_info = all_state_dict_info["addition_info"]
        sliced_param_dict = merge_and_slice_parameter(
            all_param_dict, all_pre_dist_attr, all_cur_dist_attr)
        load_parameter_into_program(sliced_param_dict, program)

        if addition_info:
            return addition_info

    elif checkpoint_path:
        assert len(checkpoint_path) == paddle.distributed.get_world_size(), \
            "If distributed attribute is not provided, the number of 'checkpoint_path' \
                must equal to the number of world_size"

        rank = paddle.distributed.get_rank()
        state_dict_info = paddle.load(checkpoint_path[rank])

        param_dict = state_dict_info["model"]
        addition_info = state_dict_info["addition_info"]
        load_parameter_into_program(param_dict, program)

        if addition_info:
            return addition_info

    else:
        raise ValueError("'checkpoint_path' can not be None.")


def load_parameter_into_program(param_dict, program):
    """ Load parameters into program """
    assert program, "'program' can not be 'None'"
    for name, value in param_dict.items():
        assert isinstance(name, str), \
            "The type of 'param_dict' key should be 'str', but got '{}'.".format(str(type(name)))
        assert isinstance(value, paddle.fluid.LoDTensor), \
            "The type of 'param_dict' value should be 'LoDTensor', but got '{}'".format(str(type(value)))

    program.set_state_dict(param_dict)


def save_distributed_attribute(program, path):
    """ Save distributed attribute of all parameters """
    # TODO: just save a complete distributed attribute file
    rank_id = paddle.distributed.get_rank()
    dist_attr_name = os.path.join(path,
                                  "dist_attr_rank{}.pdattr".format(rank_id))

    dist_attr_dict = {
        "model": get_dist_attr(program),
        "world_size": paddle.distributed.get_world_size()
    }
    paddle.save(dist_attr_dict, dist_attr_name)
    logging.info("Already saved distributed attribute to {}".format(path))


def load_distributed_attribute(dist_attr_path):
    """ Load parameters' distributed attribute from dist_attr_path """
    total_dist_attr = {}
    for dist_attr_file in dist_attr_path:
        dist_attr = paddle.load(dist_attr_file)
        pre_world_size = dist_attr["world_size"]
        assert pre_world_size == len(dist_attr_path), \
            "The number of distributed attribute file should equal to the number of \
                the last training world size."

        for name, attr in dist_attr["model"].items():
            if name not in total_dist_attr:
                total_dist_attr[name] = attr

    return total_dist_attr


def load_distributed_state_dict(checkpoint_path):
    """ Load parameters' state_dict from checkpoint_path """
    total_state_dict = {}
    for ckpt_file in checkpoint_path:
        state_dict_info = paddle.load(ckpt_file)
        pre_world_size = state_dict_info["world_size"]
        addition_info = state_dict_info["addition_info"]
        assert pre_world_size == len(checkpoint_path), \
            "The number of checkpoint path must equal to the number of \
                the last training world size."

        for name, value in state_dict_info["model"].items():
            if name in total_state_dict:
                total_state_dict[name].append(value)
            else:
                total_state_dict[name] = [value]

    total_state_dict_info = {
        "model": total_state_dict,
        "addition_info": addition_info
    }
    return total_state_dict_info


def get_dist_attr(program, dist_context=None):
    """ Get distributed attribute of current rank """
    from .dist_context import get_default_distributed_context
    if dist_context is None:
        dist_context = get_default_distributed_context()

    dist_attr = {}
    for var in program.list_vars():
        if is_parameter(var) or is_belong_to_optimizer(var):
            tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                var)
            process_mesh = tensor_dist_attr.process_mesh
            dims_mapping = tensor_dist_attr.dims_mapping

            dist_attr[var.name] = {
                "process_shape": process_mesh.topology,
                "process_group": process_mesh.processes,
                "dims_mapping": dims_mapping
            }

    return dist_attr


def merge_and_slice_parameter(param_dict, pre_dist_attr, cur_dist_attr):
    """
    Merge parameters with previous dist_attr and slice parameters with current dist_attr

    Arags:
        param_dict(dict): parameters' value of all ranks
        pre_dist_attr(dict): parameters' dist_attr of last training process
        cur_dist_attr(dict): parameters' dist_attr of current training process

    Returns:
        sliced_param_dict(dict): sliced parameters' of current rank
    """
    param_not_in_program = []
    param_not_in_dist_attr = []
    for name in pre_dist_attr:
        if name not in cur_dist_attr:
            param_not_in_program.append(name)

    logging.info("Start to merge and slice parameters.")
    sliced_param_dict = {}
    for var_name in cur_dist_attr.keys():
        if var_name not in pre_dist_attr:
            param_not_in_dist_attr.append(var_name)
            continue

        pre_param = param_dict[var_name]
        pre_attr = pre_dist_attr[var_name]
        cur_attr = cur_dist_attr[var_name]
        cur_slice_param = _merge_and_slice(pre_param, pre_attr, cur_attr)
        sliced_param_dict[var_name] = cur_slice_param

    if param_not_in_program:
        warnings.warn(
            "Parameters '{}' are not found in current program."\
                .format(str(param_not_in_program)))
    if param_not_in_dist_attr:
        warnings.warn(
            "Parameters '{}' are not found in pre dist_attr."\
                .format(str(param_not_in_dist_attr)))

    return sliced_param_dict


def _merge_and_slice(param_list, pre_dist_attr, cur_dist_attr):
    """
    Merge a parameter with previous dist_attr and slice the parameter with current dist_attr

    Arags:
        param_list(list[LodTensor]): a parameter's value of all ranks
        pre_dist_attr(dict): a parameter's dist_attr of last training process
        cur_dist_attr(dict): a parameter's dist_attr of current training process

    Returns:
        sliced_param(LodTensor): sliced parameter of current rank
    """
    from .reshard import _compute_complete_shape, _compute_partition_index

    pre_dims_mapping = pre_dist_attr["dims_mapping"]
    pre_process_shape = pre_dist_attr["process_shape"]
    pre_process_group = pre_dist_attr["process_group"]

    cur_dims_mapping = cur_dist_attr["dims_mapping"]
    cur_process_shape = cur_dist_attr["process_shape"]
    cur_process_group = cur_dist_attr["process_group"]

    # get the complete shape of the parameter
    params = [np.array(p) for p in param_list]
    complete_shape = _compute_complete_shape(params[0].shape, pre_process_shape,
                                             pre_dims_mapping)

    # merge the parameter with previous dist_attr
    partition_param_list = []
    for process in pre_process_group:
        partition_index = _compute_partition_index(
            process, complete_shape, pre_dims_mapping, pre_process_shape,
            pre_process_group)
        index = pre_process_group.index(process)
        _merge_parameter(partition_param_list, params[index], partition_index)
    assert len(partition_param_list) == 1 or not partition_param_list, \
        "Fail to merge parameter"

    # slice the parameter with current dist_attr
    complete_param = partition_param_list[0][0]
    partition_index_list = _get_split_indices(
        complete_param.shape, cur_dims_mapping, cur_process_shape,
        cur_process_group)
    sliced_param_list = _slice_parameter(complete_param, partition_index_list,
                                         len(partition_index_list))

    # get the current parameter's index in sliced_param_list
    rank_id = paddle.distributed.get_rank()
    sliced_param_index = _get_sliced_param_index(
        rank_id, complete_param.shape, cur_dims_mapping, cur_process_shape,
        cur_process_group)

    sliced_param = _to_LodTensor(sliced_param_list[sliced_param_index])

    return sliced_param


def _merge_parameter(partition_param_list, param, partition_index):
    """
    Merge partitial parameters to a complete one.

    Returns:
        None

    Examples:
        .. code-block:: python

            import numpy as np
            partition_param_list = [(np.array([[[1.11, 1.12]]]), [[0,1],[0,1],[0,2]])]
            param = np.array([[[1.13, 1.14]]])
            partition_index = [[0,1],[0,1],[2,4]]

            _merge_parameter(partition_param_list, param, partition_index)
            # partition_param_list: [(np.array([[[1.11, 1.12, 1.13, 1.14]]]), [[0,1],[0,1],[0,4]])]
    """
    from .reshard import _compute_concat_info

    if not partition_param_list:
        partition_param_list.append((param, partition_index))
    else:
        i = 0
        has_concat = False
        while i < len(partition_param_list):
            concat_axis, first_order, new_partition = _compute_concat_info(
                partition_param_list[i][1], partition_index)
            if concat_axis != -1:
                has_concat = True
                if first_order == 0:
                    new_param = np.concatenate(
                        (partition_param_list[i][0], param), axis=concat_axis)
                else:
                    new_param = np.concatenate(
                        (param, partition_param_list[i][0]), axis=concat_axis)

                partition_param_list.pop(i)
                _merge_parameter(partition_param_list, new_param, new_partition)
                break
            i += 1

        if not has_concat:
            need_append = True
            for i in range(len(partition_param_list)):
                if partition_index == partition_param_list[i][1]:
                    need_append = False
                    break
            if need_append:
                partition_param_list.append((param, partition_index))


def _slice_parameter(complete_param, partition_index_list, length):
    """
    Slice complete parameter with current dist_attr.

    Returns:
        sliced_param_list(list)

    Examples:
        .. code-block:: python

            import numpy as np
            complete_param = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
            rank = 2
            complete_shape = [1, 1, 6]
            dims_mapping = [-1, -1, 0]
            process_shape = [3]
            process_group = [0, 1, 2]

            sliced_param_list = _slice_parameter(complete_param, [[], [], [2, 4]], 3)
            # [array([[[1.11, 1.12]]]), array([[[1.13, 1.14]]]), array([[[1.15, 1.16]]])]
    """
    sliced_param_list = []
    axis = len(complete_param.shape) - length
    sliced_param = np.split(
        complete_param, partition_index_list[axis], axis=axis)

    if length == 1:
        return sliced_param
    for param in sliced_param:
        sliced_param_list.extend(
            _slice_parameter(param, partition_index_list, length - 1))

    return sliced_param_list


def _get_sliced_param_index(rank, complete_shape, dims_mapping, process_shape,
                            process_group):
    """
    Get slice param index of complete param with current rank

    Returns:
        sliced_param_index(int): the index of sliced param in sliced_param_list

    Examples:
        .. code-block:: python

            import numpy as np
            complete_param = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
            rank = 2
            complete_shape = [1, 1, 6]
            dims_mapping = [-1, -1, 0]
            process_shape = [3]
            process_group = [0, 1, 2]

            slice_param = _slice_parameter(complete_param, [[], [], [2, 4]], 3)
            # slice_param: 
            # [array([[[1.11, 1.12]]]), array([[[1.13, 1.14]]]), array([[[1.15, 1.16]]])]

            index = _get_sliced_param_index(rank, complete_shape, dims_mapping
                                            process_shape, process_group)
            # index: 2
    """
    from .reshard import _compute_partition_index

    partition_index = _compute_partition_index(
        rank, complete_shape, dims_mapping, process_shape, process_group)

    sliced_param_index = 0
    for i, shape in enumerate(complete_shape):
        if dims_mapping[i] == -1:
            slice_shape = shape
        else:
            slice_shape = shape // process_shape[dims_mapping[i]]

        if shape == 1:
            index = 0
        else:
            index = (partition_index[i][0] + 1) // slice_shape

        sliced_param_index = sliced_param_index * (shape // slice_shape) + index

    return sliced_param_index


def _get_split_indices(complete_shape, dims_mapping, process_shape,
                       process_group):
    """
    Get split indices of every dimension

    Returns:
        split_indices_list(list): the split indices of every dimension of the parameter

    Examples:
        .. code-block:: python

            import numpy as np
            complete_param = np.array([[[1.11, 1.12, 1.13, 1.14, 1.15, 1.16]]])
            complete_shape = [1, 1, 6]
            dims_mapping = [-1, -1, 0]
            process_shape = [3]
            process_group = [0, 1, 2]

            index = _get_split_indices(complete_shape, dims_mapping, process_shape, process_group)
            # index: [[], [], [2, 4]]
    """
    from .reshard import _compute_partition_index

    split_indices_list = []
    for process in process_group:
        partition_index = _compute_partition_index(
            process, complete_shape, dims_mapping, process_shape, process_group)

        if split_indices_list:
            for dim in range(len(partition_index)):
                split_indices_list[dim].extend(partition_index[dim])
        else:
            split_indices_list = partition_index

    split_indices_list = list(
        map(lambda x, y: list(set(x) - set([y]) - set([0])), split_indices_list,
            complete_shape))
    split_indices_list = [sorted(x) for x in split_indices_list]

    return split_indices_list
