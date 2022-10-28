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
# limitations under the License.

import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import re
import paddle
from paddle.distributed.fleet.utils.log_util import logger
import os
import pickle
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage3 import (
    GroupShardedStage3,
)
from paddle.fluid.framework import dygraph_only
import copy

import numpy as np

__all__ = ["save_for_auto_inference"]


@dygraph_only
def save_for_auto_inference(path_prefix, dist_model, cvt2cpu=False):
    """
    Description：
        Save model parameters for auto parallel inference.
        Supporting dp + mp + pp + sharding(stage1), dp + sharding stage2-3.
        MoE not sdupported till MoE is supported in auto parallel mode.
    Args:
        path_prefix: path prefix to save
                    If `path_preifx` ends with path sepreator,
                        the path is processed as a directory and parameters will be saved in it,
                        automatically named saved_parameters.
                    Otherwisw, the parameters will be saved with name
                        path_preifx_dist{global_rank}.pdparams and  path_preifx_dist{global_rank}.pdattrs

        dist_model:
                model in distributed modeß
        cvt2cpu: wheather to move parameters to CPU when using sharding stage 3.
                The var is invalid if not using sharding stage 3.
    Returns:
        None
    Examples:
        dist_model = build_distributed_model()

        path_prefix = "path/to/save_infer"

        save_for_auto_inference(path_prefix, dist_model=dist_model, original_model=single_model, cvt2cpu=False)

    Outputs:
        path/to/save_infer_dist0.pdparams path/to/save_infer_dist1.pdparams path/to/save_infer_dist2.pdparams ...
        path/to/save_infer_dist0.pdattr  path/to/save_infer_dist0.pdattr   path/to/save_infer_dist0.pdattr ...

    """

    save_dir, basename_prefix = _get_abs_saved_prefix(path_prefix)

    if isinstance(dist_model, GroupShardedStage3):
        dist_model.get_all_parameters(cvt2cpu)

    wrapped_dict = _get_wrapped_dist_state_dict(dist_model.state_dict())
    global_rank = paddle.distributed.get_rank()

    # save parameters
    paddle.save(
        wrapped_dict,
        os.path.join(save_dir, f"{basename_prefix}_dist{global_rank}.pdparams"),
    )

    # save attributes
    _save_param_attr(
        wrapped_dict,
        os.path.join(save_dir, f"{basename_prefix}_dist{global_rank}.pdattr"),
    )

    # unset dims mapping after saving attrs
    for _, dist_param in wrapped_dict.items():
        _unset_dims_mapping(dist_param)


def _is_first_used(param):
    return not (
        hasattr(param, "is_firstly_shared") and not param.is_firstly_shared
    )


def _is_first_shared(param):
    return hasattr(param, "is_firstly_shared") and param.is_firstly_shared


def _get_all_ranks_of_pp(pp_rank):
    """
    Description:
        get all global ranks involving given pp_rank
    """
    hcg = fleet.get_hybrid_communicate_group()
    dp_degree = hcg.get_data_parallel_world_size()
    mp_degree = hcg.get_model_parallel_world_size()
    pp_degree = hcg.get_pipe_parallel_world_size()
    sharding_degree = hcg.get_sharding_parallel_world_size()

    process_group = []

    dp_degree = dp_degree * sharding_degree

    for i in range(dp_degree):
        for k in range(mp_degree):
            process_group.append(
                i * dist.get_world_size() // dp_degree
                + pp_rank * dist.get_world_size() // dp_degree // pp_degree
                + k
            )
    return process_group


def _save_param_attr(state_dict_, path, dims_mapping_dict=None):
    """
    Description:
        save params' attr dict
    Args:
        state_dict_:
            state for which to save attrs, when the state is optimzier state, the master and LRScheduler will be reomoved.
        path:
            path to save
        dims_mapping_dict:
            Dims mapping dict, mapping from parameter name in state_dict_ to dims_mapping.
            If parameter in state_dict_ has attribute 'dims_mapping', the dims_mapping is ignored.
            If parameter has no attribute 'dims_mapping', the dims mapping must contains the parameter's name.
    """
    state_dict = copy.copy(state_dict_)

    # remove master_weights and LRScheduler, which needs no parameter attributes to save
    state_dict.pop("master_weights", None)
    state_dict.pop("LR_Scheduler", None)

    if dims_mapping_dict is not None:
        assert isinstance(
            dims_mapping_dict, dict
        ), "dims_mapping_dict must be an instance of dict"
        for k in state_dict.keys():
            assert (
                k in dims_mapping_dict
            ), f"param {k} cannot find dims mapping in dims_mapping_dict"
    if dist.get_world_size() > 1:
        hcg = fleet.get_hybrid_communicate_group()
        dp_degree = hcg.get_data_parallel_world_size()
        mp_degree = hcg.get_model_parallel_world_size()

        dp_group = hcg.get_data_parallel_group()
        mp_group = hcg.get_model_parallel_group()
        pp_group = hcg.get_pipe_parallel_group()
    else:
        pp_group = None
        dp_group = None
        mp_group = None
        hcg = None

    logger.debug(f"dp group: {dp_group}")
    logger.debug(f"mp group: {mp_group}")
    logger.debug(f"pp group: {pp_group}")
    logger.debug(f"sharding group: {pp_group}")

    pp_rank = dist.get_rank(pp_group)

    # Why condition 'pp_rank < 0' exists?
    # Because if pp_degree = 1, pp_rank is set -1
    pp_rank = 0 if pp_rank <= 0 else pp_rank

    process_group = _get_all_ranks_of_pp(pp_rank)

    attr_dict = {}
    for k, v in state_dict.items():
        dims = len(v.shape)
        print("shape: ", k, dims)
        attr_d = {
            "process_shape": [dp_degree, mp_degree] if hcg else [1],
            "process_group": process_group,
            "dims_mapping": v.dims_mapping
            if hasattr(v, "dims_mapping")
            else dims_mapping_dict[k],
        }
        attr_dict[k] = attr_d

    with open(path, "wb") as f:
        pickle.dump(attr_dict, f)


def _unset_dims_mapping(param):
    if hasattr(param, "dims_mapping"):
        delattr(param, "dims_mapping")


def _get_dims_mapping(dist_parameter, mp_group, single_parameter=None):
    """
    Description:
        return the sliting mapping:
            {tensor_name: spiting_strategy}

    Examples:
        spliting_strategy's format (-1, -1, -1, 0), meaing the dims
        of  the tennsor is 4 and it is splited along the first strategy axis in mesh

    Mesh Examples: (2, 4) means dp=2, mp=4

    """

    import numpy as np

    dist_shape = np.array(dist_parameter.shape)
    if single_parameter is not None:
        single_shape = np.array(single_parameter.shape)
        logger.debug(f"single shape: {single_shape}, dist shape: {dist_shape}")
        assert len(dist_shape) == len(single_shape)
        diff = dist_shape - single_shape

        assert (diff <= 0).all(), (
            f"Dist shape is larger than single shape in some axis, please check the shapes. "
            f"dist shape: {dist_shape}, single shape: {single_shape}"
        )
        assert np.min(diff) == np.sum(diff), (
            f"There are more than one axis along which the tensor is splited, which are not allowed now. "
            f"dist shape: {dist_shape}, single shape: {single_shape}"
        )

        index = np.argsort(diff)[0]

        if diff[index] < 0:
            assert (
                single_shape[index] % dist_shape[index] == 0
                and dist.get_world_size(mp_group)
                == single_shape[index] // dist_shape[index]
            )

        # only tensor for mp is splited, and the mp axis is 1
        mapping = [-1 if d == 0 else 1 for d in diff]
    elif re.search(
        r"^column_|^row_parallel_linear.+\.w_[0-9]+$|^vocab_parallel_embedding",
        dist_parameter.name,
    ):
        assert re.search(
            r"^column_parallel_linear|^row_parallel_linear.+\.w_[0-9]+$|^vocab_parallel_embedding",
            dist_parameter.name,
        ), (
            f"Only 'column_parallel_linear', 'row_parallel_linear' and 'vocab_parallel_embedding' are allowed to be distributed, "
            f"while this parameter({dist_parameter.name}) is distributed now."
        )
        # using parameter name to determine the aixs along which the parameter is splited
        assert (
            1 <= len(dist_shape) <= 2
        ), f"Only 1 <= dims <= 2 is supported for distributed parameters, while the paramater's shape is {dist_shape}, name: {dist_parameter.name}"
        mapping = [-1 for _ in dist_shape]

        if len(dist_shape) == 1:
            mapping[0] = 1
        else:
            if re.search(r"^(row|vocab)", dist_parameter.name):
                mapping[0] = 1
            else:
                mapping[1] = 1
    elif hasattr(dist_parameter, "split_axis"):
        aixs = getattr(dist_parameter, "split_axis")
        mapping = [-1 for _ in dist_shape]
        mapping[aixs] = 1
        logger.debug(
            f"{dist_parameter.name} has attr split_axis: mapping: {mapping}"
        )
    else:
        mapping = [-1 for _ in dist_shape]
        logger.debug(f"normal parameter: {dist_parameter.name}")
    return mapping


def _get_abs_saved_prefix(path_prefix):
    """
    Description:
        Get absolute dir path and basename prefix of path_prefix, with making path_prefix's directories.
        If path_prefix is a directory name, basename is set 'saved_parameters'.
        If path_prefix is a file name, basename is extracted from path_prefix.
    Args:
        path_prefix: str
    Return:
        (dirpath: str, basename: str)
    """
    abs_prefix = os.path.abspath(path_prefix)
    if abs_prefix[-1] == os.path.sep:
        save_dir = abs_prefix
        basename_prefix = "saved_parameters"
    else:
        save_dir = os.path.dirname(abs_prefix)
        basename_prefix = os.path.basename(abs_prefix)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir, basename_prefix


def _name_mapping_dist2single(state_dict, pp_group):

    key_list = []
    param_keys = [
        v.name
        for _, v in state_dict.items()
        if isinstance(v, paddle.Tensor) and _is_first_used(v)
    ]
    dist.all_gather_object(key_list, param_keys, pp_group)

    # find how many a op in a each pp:
    # {"linear:"[0, 2,0,1,1,...]}
    param_types = {}

    for pp, keys in enumerate(key_list):
        param_type_idx = {}
        for k in keys:
            matched = re.search(r"^\w+_\d+(?=\.)", k)
            logger.debug(f"matched: {k}: {matched}")
            assert matched is not None
            name_idx = k[matched.start() : matched.end()]
            logger.debug(f"get param_type_idx: {name_idx}")
            if name_idx not in param_type_idx:
                name = "_".join(name_idx.split("_")[:-1])
                idx = int(name_idx.split("_")[-1])
                param_type_idx.update({name_idx: (name, idx)})
                if name not in param_types:
                    param_types[name] = [0] * pp_group.nranks
                param_types[name][pp] += 1

        # check if continous
        types_idx = {}
        for _, v in param_type_idx.items():
            if v[0] not in types_idx:
                types_idx.update({v[0]: [v[1]]})
            else:
                types_idx[v[0]].append(v[1])
        for k, v in types_idx.items():
            assert v == list(
                range(v[0], v[-1] + 1)
            ), f"{k} is not continous: {v}"

    logger.debug(f"param type: {param_types}")

    # analyse starting index
    for k in param_types.keys():
        param_types[k] = np.cumsum([0] + param_types[k][:-1])

    logger.debug(f"params type: {param_types}")

    name_mapping = {}
    pp_rank = dist.get_rank(pp_group)
    for k in key_list[pp_rank]:
        matched = re.search(r"^\w+_\d+(?=\.)", k)
        assert matched is not None
        name_idx = k[matched.start() : matched.end()]
        name = "_".join(name_idx.split("_")[:-1])
        idx = int(name_idx.split("_")[-1])
        logger.debug(f"idx: {idx}")
        new_idx = param_types[name][pp_rank] + idx
        logger.debug(f"new idx: {new_idx}")
        new_name_idx = name + "_" + str(new_idx)
        name_mapping[k] = new_name_idx + k[matched.end() :]

    return name_mapping


def _get_wrapped_dist_state_dict(dist_state_dict):

    state = dist_state_dict
    wrapped_state_dict = dict()
    if dist.get_world_size() <= 1:

        for _, v in state.itmes():
            wrapped_state_dict[v.name] = v
        return wrapped_state_dict

    hcg = fleet.get_hybrid_communicate_group()
    if hcg.get_pipe_parallel_world_size() <= 1:
        for _, v in state.itmes():
            wrapped_state_dict[v.name] = v
        return wrapped_state_dict

    pp_group = hcg.get_pipe_parallel_group()
    mp_group = hcg.get_model_parallel_group()

    name_mapping = _name_mapping_dist2single(state, pp_group)
    for _, v in state.items():
        if not _is_first_used(v):
            continue
        wrapped_state_dict[name_mapping[v.name]] = v
        setattr(v, "dims_mapping", _get_dims_mapping(v, mp_group))
        logger.debug(
            f"saving param: {v.name} -> {name_mapping[v.name]} shape: {v.shape}"
        )
    return wrapped_state_dict
