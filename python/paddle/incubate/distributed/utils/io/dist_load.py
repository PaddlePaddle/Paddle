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

import copy
import re

import paddle
import paddle.distributed as dist
from paddle.base.framework import dygraph_only
from paddle.distributed import fleet


@dygraph_only
def load(path, **configs):
    """
    Load an object can be used in paddle from specified path.
    The file is saved by distributed.save

    Note:
        The file to load must be saved bu the API paddle.incubate.distributed.utils.io.save

    Args:
        path(str|BytesIO) : The path/buffer to load the target object. Generally, the path is the target
            file path. When loading state_dict from the saved result of the API used to save
            the inference model, the path may be a file prefix or directory.
        **configs (dict, optional): other load configuration options for compatibility. We do not
            recommend using these configurations, they may be removed in the future. If not necessary,
            DO NOT use them. Default None.
            The following options are currently supported:
                (1) place: where to place the loaded state dict.
                     If the state dict is too large, the place should be set 'cpu'.
            Note:
                Other config value may cause some error.Please don't use any more config options.
    Returns:
        Object(Object): a target object can be used in paddle

    Examples:
        import paddle
        paddle.distributed.init_process_group(backend='nccl')
        paddle.distributed.fleet.init(is_collective=True)

        model = build_model()
        optimizer = build_optimizer(model)

        dist_model = paddle.distributed_optimizer(model)
        dist_optimizer = paddle.distributed_optimizer(optimizer)


        # load model state dict
        model_state_dict = paddle.incubate.distributed.utils.io.load(path="path/to/load.pdparams")
        dist_model.set_state_dict(model_state_dict)

        # load optimizer state dict
        optimizer_state_dict = paddle.incubate.distributed.utils.io.load(path="path/to/load.pdopt")
        dist_optimizer.set_state_dict(optimizer_state_dict)

    """
    if dist.get_world_size() == 1:
        return paddle.load(path, **configs)

    hcg = fleet.get_hybrid_communicate_group()
    assert (
        hcg.get_model_parallel_world_size() == 1
        and hcg.get_pipe_parallel_world_size() == 1
    ), "Sharding and DP are supported only now"

    # assert (
    #     "place" in configs
    # ), "the arg place ('cpu' or 'gpu:0', 'gpus:1' ...)must be passed"
    if "place" not in configs:
        configs["place"] = "cpu"
    place = configs["place"]
    assert isinstance(
        place, str
    ), f"configs[place] must be a str, but this is a {type(place)}"

    assert re.search(
        "^(cpu|gpu:[0-9]*)$", place
    ), "configs[place] must be cpu, gpu:0, gpu:1 ..."

    return load_with_place(path, **configs)


def load_with_place(path, **configs):
    place = configs["place"]
    if place is None:
        return paddle.load(path)

    origin_place = paddle.get_device()
    paddle.set_device(place)

    configs = _remove_not_supported_items(configs)
    state_dict = paddle.load(path, **configs)

    paddle.set_device(origin_place)

    return state_dict


def _remove_not_supported_items(configs):
    __supported_by_load__ = [
        "model_filename",
        "params_filename",
        "return_numpy",
    ]
    _configs = copy.copy(configs)
    for k in configs.keys():
        if k not in __supported_by_load__:
            _configs.pop(k, None)
    return _configs
