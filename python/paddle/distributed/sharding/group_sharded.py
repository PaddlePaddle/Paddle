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

import os
import logging
from enum import Enum

import paddle

from paddle.optimizer import Optimizer
from paddle.distributed.utils.log_utils import get_logger
from paddle.fluid.framework import in_dygraph_mode

# Old version
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.sharding_optimizer_stage2 import (
    ShardingOptimizerStage2,
)
from paddle.distributed.fleet.meta_parallel.sharding.sharding_stage2 import (
    ShardingStage2,
)
from paddle.distributed.fleet.meta_parallel.sharding.sharding_stage3 import (
    ShardingStage3,
)
from paddle.distributed.fleet.meta_parallel.sharding.sharding_utils import (
    ShardingScaler,
)

# New version
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import (
    GroupShardedOptimizerStage2,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage2 import (
    GroupShardedStage2,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage3 import (
    GroupShardedStage3,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import (
    GroupShardedScaler,
)

logger_ = get_logger(logging.WARNING)


def group_sharded_parallel(
    model,
    optimizer,
    level,
    scaler=None,
    group=None,
    offload=False,
    sync_buffers=False,
    buffer_max_size=2**23,
    segment_size=2**20,
    sync_comm=False,
):
    """
    Use group_sharded_parallel can perform group shared configuration on the model, optimizer and GradScaler. Level has three string options, 'os', 'os_g' and 'p_g_os' corresponds to three different usage scenarios: optimizer state segmentation, optimizer state + gradient segmentation, and parameter + gradient + optimizer state segmentation.
    Usually, optimizer state + gradient segmentation is actually a re optimization of optimizer state segmentation, so optimizer state + gradient segmentation can be used to realize optimizer state segmentation.

    Args:
        model (Layer): The layer to be wrapped with group_sharded_parallel.
        optimizer (Optimizer): The optimizer to be wrapped with group_sharded_parallel.
        level (str): The different level of the group sharded. Such as `os`, `os_g`, `p_g_os`.
        scaler (GradScaler, optional): If AMP is used, you need to pass GradScaler. Defaults to None, indicating that GradScaler is not used.
        group (Group, optional): The group instance. Defaults to None, indicating that the default environment group is used.
        offload (bool, optional): Whether to use the offload function. Defaults to False, which means that the offload function is not used.
        sync_buffers (bool, optional): Whether to broadcast model buffers. It is generally used when there are registered model buffers. Defaults to False, indicating that model buffers are not used.
        buffer_max_size (int, optional): The max size of the buffer used to integrate gradient in `os_g`. The larger the size, the more GPU memory will be used. Defaults to 2**23, which means that the dimension of the buffer is 2**23.
        segment_size (int, optional): The smallest size of parameter to be sharded in `p_g_os`. Defaults to 2**20, indicating that the dimension of the minimum segmented parameter is 2**20.
        sync_comm (bool, optional): Whether to use synchronous communication, only in `p_g_os` used. Defaults to False, indicating that asynchronous communication is used.

    Returns:
        model: A wrapper for group sharded given model.
        optimizer: A wrapper for group sharded given optimizer.
        scaler: A wrapper for group sharded given scaler.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            from paddle.fluid.dygraph.nn import Linear
            from paddle.distributed import fleet
            from paddle.distributed.sharding import group_sharded_parallel

            fleet.init(is_collective=True)
            group = paddle.distributed.new_group([0, 1])
            model = Linear(1000, 1000)

            clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
            optimizer = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters(), weight_decay=0.00001, grad_clip=clip)

            # wrap sharding model, optimizer and scaler
            model, optimizer, scaler = group_sharded_parallel(model, optimizer, "p_g", scaler=scaler)

            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True

            out = model(img)
            loss = paddle.nn.functional.cross_entropy(input=out, label=label)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
    """
    # check optition type
    assert isinstance(
        model, paddle.nn.Layer
    ), "The model must be the instance of paddle.nn.Layer."
    assert isinstance(
        optimizer, Optimizer
    ), "The optimizer must be the instance of paddle.optimizer.Optimizer."
    assert level in [
        'os',
        'os_g',
        'p_g_os',
    ], "The level must be os, os_g or p_g_os."

    def check_dtype(param):
        return param.dtype == paddle.float16

    params_fp16 = list(filter(check_dtype, model.parameters()))
    if scaler is None and len(params_fp16) > 0:
        raise ValueError("Please enter the correct scaler.")
    # convert model/optimizer/scaler
    if level in ['os', 'os_g']:
        logger_.info("*" * 30)
        logger_.info("Sharded level os uses sharded level os_g achieved now.")
        logger_.info("*" * 30)
        if in_dygraph_mode():
            optimizer = GroupShardedOptimizerStage2(
                params=optimizer._parameter_list,
                optim=optimizer,
                group=group,
                offload=offload,
            )
            model = GroupShardedStage2(
                model,
                optimizer,
                group=group,
                sync_buffers=sync_buffers,
                buffer_max_size=buffer_max_size,
            )
        else:
            optimizer = ShardingOptimizerStage2(
                params=model.parameters(),
                optim=optimizer,
                group=group,
                offload=offload,
            )
            model = ShardingStage2(
                model,
                optimizer,
                group=group,
                sync_buffers=sync_buffers,
                buffer_max_size=buffer_max_size,
            )
    elif level == 'p_g_os':
        if in_dygraph_mode():
            model = GroupShardedStage3(
                model,
                optimizer=optimizer,
                group=group,
                sync_buffers=sync_buffers,
                segment_size=segment_size,
                offload=offload,
                sync_comm=sync_comm,
            )
        else:
            model = ShardingStage3(
                model,
                optimizer=optimizer,
                group=group,
                sync_buffers=sync_buffers,
                segment_size=segment_size,
                offload=offload,
                sync_comm=sync_comm,
            )
    else:
        raise ValueError("Please enter the correct level.")
    if isinstance(scaler, paddle.amp.GradScaler):
        if in_dygraph_mode():
            scaler = GroupShardedScaler(scaler)
        else:
            scaler = ShardingScaler(scaler)
    logger_.info("*" * 30)
    logger_.info(
        "If there is a communication hang using group sharded, please check whether the communication operations of each process are unified."
    )
    logger_.info("*" * 30)

    return model, optimizer, scaler


def save_group_sharded_model(model, output, optimizer=None):
    """
    Group sharded encapsulated model and optimizer state saving module.

    Note:
        If using save_group_sharded_model saves the model. When loading again, you need to set the model or optimizer state before using group_sharded_parallel.

    Args:
        model (Layer): A wrapper for group sharded given model.
        output (str): Save directory.
        optimizer (Optimizer, optional): Group sharded encapsulated optimizer. Defaults to None, indicating that the optimizer state is not saved.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            from paddle.fluid.dygraph.nn import Linear
            from paddle.distributed import fleet
            from paddle.distributed.sharding import group_sharded_parallel, save_group_sharded_model

            fleet.init(is_collective=True)
            group = paddle.distributed.new_group([0, 1])
            model = Linear(1000, 1000)

            clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
            optimizer = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters(), weight_decay=0.00001, grad_clip=clip)

            # wrap sharding model, optimizer and scaler
            model, optimizer, scaler = group_sharded_parallel(model, optimizer, "p_g", scaler=scaler)

            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True

            out = model(img)
            loss = paddle.nn.functional.cross_entropy(input=out, label=label)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            # save model and optimizer state_dict
            save_group_sharded_model(model, optimizer, output=output_dir)
    """
    logger_.info(
        "==========Begin to save group sharded model and optimizer=========="
    )
    assert not os.path.isfile(
        output
    ), "Saving directory ({}) should be a directory, not a file".format(output)
    os.makedirs(output, exist_ok=True)
    output_model = os.path.join(output, "model.pdmodel")
    if isinstance(model, (ShardingStage2, GroupShardedStage2)):
        paddle.save(model._layer.state_dict(), output_model)
    elif isinstance(model, (ShardingStage3, GroupShardedStage3)):
        convert2cpu = True if model._offload else False
        model.get_all_parameters(convert2cpu=convert2cpu)
        paddle.save(model._layer.state_dict(), output_model)
    else:
        raise ValueError(
            "Please use the layer which is wrapped with group_sharded_parallel."
        )

    if optimizer is not None:
        assert hasattr(
            optimizer, "_optim"
        ), "Please use the optimizer which is wrapped with group_sharded_parallel."
        output_opt = os.path.join(output, "model.pdopt")
        paddle.save(optimizer._optim.state_dict(), output_opt)
    logger_.info(
        "==========End to save group sharded model and optimizer=========="
    )
