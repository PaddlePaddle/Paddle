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

import paddle
from paddle.distributed import fleet

from .base.topology import ParallelMode
from .meta_parallel import (
    PipelineLayer,
    PipelineParallel,
    PipelineParallelWithInterleave,
    ShardingParallel,
    TensorParallel,
)

_grad_scalar = None


def distributed_model(model):
    """
    Return distributed data parallel model (Only work in dygraph mode)

    Args:
        model (Layer): the user-defind model which inherits Layer.

    Returns:
        distributed data parallel model which inherits Layer.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.nn as nn
            from paddle.distributed import fleet

            class LinearNet(nn.Layer):
                def __init__(self):
                    super().__init__()
                    self._linear1 = nn.Linear(10, 10)
                    self._linear2 = nn.Linear(10, 1)

                def forward(self, x):
                    return self._linear2(self._linear1(x))

            # 1. initialize fleet environment
            fleet.init(is_collective=True)

            # 2. create layer & optimizer
            layer = LinearNet()
            loss_fn = nn.MSELoss()
            adam = paddle.optimizer.Adam(
                learning_rate=0.001, parameters=layer.parameters())

            # 3. get data_parallel model using fleet
            adam = fleet.distributed_optimizer(adam)
            dp_layer = fleet.distributed_model(layer)

            # 4. run layer
            inputs = paddle.randn([10, 10], 'float32')
            outputs = dp_layer(inputs)
            labels = paddle.randn([10, 1], 'float32')
            loss = loss_fn(outputs, labels)

            print("loss:", loss.numpy())

            loss.backward()

            adam.step()
            adam.clear_grad()


    """
    fleet_env = fleet.fleet

    assert model is not None, "model should not be None"
    if fleet_env.worker_num() <= 1:
        return model

    amp_enable = False
    strategy = fleet_env._user_defined_strategy
    if strategy.amp:
        amp_enable = True
        amp_level = "O2" if strategy.amp_configs['use_pure_fp16'] else "O1"
        if amp_level.upper() == "O2":
            model = paddle.amp.decorate(
                models=model,
                optimizers=None,
                level="O2",
                master_weight=None,
                save_dtype=None,
            )
        init_loss_scaling = strategy.amp_configs['init_loss_scaling']
        incr_ratio = strategy.amp_configs['incr_ratio']
        decr_ratio = strategy.amp_configs['decr_ratio']
        incr_every_n_steps = strategy.amp_configs['incr_every_n_steps']
        decr_every_n_nan_or_inf = strategy.amp_configs[
            'decr_every_n_nan_or_inf'
        ]
        use_dynamic_loss_scaling = strategy.amp_configs[
            'use_dynamic_loss_scaling'
        ]

        global _grad_scalar
        _grad_scalar = paddle.amp.GradScaler(
            init_loss_scaling=init_loss_scaling,
            incr_ratio=incr_ratio,
            decr_ratio=decr_ratio,
            incr_every_n_steps=incr_every_n_steps,
            decr_every_n_nan_or_inf=decr_every_n_nan_or_inf,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling,
        )

    if strategy.heter_ccl_mode:
        distributed_model = paddle.DataParallel(
            model,
            comm_buffer_size=strategy.fuse_grad_size_in_MB,
            last_comm_buffer_size=strategy.last_comm_group_size_MB,
            find_unused_parameters=strategy.find_unused_parameters,
        )
        return distributed_model

    if fleet_env._hcg.get_parallel_mode() == ParallelMode.SHARDING_PARALLEL:
        model = ShardingParallel(model, fleet_env._hcg, strategy=strategy)
    elif fleet_env._hcg.get_parallel_mode() == ParallelMode.DATA_PARALLEL:

        # NOTE (JZ-LIANG) init parameters broadcast within sharding group
        # normally it should be done inside DataParallel
        if fleet_env.sharding_degree > 1:
            from paddle.distributed.fleet.utils.hybrid_parallel_util import (
                broadcast_sharding_parameters,
            )

            assert (
                fleet_env.sharding_degree
                == fleet_env._hcg.get_sharding_parallel_world_size()
            )
            broadcast_sharding_parameters(model, fleet_env._hcg)
        model = paddle.DataParallel(
            model,
            comm_buffer_size=strategy.fuse_grad_size_in_MB,
            last_comm_buffer_size=strategy.last_comm_group_size_MB,
            find_unused_parameters=strategy.find_unused_parameters,
            group=fleet_env._hcg.get_data_parallel_group(),
        )
    elif fleet_env._hcg.get_parallel_mode() == ParallelMode.TENSOR_PARALLEL:
        model = TensorParallel(model, fleet_env._hcg, strategy=strategy)
    elif fleet_env._hcg.get_parallel_mode() == ParallelMode.PIPELINE_PARALLEL:
        assert isinstance(
            model, PipelineLayer
        ), "For pipeline parallel, the model should an instance of PipelineLayer"
        if model.get_num_virtual_stages() == 1:
            # 1f1b pipeline
            model = PipelineParallel(model, fleet_env._hcg, strategy=strategy)
        else:
            # interleave pipeline
            model = PipelineParallelWithInterleave(
                model, fleet_env._hcg, strategy=strategy
            )

    return model
