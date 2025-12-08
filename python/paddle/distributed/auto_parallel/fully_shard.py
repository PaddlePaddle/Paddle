# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from types import MethodType

import paddle.distributed as dist


def shard_accumulators(parameters_and_grads, optimizer, target_block):
    for param, _ in parameters_and_grads:
        del param._need_shard_auto
        optimizer._create_accumulators(
            target_block,
            [param],
        )
        target_name = param.name
        if param.name in optimizer._master_weights.keys():
            master_weight = optimizer._master_weights[param.name]
            target_name = master_weight.name
        for key in optimizer._accumulators.keys():
            accumulator = optimizer._accumulators[key][target_name]
            if accumulator.is_dist():
                continue
            origin_accumulator_name = accumulator.name

            if 'beta' not in key:
                placements = param.placements
            else:
                placements = [
                    dist.Replicate()
                    for _ in range(len(param.process_mesh.shape))
                ]
            optimizer._accumulators[key][target_name] = dist.shard_tensor(
                accumulator,
                mesh=param.process_mesh,
                placements=placements,
            )
            optimizer._accumulators[key][
                target_name
            ].name = origin_accumulator_name

    def _finish_update_impl(self, block, parameters_and_grads):
        if not isinstance(parameters_and_grads, list):
            parameters_and_grads = parameters_and_grads['params']
        for p, _ in parameters_and_grads:
            p.main_grad = None

    optimizer._finish_update = MethodType(_finish_update_impl, optimizer)


class FullyShardAuto:
    def __init__(self, model, mesh):
        self.model = model
        # use first dims as sharding axis
        self._shard_fn = dist.ShardingStage3(0, mesh)
        for param in self.model.parameters():
            param._need_shard_auto = True
            self._shard_fn._shard_parameter(param)
            self._shard_fn._register_hook_for_param_grad(param)
        os.environ["skip_sharding3_output_reshard"] = "1"
