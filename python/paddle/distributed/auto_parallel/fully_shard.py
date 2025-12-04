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

import paddle.distributed as dist
from paddle.distributed import fleet


class FullyShardAuto:
    def __init__(self, model, shard_fn=None, gradient_accumulation_steps=1):
        mesh = fleet.auto.get_mesh()
        sharding_mesh_dim = 'dp'
        self._shard_fn = dist.ShardingStage3(sharding_mesh_dim, mesh)
        self._sharding_axis = None
        self._sharding_degree = None
        self.gradient_accumulation_steps = gradient_accumulation_steps

        global_mesh = fleet.auto.get_mesh()
        if global_mesh:
            self._sharding_degree = global_mesh.get_dim_size(
                self._shard_fn._sharding_mesh_dim
            )
        elif self._shard_fn._mesh:
            self._sharding_degree = self._shard_fn._mesh.get_dim_size(
                self._shard_fn._sharding_mesh_dim
            )
        self._sharding_axis = 0
        self._shard_fn._set_sharding_axis(self._sharding_axis)
        self.model = model
        for param in self.model.parameters():
            param._need_shard = True
        for param in self.model.parameters():
            self._shard_fn._shard_parameter(param)
        for param in self.model.parameters():
            self._shard_fn._register_hook_for_param_grad(param)
        os.environ["skip_sharding3_output_reshard"] = "1"
