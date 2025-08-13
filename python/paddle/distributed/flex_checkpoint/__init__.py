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

from .aoa.aoa_engine import (
    AoAEngine,
    ShardedTensorDesc,
    ShardMapping,
)
from .dcp.reshard import (
    reshard_sharded_state_dict,
)
from .dcp.sharded_tensor import (
    ShardedStateDict,
    ShardedTensor,
    build_sharded_state_dict,
    create_sharded_tensor_with_new_local,
    make_replicated_sharded_tensor,
    make_tp_sharded_tensor_for_checkpoint,
    shard_weight,
)

__all__ = [
    "AoAEngine",
    "ShardedTensorDesc",
    "ShardMapping",
    "ShardedTensor",
    "ShardedStateDict",
    "shard_weight",
    "make_tp_sharded_tensor_for_checkpoint",
    "make_replicated_sharded_tensor",
    "build_sharded_state_dict",
    "create_sharded_tensor_with_new_local",
    "reshard_sharded_state_dict",
]
