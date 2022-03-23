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
from paddle.distributed.models.moe import utils as _utils
expert_count = _utils._number_count
assign_pos = _utils._assign_pos
limit_by_capacity = _utils._limit_by_capacity
prune_gate_by_capacity = _utils._prune_gate_by_capacity
random_routing = _utils._random_routing
