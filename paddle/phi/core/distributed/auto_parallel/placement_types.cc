// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/core/distributed/auto_parallel/placement_types.h"

namespace phi {
namespace distributed {

int64_t DistTensorMeta::num_shard() const {
  int64_t num_shard = 1;
  const auto& mesh_shape = process_mesh_->shape();
  for (size_t i = 0; i < placements_.size(); i++) {
    if (placements_[i]->is_shard()) {
      num_shard *= mesh_shape[i];
    }
  }
  return num_shard;
}

std::vector<int64_t> DistTensorMeta::dim_mapping() const {
  int64_t ndim = dims().size();
  std::vector<int64_t> dim_map(ndim, -1);
  for (size_t i = 0; i < placements_.size(); i++) {
    auto& placement = placements_[i];
    if (placement->is_shard()) {
      auto shard_dim = dynamic_cast<const Shard&>(*placement).get_dim();
      PADDLE_ENFORCE_EQ(
          dim_map[shard_dim],
          -1,
          phi::errors::InvalidArgument(
              "Tensor dim %lld is already sharded on mesh dim %lld,"
              " DistTensor operator implementation does not support things "
              "like hybrid"
              " sharding strategies yet (i.e. [Shard(0), Shard(0)])",
              shard_dim,
              dim_map[shard_dim]));
      dim_map[shard_dim] = i;
    }
  }
  return dim_map;
}

bool DistTensorMeta::is_replicated() const {
  return std::all_of(placements_.cbegin(),
                     placements_.cend(),
                     [](const auto& p) { return p->is_replicated(); });
}

}  // namespace distributed
}  // namespace phi
