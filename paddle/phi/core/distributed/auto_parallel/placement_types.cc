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

namespace phi::distributed {

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
          common::errors::InvalidArgument(
              "Tensor dim %lld is already sharded on mesh dim %lld,"
              " DistTensor operator implementation does not support things "
              "like hybrid"
              " sharding strategies yet (i.e. [Shard(0), Shard(0)])",
              shard_dim,
              dim_map[shard_dim]));
      dim_map[shard_dim] = i;  // NOLINT
    }
  }
  return dim_map;
}

bool DistTensorMeta::is_replicated() const {
  return std::all_of(placements_.cbegin(),
                     placements_.cend(),
                     [](const auto& p) { return p->is_replicated(); });
}

bool equal_placements(const Placements& a, const Placements& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (*a[i] != *b[i]) {
      return false;
    }
  }
  return true;
}

phi::distributed::Placements cvt_dim_map_to_placements(
    const ProcessMesh& process_mesh,
    const std::vector<int64_t>& dim_mapping,
    const paddle::flat_hash_map<int64_t, phi::ReduceType>& partial_status) {
  phi::distributed::Placements placements;
  placements.resize(process_mesh.ndim(),
                    std::make_shared<phi::distributed::Replicate>());

  for (const auto& pair : partial_status) {
    placements[pair.first] =
        std::make_shared<phi::distributed::Partial>(pair.second);
  }

  for (size_t i = 0; i < dim_mapping.size(); ++i) {
    auto& mesh_id = dim_mapping[i];
    if (mesh_id >= 0) {
      auto& p = placements[mesh_id];
      if (p->is_shard()) {
        PADDLE_THROW(common::errors::PreconditionNotMet(
            "ProcessMesh dimension cann't be mapped to two  dimension of the "
            "same tensor: {%d} and {%d}",
            i,
            dynamic_cast<phi::distributed::Shard&>(*p).get_dim()));
      } else if (p->is_partial()) {
        PADDLE_THROW(common::errors::PreconditionNotMet(
            "ProcessMesh dimension {%d} cannot be both shard and partial!",
            mesh_id));
      }
      placements[mesh_id] = std::make_shared<phi::distributed::Shard>(i);
    }
  }
  return placements;
}

}  // namespace phi::distributed
