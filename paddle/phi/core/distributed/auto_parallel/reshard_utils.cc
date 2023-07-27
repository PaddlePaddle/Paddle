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

#include "paddle/phi/core/distributed/auto_parallel/reshard_utils.h"

#include <cstdlib>
#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"

namespace phi {
namespace distributed {

bool IsDimsMappingShard(const std::vector<int64_t>& dims_mapping) {
  return std::any_of(dims_mapping.begin(),
                     dims_mapping.end(),
                     [](int64_t value) { return value != -1; });
}

bool IsDimsMappingReplicated(const std::vector<int64_t>& dims_mapping) {
  return std::all_of(dims_mapping.begin(),
                     dims_mapping.end(),
                     [](int64_t value) { return value == -1; });
}

int64_t GetCurGlobalRank() {
  const char* cur_rank = std::getenv("PADDLE_TRAINER_ID");
  PADDLE_ENFORCE_NOT_NULL(
      cur_rank,
      phi::errors::NotFound(
          "The environment variable 'PADDLE_TRAINER_ID' cannot be found."));
  return std::atoi(cur_rank);
}

std::vector<int64_t> GetCurRankCoordInMesh(const ProcessMesh& process_mesh) {
  const auto& process_shape = process_mesh.shape();
  const auto& process_ids = process_mesh.process_ids();
  int64_t ndims_mesh = process_shape.size();
  int64_t cur_global_rank = GetCurGlobalRank();

  VLOG(3) << "Searching current global rank " << cur_global_rank
          << " in process_mesh " << process_mesh;

  auto iter =
      std::find(process_ids.begin(), process_ids.end(), cur_global_rank);
  PADDLE_ENFORCE_NE(
      iter,
      process_ids.end(),
      phi::errors::NotFound("Rank %lld cannot be found in process_mesh",
                            cur_global_rank));

  int64_t flat_idx_in_mesh = iter - process_ids.begin();

  std::vector<int64_t> coord(ndims_mesh, -1);
  for (int64_t i = ndims_mesh - 1; i >= 0; --i) {
    coord[i] = flat_idx_in_mesh % process_shape[i];
    flat_idx_in_mesh /= process_shape[i];
  }
  return coord;
}

std::map<int64_t, int64_t> GetSplitAxisWithDimsMapping(
    const std::vector<int64_t>& dims_mapping) {
  std::map<int64_t, int64_t> split_axis_to_mesh_axis;
  for (size_t i = 0; i < dims_mapping.size(); ++i) {
    if (dims_mapping[i] != -1) {
      split_axis_to_mesh_axis.emplace(i, dims_mapping[i]);
    }
  }
  return split_axis_to_mesh_axis;
}

}  // namespace distributed
}  // namespace phi
