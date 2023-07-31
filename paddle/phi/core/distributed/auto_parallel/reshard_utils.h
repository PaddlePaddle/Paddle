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

#pragma once

#include <cstdint>
#include <map>
#include <vector>

namespace phi {
namespace distributed {
namespace auto_parallel {

class ProcessMesh;
}  // namespace auto_parallel

using auto_parallel::ProcessMesh;

bool IsDimsMappingShard(const std::vector<int64_t>& dims_mapping);

bool IsDimsMappingReplicated(const std::vector<int64_t>& dims_mapping);

int64_t GetCurGlobalRank();

// Get the coordinate of cur rank in process mesh. For example, the process mesh
// is [[0, 1], [2, 3], [4, 5], [6, 7]], if the current rank is 4, then will
// return [2, 0]; if the current rank is 3, then will return [1, 1].
std::vector<int64_t> GetCurRankCoordInMesh(const ProcessMesh& process_mesh);

// If the index i's value in dims_mapping is x ( x != -1), means the ith axis of
// tensor need be split by xth axis of process_mesh. The function analyze the
// input vector, return a key-value map of tensor_split_axis and
// process_mesh_split_axis.
// For example, if dims_mapping is [-1, 1, -1, 0], will return {1: 1, 3: 0}.
std::map<int64_t, int64_t> GetSplitAxisWithDimsMapping(
    const std::vector<int64_t>& dims_mapping);

}  // namespace distributed
}  // namespace phi
