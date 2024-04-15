// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/pir/include/core/value.h"

namespace paddle {
namespace dialect {

pir::Value shard_tensor(const pir::Value& x,
                        const phi::distributed::ProcessMesh& process_mesh,
                        const std::vector<int64_t>& dims_mapping,
                        const flat_hash_map<int64_t, phi::ReduceType>& partial_status = {});

pir::Value reshard(
    const pir::Value& x,
    const phi::distributed::ProcessMesh& process_mesh,
    const std::vector<int64_t>& dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& partial_status = {});

pir::Value reshard(const pir::Value& x,
                   const TensorDistAttribute& tensor_dist_attr);

}  // namespace dialect
}  // namespace paddle
