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
#include "paddle/fluid/pir/dialect/distributed/ir/dist_op.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/pir/include/core/value.h"

namespace paddle {
namespace dialect {

pir::Value shard_tensor(
    const pir::Value& x,
    const phi::distributed::ProcessMesh& process_mesh,
    const std::vector<int64_t>& dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& partial_status = {});

pir::Value reshard(
    const pir::Value& x,
    const phi::distributed::ProcessMesh& process_mesh,
    const std::vector<int64_t>& dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& partial_status = {},
    const phi::distributed::Placements& placements = {});

pir::Value reshard(const pir::Value& x,
                   const TensorDistAttribute& tensor_dist_attr);

pir::Value dtensor_from_local(
    const pir::Value& x,
    const phi::distributed::ProcessMesh& process_mesh,
    const std::vector<int64_t>& dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& partial_status = {});
pir::Value dtensor_from_local(const pir::Value& x,
                              const TensorDistAttribute& tensor_dist_attr);

pir::Value dtensor_to_local(const pir::Value& x);

std::vector<pir::Value> moe_sub_mesh_tensors(
    const pir::Value& input,
    const std::vector<phi::distributed::ProcessMesh>& local_mesh_list,
    const std::vector<int64_t>& local_dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& local_partial_status,
    const phi::distributed::ProcessMesh& global_mesh,
    const std::vector<int64_t>& global_dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& global_partial_status);

pir::Value moe_global_mesh_tensor(
    const std::vector<pir::Value>& inputs,
    const std::vector<phi::distributed::ProcessMesh>& local_mesh_list,
    const std::vector<int64_t>& local_dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& local_partial_status,
    const phi::distributed::ProcessMesh& global_mesh,
    const std::vector<int64_t>& global_dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& global_partial_status,
    const std::vector<int64_t>& global_shape);

pir::Value dist_reshape(
    const pir::Value& x,
    const phi::distributed::Placements& x_placements,
    const std::vector<int64_t>& global_shape,
    const std::vector<int64_t>& local_shape,
    const phi::distributed::ProcessMesh& mesh,
    const phi::distributed::Placements& placements,
    const std::vector<int64_t>& dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& partial_status);

}  // namespace dialect
}  // namespace paddle
