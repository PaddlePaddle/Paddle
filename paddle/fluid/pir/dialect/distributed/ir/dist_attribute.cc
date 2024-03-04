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

#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/attribute_storage.h"
namespace paddle {
namespace dialect {
///
/// \brief ProcessMeshAttribute interface.
///
const phi::distributed::ProcessMesh& ProcessMeshAttribute::process_mesh()
    const {
  return storage()->process_mesh;
}
ProcessMeshAttribute ProcessMeshAttribute::get(
    pir::IrContext* ctx, const phi::distributed::ProcessMesh& mesh) {
  return Base::get(ctx, mesh);
}
ProcessMeshAttribute ProcessMeshAttribute::get(
    pir::IrContext* ctx,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& process_ids,
    const std::vector<std::string>& dim_names) {
  return Base::get(ctx, shape, process_ids, dim_names);
}

///
/// \brief TensorDistAttribute interface.
///
ProcessMeshAttribute TensorDistAttribute::mesh_attr() const {
  return storage()->process_mesh;
}
const std::vector<int64_t>& TensorDistAttribute::dims_mapping() const {
  return storage()->dims_mapping;
}

std::set<int64_t> TensorDistAttribute::partial_dims() const {
  auto& partial = partial_status();
  std::set<int64_t> keys;
  for (auto& kv : partial) {
    keys.emplace(kv.first);
  }
  return keys;
}

const flat_hash_map<int64_t, phi::ReduceType>&
TensorDistAttribute::partial_status() const {
  return storage()->partial_status;
}

TensorDistAttribute TensorDistAttribute::get(
    pir::IrContext* ctx,
    ProcessMeshAttribute mesh,
    const std::vector<int64_t>& dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& partial_status) {
  return Base::get(ctx, mesh, dims_mapping, partial_status);
}

}  // namespace dialect
}  // namespace paddle
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ProcessMeshAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::TensorDistAttribute)
