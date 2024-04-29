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
#include "paddle/phi/core/enforce.h"
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
ProcessMeshAttribute TensorDistAttribute::process_mesh_attr() const {
  return storage()->mesh_attr;
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
  PADDLE_ENFORCE_NOT_NULL(mesh,
                          common::errors::PreconditionNotMet(
                              "Building tensor_dist_attr through a nullptr "
                              "mesh attribute is currently not supported."));
  return Base::get(ctx, mesh, dims_mapping, partial_status);
}

///
/// \brief OperationDistAttribute interface.
///
ProcessMeshAttribute OperationDistAttribute::process_mesh_attr() const {
  return storage()->mesh_attr;
}
const std::vector<pir::Attribute>& OperationDistAttribute::operands() const {
  return storage()->operands;
}

uint32_t OperationDistAttribute::num_operands() const {
  return operands().size();
}

const std::vector<pir::Attribute>& OperationDistAttribute::results() const {
  return storage()->results;
}

uint32_t OperationDistAttribute::num_results() const {
  return results().size();
}

OperationDistAttribute OperationDistAttribute::get(
    pir::IrContext* ctx,
    ProcessMeshAttribute mesh,
    const std::vector<pir::Attribute>& operands,
    const std::vector<pir::Attribute>& results) {
  auto check_dist_attr = [=](pir::Attribute attr) {
    auto dist_attr = attr.dyn_cast<TensorDistAttribute>();
    auto ids = mesh.process_ids();
    for (const auto& id : dist_attr.process_mesh_attr().process_ids()) {
      PADDLE_ENFORCE_EQ(std::find(ids.begin(), ids.end(), id) != ids.end(),
                        true,
                        common::errors::PreconditionNotMet(
                            "operand_dist_attrs element's mesh(%s) not belong "
                            "to input mesh(%s)",
                            dist_attr.process_mesh_attr(),
                            mesh));
    }
  };
  for (auto attr : operands) {
    // NOTE: The operand dist attr maybe empty while the corresponding input is
    // optional.
    if (!attr) continue;
    if (auto array_attr = attr.dyn_cast<pir::ArrayAttribute>()) {
      for (size_t i = 0; i < array_attr.size(); ++i) {
        check_dist_attr(array_attr[i]);
      }
    } else {
      check_dist_attr(attr);
    }
  }
  return Base::get(ctx, mesh, operands, results);
}

}  // namespace dialect
}  // namespace paddle
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ProcessMeshAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::TensorDistAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::OperationDistAttribute)
