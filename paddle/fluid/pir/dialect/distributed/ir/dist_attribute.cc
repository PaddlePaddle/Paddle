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
namespace paddle::dialect {
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

const phi::distributed::Placements& PlacementsAttribute::placements() const {
  return storage()->placements;
}

PlacementsAttribute PlacementsAttribute::get(
    pir::IrContext* ctx, const phi::distributed::Placements& placements) {
  return Base::get(ctx, placements);
}

std::string PlacementsAttribute::to_string() const {
  return PlacementsAttrStorage::to_string(placements());
}

size_t PlacementsAttribute::hash() const {
  return std::hash<std::string>()(to_string());
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
std::optional<PlacementsAttribute> TensorDistAttribute::placements_attr()
    const {
  return storage()->placements_;
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

phi::distributed::Placements TensorDistAttribute::placements() const {
  auto process_mesh = process_mesh_attr();
  phi::distributed::Placements placements;
  placements.resize(process_mesh.ndim(),
                    std::make_shared<phi::distributed::Replicate>());

  for (const auto& pair : partial_status()) {
    placements[pair.first] =
        std::make_shared<phi::distributed::Partial>(pair.second);
  }

  auto& dim_mapping = dims_mapping();
  for (size_t i = 0; i < dim_mapping.size(); ++i) {
    auto& mesh_id = dim_mapping[i];
    if (mesh_id >= 0) {
      auto& p = placements[mesh_id];
      if (p->is_shard()) {
        PADDLE_THROW(common::errors::PreconditionNotMet(
            "ProcessMesh dimension can't be mapped to two  dimension of the "
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

TensorDistAttribute TensorDistAttribute::get(
    pir::IrContext* ctx,
    ProcessMeshAttribute mesh,
    const std::vector<int64_t>& dims_mapping,
    const flat_hash_map<int64_t, phi::ReduceType>& partial_status,
    const std::optional<PlacementsAttribute>& placements) {
  PADDLE_ENFORCE_NOT_NULL(mesh,
                          common::errors::PreconditionNotMet(
                              "Building tensor_dist_attr through a nullptr "
                              "mesh attribute is currently not supported."));

  if (!placements.has_value() && !mesh.empty()) {
    phi::distributed::Placements p =
        phi::distributed::cvt_dim_map_to_placements(
            mesh.process_mesh(), dims_mapping, partial_status);
    return Base::get(ctx,
                     mesh,
                     dims_mapping,
                     partial_status,
                     PlacementsAttribute::get(ctx, p));
  } else {
    return Base::get(ctx, mesh, dims_mapping, partial_status, placements);
  }
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

int64_t OperationDistAttribute::chunk_id() const { return storage()->chunk_id; }

OperationDistAttribute OperationDistAttribute::get(
    pir::IrContext* ctx,
    ProcessMeshAttribute mesh,
    const std::vector<pir::Attribute>& operands,
    const std::vector<pir::Attribute>& results,
    const int64_t& chunk_id) {
  auto check_dist_attr = [=](pir::Attribute attr) {
    auto dist_attr = attr.dyn_cast<TensorDistAttribute>();
    auto ids = mesh.process_ids();
    const ProcessMeshAttribute& dist_mesh = dist_attr.process_mesh_attr();
    for (const auto& id : dist_mesh.process_ids()) {
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
  return Base::get(ctx, mesh, operands, results, chunk_id);
}

}  // namespace paddle::dialect
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ProcessMeshAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::PlacementsAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::TensorDistAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::OperationDistAttribute)
