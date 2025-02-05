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

#include "paddle/phi/common/reduce_type.h"
#include "paddle/phi/core/distributed/auto_parallel/placement_types.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_attribute_storage.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/utils/flat_hash_map.h"
namespace paddle {
namespace dialect {
class ProcessMeshAttrStorage;
class TensorDistAttrStorage;
class OperationDistAttrStorage;
class PlacementsAttrStorage;

class ProcessMeshAttribute : public pir::AttrBase<ProcessMeshAttribute,
                                                  pir::Attribute,
                                                  ProcessMeshAttrStorage> {
 public:
  using Base::Base;
  const phi::distributed::ProcessMesh& process_mesh() const;
  const std::vector<int64_t>& shape() const { return process_mesh().shape(); }
  const std::vector<int64_t>& process_ids() const {
    return process_mesh().process_ids();
  }
  const std::vector<std::string>& dim_names() const {
    return process_mesh().dim_names();
  }
  int64_t size() const { return process_mesh().size(); }
  int64_t ndim() const { return process_mesh().ndim(); }
  int64_t dim_size(int64_t dim) const { return process_mesh().dim_size(dim); }
  int64_t dim_size(const std::string& dim_name) const {
    return process_mesh().dim_size(dim_name);
  }
  bool empty() const { return process_mesh().empty(); }
  bool contains(int64_t process_id) const {
    return process_mesh().contains(process_id);
  }
  size_t hash() const { return process_mesh().hash(); }

  std::string to_string() const { return process_mesh().to_string(); }

  static ProcessMeshAttribute get(pir::IrContext* ctx,
                                  const phi::distributed::ProcessMesh& mesh);
  static ProcessMeshAttribute get(pir::IrContext* ctx,
                                  const std::vector<int64_t>& shape,
                                  const std::vector<int64_t>& process_ids,
                                  const std::vector<std::string>& dim_names);

  static std::string name() { return "a_process_mesh"; }
};

class PlacementsAttribute : public pir::AttrBase<PlacementsAttribute,
                                                 pir::Attribute,
                                                 PlacementsAttrStorage> {
 public:
  using Base::Base;
  const phi::distributed::Placements& placements() const;

  size_t hash() const;
  std::string to_string() const;

  static std::string name() { return "a_placements"; }

  static PlacementsAttribute get(
      pir::IrContext* ctx, const phi::distributed::Placements& placements);
};

class TensorDistAttribute : public pir::AttrBase<TensorDistAttribute,
                                                 pir::Attribute,
                                                 TensorDistAttrStorage> {
 public:
  using Base::Base;
  ProcessMeshAttribute process_mesh_attr() const;
  const std::vector<int64_t>& dims_mapping() const;
  std::optional<PlacementsAttribute> placements_attr() const;

  // return vector of mesh dims on which the this tensor is partial on
  std::set<int64_t> partial_dims() const;

  const flat_hash_map<int64_t, phi::ReduceType>& partial_status() const;

  phi::distributed::Placements placements() const;

  // construct a new attribute with new mesh attribute.
  TensorDistAttribute CopyWithNewMesh(ProcessMeshAttribute mesh) const {
    return get(ir_context(), mesh, dims_mapping(), partial_status());
  }

  static TensorDistAttribute get(
      pir::IrContext* ctx,
      ProcessMeshAttribute mesh,
      const std::vector<int64_t>& dims_mapping,
      const flat_hash_map<int64_t, phi::ReduceType>& partial_status = {},
      const std::optional<PlacementsAttribute>& placements = std::nullopt);
  static TensorDistAttribute get(
      pir::IrContext* ctx,
      const phi::distributed::ProcessMesh& mesh,
      const std::vector<int64_t>& dims_mapping,
      const flat_hash_map<int64_t, phi::ReduceType>& partial_status = {},
      const std::optional<PlacementsAttribute>& placements = std::nullopt) {
    return get(ctx,
               ProcessMeshAttribute::get(ctx, mesh),
               dims_mapping,
               partial_status,
               placements);
  }

  static std::string name() { return "a_tensor_dist"; }
};

class OperationDistAttribute : public pir::AttrBase<OperationDistAttribute,
                                                    pir::Attribute,
                                                    OperationDistAttrStorage> {
 public:
  using Base::Base;
  ProcessMeshAttribute process_mesh_attr() const;

  const std::vector<Attribute>& operands() const;
  pir::Attribute operand(uint32_t index) const { return operands().at(index); }
  uint32_t num_operands() const;

  const std::vector<Attribute>& results() const;

  pir::Attribute result(uint32_t index) const { return results().at(index); }

  uint32_t num_results() const;

  int64_t chunk_id() const;

  static OperationDistAttribute get(pir::IrContext* ctx,
                                    ProcessMeshAttribute mesh,
                                    const std::vector<Attribute>& operands,
                                    const std::vector<Attribute>& results,
                                    const int64_t& chunk_id = -1);

  static OperationDistAttribute get(pir::IrContext* ctx,
                                    const phi::distributed::ProcessMesh& mesh,
                                    const std::vector<Attribute>& operands,
                                    const std::vector<Attribute>& results,
                                    const int64_t& chunk_id = -1) {
    return get(
        ctx, ProcessMeshAttribute::get(ctx, mesh), operands, results, chunk_id);
  }

  static std::string name() { return "a_op_dist"; }
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ProcessMeshAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::PlacementsAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::TensorDistAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::OperationDistAttribute)
