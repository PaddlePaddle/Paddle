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
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/builtin_attribute_storage.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/utils/flat_hash_map.h"

namespace paddle {
namespace dialect {
class ProcessMeshAttrStorage;
class TensorDistAttrStorage;

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
};

class TensorDistAttribute : public pir::AttrBase<TensorDistAttribute,
                                                 pir::Attribute,
                                                 TensorDistAttrStorage> {
 public:
  using Base::Base;
  ProcessMeshAttribute mesh_attr() const;
  const phi::distributed::ProcessMesh& process_mesh() const {
    return mesh_attr().process_mesh();
  }
  const std::vector<int64_t>& dims_mapping() const;

  // return vector of mesh dims on which the this tensor is partial on
  std::set<int64_t> partial_dims() const;

  const flat_hash_map<int64_t, phi::ReduceType>& partial_status() const;

  static TensorDistAttribute get(
      pir::IrContext* ctx,
      ProcessMeshAttribute mesh,
      const std::vector<int64_t>& dims_mapping,
      const flat_hash_map<int64_t, phi::ReduceType>& partial_status);
  static TensorDistAttribute get(
      pir::IrContext* ctx,
      const phi::distributed::ProcessMesh& mesh,
      const std::vector<int64_t>& dims_mapping,
      const flat_hash_map<int64_t, phi::ReduceType>& partial_status) {
    return get(ctx,
               ProcessMeshAttribute::get(ctx, mesh),
               dims_mapping,
               partial_status);
  }
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ProcessMeshAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::TensorDistAttribute)
