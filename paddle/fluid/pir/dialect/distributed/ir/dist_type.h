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

#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_interface.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/type.h"

namespace paddle {
namespace dialect {

class DistDenseTensorTypeStorage;

common::DDim InferLocalDDim(const common::DDim& global_ddim,
                            TensorDistAttribute dist_attr);

common::DDim InferGlobalDDim(const common::DDim& local_ddim,
                             TensorDistAttribute dist_attr);
class DistDenseTensorType
    : public pir::Type::TypeBase<DistDenseTensorType,
                                 pir::Type,
                                 DistDenseTensorTypeStorage,
                                 pir::WrapTypeInterface,
                                 DistTypeInterface> {
 public:
  using Base::Base;
  using LegacyLoD = pir::DenseTensorTypeStorage::LegacyLoD;

  static std::string name() { return "t_dist_dtensor"; }

  pir::DenseTensorType dense_tensor_type() const;
  TensorDistAttribute tensor_dist_attr() const;
  const common::DDim& global_ddim() const { return dense_tensor_type().dims(); }
  const common::DDim& local_ddim() const;
  Type dtype() const { return dense_tensor_type().dtype(); }
  DataLayout data_layout() const { return dense_tensor_type().data_layout(); }
  const LegacyLoD& lod() const { return dense_tensor_type().lod(); }
  size_t offset() const { return dense_tensor_type().offset(); }

  pir::DenseTensorType prim_type() { return dense_tensor_type(); }
  pir::DenseTensorType local_type() const;

  ProcessMeshAttribute process_mesh_attr() const {
    return tensor_dist_attr().process_mesh_attr();
  }
  const std::vector<int64_t>& dims_mapping() const {
    return tensor_dist_attr().dims_mapping();
  }
  std::set<int64_t> partial_dims() const {
    return tensor_dist_attr().partial_dims();
  }
  const flat_hash_map<int64_t, phi::ReduceType>& partial_status() const {
    return tensor_dist_attr().partial_status();
  }

  DistDenseTensorType CopyWithNewMesh(ProcessMeshAttribute mesh) {
    return get(ir_context(),
               dense_tensor_type(),
               tensor_dist_attr().CopyWithNewMesh(mesh));
  }

  DistDenseTensorType CopyWithNewDistAttr(TensorDistAttribute dist_attr) {
    return get(ir_context(), dense_tensor_type(), dist_attr);
  }

  static DistDenseTensorType get(pir::IrContext* ctx,
                                 pir::DenseTensorType dense_tensor_type,
                                 TensorDistAttribute tensor_dist_attr,
                                 const common::DDim& local_ddim);
  static DistDenseTensorType get(pir::IrContext* ctx,
                                 pir::DenseTensorType dense_tensor_type,
                                 TensorDistAttribute tensor_dist_attr) {
    if (!dense_tensor_type) return nullptr;
    auto local_ddim =
        InferLocalDDim(dense_tensor_type.dims(), tensor_dist_attr);
    return get(ctx, dense_tensor_type, tensor_dist_attr, local_ddim);
  }

  // return the replicated dist dense tensor type.
  static DistDenseTensorType get(pir::IrContext* ctx,
                                 pir::DenseTensorType dense_tensor_type,
                                 ProcessMeshAttribute process_mesh_attr) {
    auto& ddim = dense_tensor_type.dims();
    auto attr = TensorDistAttribute::get(
        ctx, process_mesh_attr, std::vector<int64_t>(ddim.size(), -1));
    return get(ctx, dense_tensor_type, attr, ddim);
  }
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DistDenseTensorType)
