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
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/type.h"

namespace paddle {
namespace dialect {

class DistDenseTensorTypeStorage;

class DistDenseTensorType
    : public pir::Type::
          TypeBase<DistDenseTensorType, pir::Type, DistDenseTensorTypeStorage> {
 public:
  using Base::Base;

  pir::DenseTensorType dense_tensor_type() const;
  TensorDistAttribute tensor_dist_attr() const;
  const common::DDim& global_ddim() const;
  const common::DDim& local_ddim() const { return dense_tensor_type().dims(); }
  Type dtype() const { return dense_tensor_type().dtype(); }
  DataLayout data_layout() const { return dense_tensor_type().data_layout(); }

  const phi::distributed::ProcessMesh& process_mesh() const {
    return tensor_dist_attr().process_mesh();
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

  static DistDenseTensorType get(pir::IrContext* ctx,
                                 pir::DenseTensorType dense_tensor_type,
                                 TensorDistAttribute tensor_dist_attr,
                                 const common::DDim& global_ddim);
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DistDenseTensorType)
