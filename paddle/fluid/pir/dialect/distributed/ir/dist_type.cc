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

#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/distributed/ir/type_storage.h"

namespace paddle {
namespace dialect {

pir::DenseTensorType DistDenseTensorType::dense_tensor_type() const {
  return storage()->dense_tensor_type;
}

TensorDistAttribute DistDenseTensorType::tensor_dist_attr() const {
  return storage()->tensor_dist_attr;
}

const common::DDim& DistDenseTensorType::global_ddim() const {
  return storage()->global_ddim;
}

DistDenseTensorType DistDenseTensorType::get(
    pir::IrContext* ctx,
    pir::DenseTensorType dense_tensor_type,
    TensorDistAttribute tensor_dist_attr,
    const common::DDim& global_ddim) {
  return Base::get(ctx, dense_tensor_type, tensor_dist_attr, global_ddim);
}
}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DistDenseTensorType)
