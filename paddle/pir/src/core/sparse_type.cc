// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pir/include/core/sparse_type.h"

namespace paddle {
namespace dialect {
pir::Type SparseCooTensorType::dtype() const { return storage()->dtype_; }

const SparseCooTensorType::Dim& SparseCooTensorType::dims() const {
  return storage()->dims_;
}

DataLayout SparseCooTensorType::data_layout() const {
  return storage()->layout_;
}

pir::DenseTensorType SparseCooTensorType::get_indices() const {
  return storage()->non_zero_indices_;
}

pir::DenseTensorType SparseCooTensorType::get_elements() const {
  return storage()->non_zero_elements_;
}

bool SparseCooTensorType::get_coalesced() const {
  return storage()->coalesced_;
}

bool SparseCooTensorType::classof(Type type) {
  if (type) {
    if (type.type_id() == type_id()) return true;
    if (auto wrap_type = type.dyn_cast<pir::WrapTypeInterface>()) {
      return classof(wrap_type.prim_type());
    }
  }
  return false;
}

SparseCooTensorType SparseCooTensorType::dyn_cast_impl(Type type) {
  if (type) {
    if (type.type_id() == type_id()) return SparseCooTensorType(type.storage());
    if (auto wrap_type = type.dyn_cast<pir::WrapTypeInterface>()) {
      return dyn_cast_impl(wrap_type.prim_type());
    }
  }
  return nullptr;
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SparseCooTensorType)
