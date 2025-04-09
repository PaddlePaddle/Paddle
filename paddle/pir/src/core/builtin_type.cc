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

#include "paddle/pir/include/core/builtin_type.h"

namespace pir {
std::vector<Type> VectorType::data() const { return storage()->GetAsKey(); }

pir::Type DenseTensorType::dtype() const { return storage()->dtype_; }

const DenseTensorType::Dim& DenseTensorType::dims() const {
  return storage()->dims_;
}

DataLayout DenseTensorType::data_layout() const { return storage()->layout_; }

const DenseTensorType::LegacyLoD& DenseTensorType::lod() const {
  return storage()->lod_;
}

size_t DenseTensorType::offset() const { return storage()->offset_; }

bool DenseTensorType::classof(Type type) {
  if (type) {
    if (type.type_id() == type_id()) return true;
    if (auto wrap_type = type.dyn_cast<WrapTypeInterface>()) {
      return classof(wrap_type.prim_type());
    }
  }
  return false;
}

DenseTensorType DenseTensorType::dyn_cast_impl(Type type) {
  if (type) {
    if (type.type_id() == type_id()) return DenseTensorType(type.storage());
    if (auto wrap_type = type.dyn_cast<WrapTypeInterface>()) {
      return dyn_cast_impl(wrap_type.prim_type());
    }
  }
  return nullptr;
}

}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::UndefinedType)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::UInt8Type)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Int8Type)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::VectorType)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::BFloat16Type)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Float16Type)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Float32Type)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Float64Type)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Int16Type)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Int32Type)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Int64Type)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::IndexType)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::BoolType)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Complex64Type)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Complex128Type)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Float8E4M3FNType)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::Float8E5M2Type)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::DenseTensorType)
