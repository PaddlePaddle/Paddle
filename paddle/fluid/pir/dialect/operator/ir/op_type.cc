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

#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"

namespace paddle::dialect {
const pir::Type& SelectedRowsType::dtype() const { return storage()->dtype_; }

const phi::DDim& SelectedRowsType::dims() const { return storage()->dims_; }

const phi::DataLayout& SelectedRowsType::data_layout() const {
  return storage()->layout_;
}

const phi::LegacyLoD& SelectedRowsType::lod() const { return storage()->lod_; }

const size_t& SelectedRowsType::offset() const { return storage()->offset_; }

bool SelectedRowsType::classof(Type type) {
  if (type) {
    if (type.type_id() == type_id()) return true;
    if (auto wrap_type = type.dyn_cast<pir::WrapTypeInterface>()) {
      return classof(wrap_type.prim_type());
    }
  }
  return false;
}

SelectedRowsType SelectedRowsType::dyn_cast_impl(Type type) {
  if (type) {
    if (type.type_id() == type_id()) return SelectedRowsType(type.storage());
    if (auto wrap_type = type.dyn_cast<pir::WrapTypeInterface>()) {
      return dyn_cast_impl(wrap_type.prim_type());
    }
  }
  return nullptr;
}

const pir::Type& DenseTensorArrayType::dtype() const {
  return storage()->dtype_;
}
const phi::DDim& DenseTensorArrayType::dims() const { return storage()->dims_; }

const phi::DataLayout& DenseTensorArrayType::data_layout() const {
  return storage()->layout_;
}

bool DenseTensorArrayType::classof(Type type) {
  if (type) {
    if (type.type_id() == type_id()) return true;
    if (auto wrap_type = type.dyn_cast<pir::WrapTypeInterface>()) {
      return classof(wrap_type.prim_type());
    }
  }
  return false;
}

DenseTensorArrayType DenseTensorArrayType::dyn_cast_impl(Type type) {
  if (type) {
    if (type.type_id() == type_id())
      return DenseTensorArrayType(type.storage());
    if (auto wrap_type = type.dyn_cast<pir::WrapTypeInterface>()) {
      return dyn_cast_impl(wrap_type.prim_type());
    }
  }
  return nullptr;
}

pir::Type SparseCooTensorType::dtype() const { return storage()->dtype_; }

const common::DDim& SparseCooTensorType::dims() const {
  return storage()->dims_;
}

const common::DDim& SparseCooTensorType::non_zero_dims() const {
  return storage()->non_zero_dims_;
}

common::DataLayout SparseCooTensorType::data_layout() const {
  return storage()->layout_;
}

pir::DenseTensorType SparseCooTensorType::non_zero_indices() const {
  return storage()->non_zero_indices_;
}

pir::DenseTensorType SparseCooTensorType::non_zero_elements() const {
  return storage()->non_zero_elements_;
}

bool SparseCooTensorType::coalesced() const { return storage()->coalesced_; }

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

pir::Type SparseCsrTensorType::dtype() const { return storage()->dtype_; }

const common::DDim& SparseCsrTensorType::dims() const {
  return storage()->dims_;
}

common::DataLayout SparseCsrTensorType::data_layout() const {
  return storage()->layout_;
}

pir::DenseTensorType SparseCsrTensorType::non_zero_crows() const {
  return storage()->non_zero_crows_;
}

pir::DenseTensorType SparseCsrTensorType::non_zero_cols() const {
  return storage()->non_zero_cols_;
}

pir::DenseTensorType SparseCsrTensorType::non_zero_elements() const {
  return storage()->non_zero_elements_;
}

bool SparseCsrTensorType::classof(Type type) {
  if (type) {
    if (type.type_id() == type_id()) return true;
    if (auto wrap_type = type.dyn_cast<pir::WrapTypeInterface>()) {
      return classof(wrap_type.prim_type());
    }
  }
  return false;
}

SparseCsrTensorType SparseCsrTensorType::dyn_cast_impl(Type type) {
  if (type) {
    if (type.type_id() == type_id()) return SparseCsrTensorType(type.storage());
    if (auto wrap_type = type.dyn_cast<pir::WrapTypeInterface>()) {
      return dyn_cast_impl(wrap_type.prim_type());
    }
  }
  return nullptr;
}
}  // namespace paddle::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SelectedRowsType)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DenseTensorArrayType)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SparseCooTensorType)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::SparseCsrTensorType)
