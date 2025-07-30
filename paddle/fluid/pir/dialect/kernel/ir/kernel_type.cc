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

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"

namespace paddle::dialect {

pir::Type AllocatedDenseTensorType::prim_type() {
  return storage()->dense_tensor_type_;
}

const phi::Place& AllocatedDenseTensorType::place() const {
  return storage()->place_;
}

pir::Type AllocatedDenseTensorType::dtype() const {
  return storage()->dense_tensor_type_.dtype();
}

const phi::DDim& AllocatedDenseTensorType::dims() const {
  return storage()->dense_tensor_type_.dims();
}

phi::DataLayout AllocatedDenseTensorType::data_layout() const {
  return storage()->dense_tensor_type_.data_layout();
}

const phi::LegacyLoD& AllocatedDenseTensorType::lod() const {
  return storage()->dense_tensor_type_.lod();
}

size_t AllocatedDenseTensorType::offset() const {
  return storage()->dense_tensor_type_.offset();
}

pir::Type AllocatedSelectedRowsType::prim_type() {
  return storage()->selected_rows_type_;
}

const phi::Place& AllocatedSelectedRowsType::place() const {
  return storage()->place_;
}

pir::Type AllocatedSelectedRowsType::dtype() const {
  return storage()->selected_rows_type_.dtype();
}

const phi::DDim& AllocatedSelectedRowsType::dims() const {
  return storage()->selected_rows_type_.dims();
}

phi::DataLayout AllocatedSelectedRowsType::data_layout() const {
  return storage()->selected_rows_type_.data_layout();
}

const phi::LegacyLoD& AllocatedSelectedRowsType::lod() const {
  return storage()->selected_rows_type_.lod();
}

size_t AllocatedSelectedRowsType::offset() const {
  return storage()->selected_rows_type_.offset();
}

// AllocatedSparseCooTensorType
pir::Type AllocatedSparseCooTensorType::prim_type() {
  return storage()->sparsecoo_tensor_type_;
}

const phi::Place& AllocatedSparseCooTensorType::place() const {
  return storage()->place_;
}

const pir::Type AllocatedSparseCooTensorType::dtype() const {
  return storage()->sparsecoo_tensor_type_.dtype();
}

const phi::DDim& AllocatedSparseCooTensorType::dims() const {
  return storage()->sparsecoo_tensor_type_.dims();
}
const phi::DDim& AllocatedSparseCooTensorType::non_zero_dims() const {
  return storage()->sparsecoo_tensor_type_.non_zero_dims();
}
phi::DataLayout AllocatedSparseCooTensorType::data_layout() const {
  return storage()->sparsecoo_tensor_type_.data_layout();
}

pir::DenseTensorType AllocatedSparseCooTensorType::non_zero_indices() const {
  return storage()->sparsecoo_tensor_type_.non_zero_indices();
}

pir::DenseTensorType AllocatedSparseCooTensorType::non_zero_elements() const {
  return storage()->sparsecoo_tensor_type_.non_zero_elements();
}

bool AllocatedSparseCooTensorType::coalesced() const {
  return storage()->sparsecoo_tensor_type_.coalesced();
}

// AllocatedSparseCsrTensorType
pir::Type AllocatedSparseCsrTensorType::prim_type() {
  return storage()->sparsecsr_tensor_type_;
}

const phi::Place& AllocatedSparseCsrTensorType::place() const {
  return storage()->place_;
}

pir::Type AllocatedSparseCsrTensorType::dtype() const {
  return storage()->sparsecsr_tensor_type_.dtype();
}

const phi::DDim& AllocatedSparseCsrTensorType::dims() const {
  return storage()->sparsecsr_tensor_type_.dims();
}

phi::DataLayout AllocatedSparseCsrTensorType::data_layout() const {
  return storage()->sparsecsr_tensor_type_.data_layout();
}

pir::DenseTensorType AllocatedSparseCsrTensorType::non_zero_crows() const {
  return storage()->sparsecsr_tensor_type_.non_zero_crows();
}

pir::DenseTensorType AllocatedSparseCsrTensorType::non_zero_cols() const {
  return storage()->sparsecsr_tensor_type_.non_zero_cols();
}

pir::DenseTensorType AllocatedSparseCsrTensorType::non_zero_elements() const {
  return storage()->sparsecsr_tensor_type_.non_zero_elements();
}

// AllocatedDenseTensorArrayType
pir::Type AllocatedDenseTensorArrayType::prim_type() {
  return storage()->dense_tensor_array_type_;
}

const phi::Place& AllocatedDenseTensorArrayType::place() const {
  return storage()->place_;
}

const pir::Type& AllocatedDenseTensorArrayType::dtype() const {
  return storage()->dense_tensor_array_type_.dtype();
}

const pir::DDim& AllocatedDenseTensorArrayType::dims() const {
  return storage()->dense_tensor_array_type_.dims();
}

const phi::DataLayout& AllocatedDenseTensorArrayType::data_layout() const {
  return storage()->dense_tensor_array_type_.data_layout();
}

}  // namespace paddle::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedDenseTensorType)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedSelectedRowsType)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedSparseCooTensorType)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedSparseCsrTensorType)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedDenseTensorArrayType)
