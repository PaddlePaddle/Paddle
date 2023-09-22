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

namespace paddle {
namespace dialect {

const phi::Place& AllocatedDenseTensorType::place() const {
  return storage()->place_;
}

const pir::Type& AllocatedDenseTensorType::dtype() const {
  return storage()->dense_tensor_type_.dtype();
}

const phi::DDim& AllocatedDenseTensorType::dims() const {
  return storage()->dense_tensor_type_.dims();
}

const phi::DataLayout& AllocatedDenseTensorType::data_layout() const {
  return storage()->dense_tensor_type_.data_layout();
}

const phi::LoD& AllocatedDenseTensorType::lod() const {
  return storage()->dense_tensor_type_.lod();
}

const size_t& AllocatedDenseTensorType::offset() const {
  return storage()->dense_tensor_type_.offset();
}

const phi::Place& AllocatedSelectedRowsType::place() const {
  return storage()->place_;
}

const pir::Type& AllocatedSelectedRowsType::dtype() const {
  return storage()->selected_rows_type_.dtype();
}

const phi::DDim& AllocatedSelectedRowsType::dims() const {
  return storage()->selected_rows_type_.dims();
}

const phi::DataLayout& AllocatedSelectedRowsType::data_layout() const {
  return storage()->selected_rows_type_.data_layout();
}

const phi::LoD& AllocatedSelectedRowsType::lod() const {
  return storage()->selected_rows_type_.lod();
}

const size_t& AllocatedSelectedRowsType::offset() const {
  return storage()->selected_rows_type_.offset();
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedDenseTensorType)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedSelectedRowsType)
