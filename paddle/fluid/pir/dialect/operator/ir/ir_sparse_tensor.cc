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

#include "paddle/fluid/pir/dialect/operator/ir/ir_sparse_tensor.h"
#include "paddle/common/enforce.h"

namespace paddle {
namespace dialect {
IrSparseCooTensor::IrSparseCooTensor(phi::DataType dtype,
                                     const common::DDim& dims,
                                     common::DDim non_zero_dims,
                                     common::DataLayout layout,
                                     bool coalesced)
    : dims_(dims),
      non_zero_dims_(non_zero_dims),
      dtype_(dtype),
      layout_(layout),
      coalesced_(coalesced) {}

IrSparseCooTensor& IrSparseCooTensor::operator=(
    IrSparseCooTensor&& other) noexcept {
  dims_ = std::move(other.dims());
  non_zero_dims_ = std::move(other.non_zero_dims());
  dtype_ = other.dtype();
  layout_ = other.layout();
  coalesced_ = other.coalesced();
  return *this;
}

int64_t IrSparseCooTensor::numel() const { return common::product(dims_); }

const phi::Place& IrSparseCooTensor::place() const {
  IR_THROW("Don't use IrSparseCooTensor::place method.");
}

void* IrSparseCooTensor::AllocateFrom(phi::Allocator* allocator,
                                      phi::DataType dtype,
                                      size_t requested_size,
                                      bool fake_alloc) {
  IR_THROW("Don't use IrSparseCooTensor::AllocateFrom method.");
}

IrSparseCsrTensor::IrSparseCsrTensor(phi::DataType dtype,
                                     const common::DDim& dims,
                                     const common::DataLayout layout,
                                     pir::DenseTensorType non_zero_crows,
                                     pir::DenseTensorType non_zero_cols,
                                     pir::DenseTensorType non_zero_elements)
    : dims_(dims),
      dtype_(dtype),
      layout_(layout),
      non_zero_crows_(non_zero_crows),
      non_zero_cols_({non_zero_cols}),
      non_zero_elements_(non_zero_elements) {}

IrSparseCsrTensor& IrSparseCsrTensor::operator=(
    IrSparseCsrTensor&& other) noexcept {
  dims_ = std::move(other.dims());
  dtype_ = other.dtype();
  layout_ = other.layout();
  non_zero_crows_ = other.non_zero_crows();
  non_zero_cols_ = other.non_zero_cols();
  non_zero_elements_ = other.non_zero_elements();
  return *this;
}

int64_t IrSparseCsrTensor::numel() const { return common::product(dims_); }

const phi::Place& IrSparseCsrTensor::place() const {
  IR_THROW("Don't use IrSparseCsrTensor::place method.");
}

void* IrSparseCsrTensor::AllocateFrom(phi::Allocator* allocator,
                                      phi::DataType dtype,
                                      size_t requested_size,
                                      bool fake_alloc) {
  IR_THROW("Don't use IrSparseCsrTensor::AllocateFrom method.");
}

}  // namespace dialect
}  // namespace paddle
