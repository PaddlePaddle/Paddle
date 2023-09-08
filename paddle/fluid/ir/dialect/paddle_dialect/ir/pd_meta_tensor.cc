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

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_meta_tensor.h"

#include "paddle/ir/core/enforce.h"

namespace paddle {
namespace dialect {
IrMetaTensor::IrMetaTensor(phi::DataType dtype,
                           const phi::DDim& dims,
                           phi::DataLayout layout,
                           const LoD& lod,
                           size_t offset)
    : dims_(dims), dtype_(dtype), layout_(layout), lod_(lod), offset_(offset) {}

IrMetaTensor::IrMetaTensor(const IrMetaTensor& other) {
  dims_ = other.dims();
  dtype_ = other.dtype();
  layout_ = other.layout();
  lod_ = other.lod();
  offset_ = other.offset();
}

IrMetaTensor& IrMetaTensor::operator=(const IrMetaTensor& other) {
  dims_ = other.dims();
  dtype_ = other.dtype();
  layout_ = other.layout();
  lod_ = other.lod();
  offset_ = other.offset();
  return *this;
}

IrMetaTensor& IrMetaTensor::operator=(IrMetaTensor&& other) noexcept {
  dims_ = std::move(other.dims());
  dtype_ = other.dtype();
  layout_ = other.layout();
  lod_ = std::move(other.lod());
  offset_ = other.offset();
  return *this;
}

int64_t IrMetaTensor::numel() const { return phi::product(dims_); }

const phi::Place& IrMetaTensor::place() const {
  IR_THROW("Don't use IrMetaTensor::place method.");
}

void* IrMetaTensor::AllocateFrom(phi::Allocator* allocator,
                                 phi::DataType dtype,
                                 size_t requested_size,
                                 bool fake_alloc) {
  IR_THROW("Don't use IrMetaTensor::AllocateFrom method.");
}

}  // namespace dialect
}  // namespace paddle
