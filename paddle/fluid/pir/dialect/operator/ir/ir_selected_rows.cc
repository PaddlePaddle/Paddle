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

#include "paddle/fluid/pir/dialect/operator/ir/ir_selected_rows.h"

#include <utility>
#include "paddle/common/enforce.h"

namespace paddle::dialect {
IrSelectedRows::IrSelectedRows(phi::DataType dtype,
                               const phi::DDim& dims,
                               phi::DataLayout layout,
                               LegacyLoD lod,
                               size_t offset)
    : dims_(dims),
      dtype_(dtype),
      layout_(layout),
      lod_(std::move(lod)),
      offset_(offset) {}

IrSelectedRows::IrSelectedRows(const IrSelectedRows& other)
    : TensorBase(other) {
  dims_ = other.dims();
  dtype_ = other.dtype();
  layout_ = other.layout();
  lod_ = other.lod();
  offset_ = other.offset();
}

IrSelectedRows& IrSelectedRows::operator=(const IrSelectedRows& other) {
  dims_ = other.dims();
  dtype_ = other.dtype();
  layout_ = other.layout();
  lod_ = other.lod();
  offset_ = other.offset();
  return *this;
}

IrSelectedRows& IrSelectedRows::operator=(IrSelectedRows&& other) noexcept {
  dims_ = other.dims();
  dtype_ = other.dtype();
  layout_ = other.layout();
  lod_ = other.lod();
  offset_ = other.offset();
  return *this;
}

int64_t IrSelectedRows::numel() const { return common::product(dims_); }

const phi::Place& IrSelectedRows::place() const {
  IR_THROW("Don't use IrSelectedRows::place method.");
}

void* IrSelectedRows::AllocateFrom(phi::Allocator* allocator,
                                   phi::DataType dtype,
                                   size_t requested_size,
                                   bool fake_alloc) {
  IR_THROW("Don't use IrSelectedRows::AllocateFrom method.");
}

}  // namespace paddle::dialect
