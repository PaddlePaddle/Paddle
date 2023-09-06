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

#pragma once

#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"

namespace paddle {
namespace dialect {

using LoD = std::vector<std::vector<size_t>>;

class IrMetaTensor : public phi::TensorBase,
                     public phi::TypeInfoTraits<phi::TensorBase, IrMetaTensor> {
 public:
  IrMetaTensor(phi::DataType dtype,
               const phi::DDim& dims,
               phi::DataLayout layout,
               const LoD& lod,
               size_t offset = 0);

  IrMetaTensor(IrMetaTensor&& other) = default;

  IrMetaTensor(const IrMetaTensor& other);

  IrMetaTensor& operator=(const IrMetaTensor& other);

  IrMetaTensor& operator=(IrMetaTensor&& other) noexcept;

  virtual ~IrMetaTensor() = default;

 public:
  static const char* name() { return "IrMetaTensor"; }

  int64_t numel() const override;

  const phi::DDim& dims() const noexcept override { return dims_; }

  const phi::Place& place() const override;

  phi::DataType dtype() const noexcept override { return dtype_; }

  phi::DataLayout layout() const noexcept override { return layout_; }

  const LoD& lod() const noexcept { return lod_; }

  size_t offset() const noexcept { return offset_; }

  bool valid() const noexcept override { return true; }

  bool initialized() const override { return true; }

  void* AllocateFrom(phi::Allocator* allocator,
                     phi::DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false) override;

 private:
  phi::DDim dims_;
  phi::DataType dtype_{phi::DataType::UNDEFINED};
  phi::DataLayout layout_{phi::DataLayout::NCHW};
  LoD lod_;
  size_t offset_{0};
};

}  // namespace dialect
}  // namespace paddle
