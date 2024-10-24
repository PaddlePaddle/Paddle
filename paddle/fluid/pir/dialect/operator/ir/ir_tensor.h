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

#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/pir/include/core/builtin_type.h"

namespace paddle {
namespace dialect {

using LoD = std::vector<std::vector<size_t>>;

class IrTensor : public phi::TensorBase,
                 public phi::TypeInfoTraits<phi::TensorBase, IrTensor> {
 public:
  IrTensor() = default;

  IrTensor(phi::DataType dtype,
           const phi::DDim& dims,
           phi::DataLayout layout,
           LoD lod,
           size_t offset = 0);

  IrTensor(IrTensor&& other) = default;

  IrTensor(const IrTensor& other);

  IrTensor& operator=(const IrTensor& other);

  IrTensor& operator=(IrTensor&& other) noexcept;

  virtual ~IrTensor() = default;

 public:
  static const char* name() { return "IrTensor"; }

  int64_t numel() const override;

  const phi::DDim& dims() const noexcept override { return dims_; }

  void SetDims(const phi::DDim& dims) { dims_ = dims; }

  const phi::Place& place() const override;

  phi::DataType dtype() const noexcept override { return dtype_; }

  void SetDtype(phi::DataType dtype) { dtype_ = dtype; }

  phi::DataLayout layout() const noexcept override { return layout_; }

  void SetLayout(phi::DataLayout layout) { layout_ = layout; }

  const LoD& lod() const noexcept { return lod_; }

  void SetLod(LoD lod) { lod_ = lod; }

  size_t offset() const noexcept { return offset_; }

  bool valid() const noexcept override { return true; }

  bool has_allocation() const override {
    PADDLE_THROW(::common::errors::Unavailable(
        "`has_allocation` is only available at runtime"));
  }

  bool initialized() const override { return true; }

  void* AllocateFrom(phi::Allocator* allocator,
                     phi::DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false) override;

 private:
  phi::DDim dims_;
  phi::DataType dtype_{phi::DataType::FLOAT32};
  phi::DataLayout layout_{phi::DataLayout::NCHW};
  LoD lod_;
  size_t offset_{0};
};

inline pir::DenseTensorType CvtToDenseTensorType(const IrTensor& ir_tensor) {
  return pir::DenseTensorType::get(pir::IrContext::Instance(),
                                   TransToIrDataType(ir_tensor.dtype()),
                                   ir_tensor.dims(),
                                   ir_tensor.layout(),
                                   ir_tensor.lod(),
                                   ir_tensor.offset());
}

}  // namespace dialect
}  // namespace paddle
